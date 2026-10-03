"""RBF neuron: learnable Gaussian bumps over a neuron's input tuple, k-means
initialized.

Every other family fits a *global* basis over its inputs (monomials, orthogonal
polynomials): a weight multiplies a column that is nonzero everywhere. This
family fits a *local* one. Per neuron over `dim` inputs, `M` Gaussian bumps

    d_m(u) = |u - c_m|^2 / (2 s_m^2),   g_m(u) = exp(-d_m(u))

with the inputs `u` standardized per slot (mean / std from the layer's input
pass) unless `standardize=False`, the centers `c_m` and the widths `s_m`
learnable (unless `learn_centers` / `learn_widths` are off). Both are carried
relative to their k-means start and bounded by a smooth squash, so the fit can
move a bump across its neighbourhood but never off the data:

    c_m = c0_m + r_m * tanh(|delta_m| / r_m) * delta_m / |delta_m|,   r_m = center_radius * h_m
    s_m = width0_m * exp(L * tanh(rho_m / L)),                          L = ln(width_band)

`h_m` is the mean distance from the placed center to its two nearest other
centers (the local spacing), `delta_m` (dim-vector) and `rho_m` the learnable
parameters, both zero at the start. `center_radius=None` removes the center
bound (`c_m = c0_m + delta_m`), kept for optimizer experiments: the
per-member LBFGS keeps a separate curvature history per parameter tensor with
a fixed step, so as a bump loses mass its gradient and curvature shrink
together and the quasi-Newton step along that direction ejects the center;
on California housing 37-68% of the survivors' centers ended more than 5 std
from the data, dead cells with zero mass and zero gradient. A hard clamp is no
better (an LBFGS step jumps the raw parameter past it and the zero gradient
strands it there); tanh keeps a gradient everywhere and the bound is exact.
The design row is

    [ phi_1(u) .. phi_M(u),  u_1 .. u_dim (linear),  1 (only if not normalize) ]

where `phi_m = softmax_m(-d)` when `normalize=True` (a partition of unity, so
the constant is in the span and dropped; the softmax form keeps a row far from
every center from underflowing to 0/0) and `phi_m = g_m` otherwise. With the
defaults (M=16, normalized, linear) a pair neuron has num_w = 18 weights plus
32 center displacements and 16 log-widths, all fitted jointly by the trainer's
vmapped candidate fit, which takes every named parameter with in_dims=0.

Initialization happens in the trainer's per-layer input pass
(`Trainer.fit_layer_inputs`): the module receives the layer's input moments
(`fit_input_stats`), then a row sample of the layer input (`fit_input_sample`)
on which it runs k-means++ and, in sample mode, Lloyd's iterations batched
over the candidates; in stream mode (large data) only the k-means++ start is
taken from the sample and the centers are then refined by mini-batch k-means
over the whole split (`stream_input_batch`, Sculley's per-center 1/n rate),
finishing with `finish_input_stream`. `placement="grid"` replaces k-means by
the product of per-slot quantile grids (deterministic, `M` must be K^dim).
Widths start at `width` times the local spacing, radii at `center_radius`
times it, both floored so duplicated centers (tied or binary slots) never give
a zero width or radius.

Every per-neuron tensor carries the leading `num_neurons` axis, so it is
vmapped with the ensemble, index-selected by `prune`, and checkpointed in the
state dict; a restored model needs no re-placement.
"""
import itertools
import logging
import math
import time

import torch
from torch import nn

from torchsonn.neurons.base import ActivationLike, BaseTupleNeuron

logger = logging.getLogger(__name__)

PLACEMENTS = ("kmeans", "grid")
# How the k-means start is chosen: rows at the quantiles of the projection
# on the tuple's first principal axis (deterministic, data-driven), or the
# seeded k-means++ draw.
SEEDINGS = ("pca_quantiles", "kmeans++")

# Quantile grid of the `grid` placement: K points per slot, K = M^(1/dim).
_GRID_Q_LO, _GRID_Q_HI = 0.1, 0.9
# Width / radius floor in units of the slot std (1 in standardized units).
_WIDTH_FLOOR = 0.05


class RBFNeuron(BaseTupleNeuron):
    """Gaussian RBF neuron over a `dim`-tuple of inputs, see the module docstring."""

    def __init__(self,
                 num_feat: int,
                 num_src_feat: int,
                 activation: ActivationLike,
                 layer_index: int,
                 start_index: int,
                 max_neuron_models: int | None = None,
                 init_method: str = "xavier",
                 dim: int = 2,
                 centers: int = 16,
                 placement: str = "kmeans",
                 seeding: str = "pca_quantiles",
                 width: float = 1.0,
                 learn_centers: bool = True,
                 learn_widths: bool = True,
                 center_radius: float | None = 2.0,
                 width_band: float = 4.0,
                 normalize: bool = True,
                 linear: bool = True,
                 standardize: bool = True) -> None:
        if int(centers) < 2:
            raise ValueError(f"centers must be >= 2, got {centers}")
        if float(width) <= 0.0:
            raise ValueError(f"width must be > 0, got {width}")
        if center_radius is not None and float(center_radius) <= 0.0:
            raise ValueError(f"center_radius must be > 0 or null (unbounded), got {center_radius}")
        if float(width_band) <= 1.0:
            raise ValueError(f"width_band must be > 1, got {width_band}")
        placement = str(placement).lower()
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {list(PLACEMENTS)}, got {placement!r}")
        seeding = str(seeding).lower()
        if seeding not in SEEDINGS:
            raise ValueError(f"seeding must be one of {list(SEEDINGS)}, got {seeding!r}")
        if int(dim) < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        self.num_centers = int(centers)
        self.placement = placement
        self.seeding = seeding
        self.width = float(width)
        self.learn_centers = bool(learn_centers)
        self.learn_widths = bool(learn_widths)
        self.center_radius = None if center_radius is None else float(center_radius)
        self.width_band = float(width_band)
        self.normalize = bool(normalize)
        self.linear = bool(linear)
        self.standardize = bool(standardize)
        self.dim = int(dim)
        if self.placement == "grid":
            k = round(self.num_centers ** (1.0 / self.dim))
            if k ** self.dim != self.num_centers:
                raise ValueError(
                    f"placement='grid' needs centers = K^dim for an integer K; "
                    f"got centers={self.num_centers}, dim={self.dim}")
            self._grid_k = k
        # M cells + the linear part + a constant only when the cells do not
        # already sum to one. Must be set before super().__init__ allocates
        # `weight` from it.
        self.num_w = self.num_centers + (self.dim if self.linear else 0) + (0 if self.normalize else 1)
        super().__init__(
            num_feat,
            num_src_feat,
            activation,
            layer_index,
            start_index,
            max_neuron_models=max_neuron_models,
            init_method=init_method,
            dim=self.dim,
        )
        n, m, d = self.num_neurons, self.num_centers, self.dim
        # Placeholders until the input pass runs (or load_state_dict fills
        # them): start centers on the diagonal of [-1.5, 1.5]^dim, unit widths
        # and radii, zero displacements, identity stats. Correctly shaped so
        # the module is usable as built.
        self.register_buffer("centers0", torch.linspace(-1.5, 1.5, m).view(1, m, 1).expand(n, m, d).clone())
        self.register_buffer("radius", torch.ones(n, m))
        shift0 = torch.zeros(n, m, d)
        if self.learn_centers:
            self.center_shift = nn.Parameter(shift0)
        else:
            self.register_buffer("center_shift", shift0)
        log_width0 = torch.zeros(n, m)
        if self.learn_widths:
            self.log_width = nn.Parameter(log_width0)
        else:
            self.register_buffer("log_width", log_width0)
        self.register_buffer("width0", torch.ones(n, m))
        self.register_buffer("in_mean", torch.zeros(n, d))
        self.register_buffer("in_std", torch.ones(n, d))
        # Mini-batch k-means working state, only between fit_input_sample
        # (stream=True) and finish_input_stream.
        self._stream_counts: torch.Tensor | None = None
        self._stream_rows = 0
        self._stream_start: torch.Tensor | None = None

        self.params_metadata_names.extend([
            "num_centers", "placement", "seeding", "width", "learn_centers", "learn_widths",
            "center_radius", "width_band", "normalize", "linear", "standardize",
        ])

    @classmethod
    def _construct_from_metadata(cls, metadata: dict) -> "RBFNeuron":
        return cls(
            num_feat=metadata["num_feat"],
            num_src_feat=metadata["num_src_feat"],
            activation=metadata["activation"],
            layer_index=metadata["layer_index"],
            start_index=metadata["start_index"],
            max_neuron_models=metadata["src_idxs"].shape[0],
            dim=metadata["dim"],
            centers=metadata["num_centers"],
            placement=metadata["placement"],
            seeding=metadata["seeding"],
            width=metadata["width"],
            learn_centers=metadata["learn_centers"],
            learn_widths=metadata["learn_widths"],
            center_radius=metadata["center_radius"],
            width_band=metadata["width_band"],
            normalize=metadata["normalize"],
            linear=metadata["linear"],
            standardize=metadata["standardize"],
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    @property
    def _log_band(self) -> float:
        return math.log(self.width_band)

    def width_scales(self) -> torch.Tensor:
        """`exp(L * tanh(log_width / L))`, the factor on `width0`, always
        inside (1 / band, band)."""
        band = self._log_band
        return torch.exp(band * torch.tanh(self.log_width / band))

    def widths(self) -> torch.Tensor:
        """Effective bump widths `width0 * width_scales()`, shape
        (num_neurons, M) eager or (M,) under vmap."""
        return self.width0 * self.width_scales()

    def displacements(self) -> torch.Tensor:
        """Bounded displacement of every center from its start: the raw shift
        `center_shift` rescaled so its norm is `r * tanh(|shift| / r)`, i.e.
        the direction is kept and the length squashed below the radius `r`.
        The norm is computed as sqrt(|shift|^2 + eps^2) so the map is smooth
        at zero shift (the start), where its Jacobian is the identity. Shape
        (num_neurons, M, dim) eager or (M, dim) under vmap. The raw shift
        when `center_radius` is None."""
        shift = self.center_shift
        if self.center_radius is None:
            return shift
        r = self.radius.unsqueeze(-1)
        norm = torch.sqrt((shift * shift).sum(dim=-1, keepdim=True) + 1e-12)
        return shift * (r * torch.tanh(norm / r) / norm)

    def centers(self) -> torch.Tensor:
        """Effective centers `centers0 + displacements()`."""
        return self.centers0 + self.displacements()

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-slot standardization of gathered slot inputs (B, num_neurons, dim)
        eager / (B, 1, dim) under vmap; identity when standardize is off."""
        if not self.standardize:
            return x
        return (x - self.in_mean) / self.in_std

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        """Design row: the M bump activations of the standardized inputs, then
        the inputs themselves when `linear` is on, then a constant 1 when
        `normalize` is off.
        """
        u = self._standardize(x)
        # (B, n, 1, dim) - (n, M, dim) -> (B, n, M, dim); under vmap
        # (B, 1, 1, dim) - (M, dim) -> (B, 1, M, dim). Same broadcasting rule
        # as the orthogonal families' squash stats.
        diff = u.unsqueeze(-2) - self.centers()
        s = self.widths()
        d = (diff * diff).sum(dim=-1) / (2.0 * s * s)
        phi = torch.softmax(-d, dim=-1) if self.normalize else torch.exp(-d)
        parts = [phi]
        if self.linear:
            parts.append(u)
        if not self.normalize:
            parts.append(torch.ones((*u.shape[:-1], 1), device=u.device, dtype=u.dtype))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # input pass: statistics
    # ------------------------------------------------------------------
    @property
    def needs_input_stats(self) -> bool:
        """Always True: the inputs are standardized and the width floor is in raw units."""
        # Always: the standardization, and the width / radius floor in raw units.
        return True

    def fit_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store the per-slot input mean and std used for standardization.

        `mean` / `std` are per layer-input feature; indexing them with
        `src_idxs` gives each candidate's slot statistics. A constant slot
        (std <= 1e-8) gets std 1.
        """
        idx = self.src_idxs.to(device=mean.device)
        slot_mean = mean[idx]
        slot_std = std[idx]
        # A constant slot gets std 1 (sklearn's convention) so nothing divides
        # by zero; the slot then contributes a constant to every bump.
        slot_std = torch.where(slot_std <= 1e-8, torch.ones_like(slot_std), slot_std)
        with torch.no_grad():
            self.in_mean.copy_(slot_mean.to(dtype=self.in_mean.dtype, device=self.in_mean.device))
            self.in_std.copy_(slot_std.to(dtype=self.in_std.dtype, device=self.in_std.device))

    def _slots(self, x: torch.Tensor) -> torch.Tensor:
        """Layer input (N, num_feat) -> standardized slots (num_neurons, N, dim),
        candidate-major for the batched k-means."""
        n = x.shape[0]
        gathered = torch.index_select(x, 1, self.src_idxs.view(-1).to(device=x.device)).view(n, -1, self.dim)
        return self._standardize(gathered.to(dtype=self.in_mean.dtype)).transpose(0, 1).contiguous()

    # ------------------------------------------------------------------
    # input pass: placement
    # ------------------------------------------------------------------
    @property
    def needs_input_sample(self) -> bool:
        """Always True: the centers are placed on a sample of the layer input."""
        return True

    @property
    def needs_input_stream(self) -> bool:
        """True for k-means placement, which can be refined by a streaming pass."""
        return self.placement == "kmeans"

    def fit_input_sample(self, x_sample: torch.Tensor, stream: bool = False,
                         seed: int | None = None, iters: int = 20) -> None:
        """Place the centers on a sample of the layer input.

        Grid placement builds the quantile grid. k-means placement seeds with
        k-means++ or PCA quantiles, then runs `iters` Lloyd iterations on the
        sample, batched over the candidates. With `stream=True` only the seeds
        are stored, and `stream_input_batch` refines them over the whole split.
        """
        t0 = time.perf_counter()
        u = self._slots(x_sample)                                   # (n, N, dim)
        n_rows = u.shape[1]
        if n_rows == 0:
            logger.warning("%s: empty input sample, centers keep their placeholders", self.get_short_name())
            return
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) if seed is not None else 0)
        if self.placement == "grid":
            centers = self._grid_centers(u)
            self._finish_placement(centers, u, f"grid on {n_rows} rows", t0)
            return
        centers = (self._kmeans_pp(u, gen) if self.seeding == "kmeans++"
                   else self._pca_quantile_seeds(u, self.num_centers))
        if stream:
            # Start only; the trainer's streaming pass refines the centers.
            with torch.no_grad():
                self.centers0.copy_(centers.to(dtype=self.centers0.dtype))
            self._stream_counts = torch.zeros((self.num_neurons, self.num_centers),
                                              dtype=torch.float64, device=centers.device)
            self._stream_rows = 0
            self._stream_start = centers.detach().clone()
            logger.info("%s: k-means++ start on %d sampled rows in %.2fs; streaming pass follows",
                        self.get_short_name(), n_rows, time.perf_counter() - t0)
            return
        centers = self._lloyd(u, centers, int(iters))
        self._finish_placement(centers, u, f"k-means on {n_rows} rows, {int(iters)} Lloyd iterations", t0)

    def stream_input_batch(self, x_batch: torch.Tensor) -> None:
        """Mini-batch k-means update of the centers from one batch of layer input.

        Each center moves toward the mean of its assigned rows with the
        per-center rate 1 / n_m (Sculley). No-op unless `fit_input_sample` ran
        with `stream=True`.
        """
        if self._stream_counts is None:
            return
        u = self._slots(x_batch)                                    # (n, b, dim)
        with torch.no_grad():
            c = self.centers0.to(dtype=u.dtype)
            assign = self._assign(u, c)                             # (n, b)
            sums, counts = self._cluster_sums(u, assign, self.num_centers)
            self._stream_counts += counts.to(torch.float64)
            self._stream_rows += u.shape[1]
            # Sculley's update with the per-center rate 1 / n_m:
            #   c_m += (1 / n_m) * sum_{i in batch, a_i = m} (u_i - c_m)
            step = (sums - counts.unsqueeze(-1) * c) / self._stream_counts.clamp(min=1.0).unsqueeze(-1).to(u.dtype)
            self.centers0.add_(step.to(dtype=self.centers0.dtype))

    def finish_input_stream(self) -> None:
        """Finish the streaming pass: fix the centers, set widths and radii, and
        log how far the centers moved from their seeds.
        """
        if self._stream_counts is None:
            return
        counts = self._stream_counts
        start = self._stream_start
        rows = self._stream_rows
        self._stream_counts = None
        self._stream_start = None
        self._stream_rows = 0
        centers = self.centers0.detach().clone()
        moved = (centers.to(start.dtype) - start).norm(dim=-1)      # (n, M)
        note = (f"mini-batch k-means over {rows} streamed rows; center movement since the start "
                f"median {moved.median().item():.3g}, max {moved.max().item():.3g}")
        self._finish_placement(centers, None, note, None, counts=counts)

    # -- pieces ----------------------------------------------------------
    @staticmethod
    def _assign(u: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Nearest center per row: u (n, b, dim), centers (n, M, dim) -> (n, b)."""
        diff = u.unsqueeze(2) - centers.unsqueeze(1)                # (n, b, M, dim)
        return (diff * diff).sum(dim=-1).argmin(dim=-1)

    @staticmethod
    def _cluster_sums(u: torch.Tensor, assign: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-cluster row sums and counts as a one-hot matrix product.
        `scatter_add_` was used first; on CUDA its summation order varies
        between runs, and k-means has enough near-tied assignments that the
        rounding difference flipped rows and the runs drifted apart. A
        batched matmul is evaluated deterministically."""
        onehot = torch.nn.functional.one_hot(assign, m).to(dtype=u.dtype)       # (n, b, m)
        sums = torch.einsum("nbm,nbd->nmd", onehot, u)
        counts = onehot.sum(dim=1)
        return sums, counts

    @staticmethod
    def _pca_quantile_seeds(u: torch.Tensor, m: int | None = None) -> torch.Tensor:
        """Deterministic, data-driven k-means start: project each candidate's
        rows onto the first principal axis of its tuple and take the rows at
        the quantiles (j + 0.5) / M of that projection. Every seed is a real
        row, the seeds span the joint distribution's main direction, and
        Lloyd spreads them across the others. u: (n, N, dim) -> (n, M, dim)."""
        n, n_rows, d = u.shape
        centred = u - u.mean(dim=1, keepdim=True)
        cov = torch.einsum("nbd,nbe->nde", centred, centred) / max(1, n_rows)
        # eigh returns ascending eigenvalues; the last vector is the first axis.
        _, vecs = torch.linalg.eigh(cov.to(torch.float64))
        axis = vecs[:, :, -1].to(dtype=u.dtype)                                   # (n, d)
        # An eigenvector's sign is arbitrary; make the largest component
        # positive so the seed order (and any downstream reproduction) is fixed.
        lead = axis.gather(1, axis.abs().argmax(dim=1, keepdim=True))
        axis = axis * torch.sign(lead).clamp(min=0).mul(2).sub(1)
        proj = torch.einsum("nbd,nd->nb", centred, axis)                          # (n, N)
        order = torch.sort(proj, dim=1, stable=True).indices
        q = ((torch.arange(m, device=u.device, dtype=torch.float64) + 0.5) / m * n_rows).floor().long().clamp(max=n_rows - 1)
        picks = order[:, q]                                                       # (n, M)
        return torch.gather(u, 1, picks.unsqueeze(-1).expand(n, m, d)).clone()

    def _row_chunk(self, n_rows: int) -> int:
        # Keep the (n, chunk, M, dim) distance tensor around 64M floats.
        per_row = max(1, self.num_neurons * self.num_centers * self.dim)
        return max(1, min(n_rows, (1 << 26) // per_row))

    def _kmeans_pp(self, u: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Batched k-means++ start: (n, N, dim) -> (n, M, dim). Sampling on the
        CPU generator so the start is reproducible for a fixed seed."""
        n, n_rows, _ = u.shape
        ar = torch.arange(n, device=u.device)
        first = torch.randint(0, n_rows, (n,), generator=gen).to(device=u.device)
        chosen = [u[ar, first]]                                     # (n, dim)
        d2 = ((u - chosen[0].unsqueeze(1)) ** 2).sum(dim=-1)       # (n, N)
        for _ in range(1, self.num_centers):
            p = d2.detach().to("cpu", torch.float64).clamp(min=0.0)
            total = p.sum(dim=1, keepdim=True)
            p = torch.where(total > 0, p / total.clamp(min=1e-300), torch.full_like(p, 1.0 / n_rows))
            idx = torch.multinomial(p, 1, generator=gen).squeeze(1).to(device=u.device)
            c = u[ar, idx]
            chosen.append(c)
            d2 = torch.minimum(d2, ((u - c.unsqueeze(1)) ** 2).sum(dim=-1))
        return torch.stack(chosen, dim=1)

    def _lloyd(self, u: torch.Tensor, centers: torch.Tensor, iters: int) -> torch.Tensor:
        n, n_rows, d = u.shape
        m = self.num_centers
        chunk = self._row_chunk(n_rows)
        for _ in range(iters):
            sums = torch.zeros((n, m, d), dtype=u.dtype, device=u.device)
            counts = torch.zeros((n, m), dtype=u.dtype, device=u.device)
            for start in range(0, n_rows, chunk):
                ub = u[:, start:start + chunk]
                s, c = self._cluster_sums(ub, self._assign(ub, centers), m)
                sums += s
                counts += c
            new = sums / counts.clamp(min=1.0).unsqueeze(-1)
            # An empty cluster keeps its previous center.
            centers = torch.where(counts.unsqueeze(-1) > 0, new, centers)
        return centers

    def _grid_centers(self, u: torch.Tensor) -> torch.Tensor:
        """Product of per-slot quantile grids: K points per slot between the
        10th and 90th percentile, K^dim centers."""
        n, n_rows, d = u.shape
        k = self._grid_k
        qs = torch.linspace(_GRID_Q_LO, _GRID_Q_HI, k, dtype=torch.float64, device=u.device)
        # torch.quantile caps its input size; one slot at a time keeps it small.
        per_slot = [torch.quantile(u[:, :, s].to(torch.float64), qs, dim=1) for s in range(d)]  # each (k, n)
        grid = torch.tensor(list(itertools.product(range(k), repeat=d)), device=u.device)       # (M, d)
        cols = [per_slot[s][grid[:, s]].transpose(0, 1) for s in range(d)]                      # each (n, M)
        return torch.stack(cols, dim=-1).to(dtype=u.dtype)

    def _finish_placement(self, centers: torch.Tensor, u: torch.Tensor | None, note: str,
                          t0: float | None, counts: torch.Tensor | None = None) -> None:
        """Write the placed centers as the start, reset the displacements,
        widths and radii from the local spacing, and log one line."""
        n, m, _ = centers.shape
        # h_m: mean distance from c_m to its two nearest other centers.
        cc = (centers.unsqueeze(2) - centers.unsqueeze(1)).norm(dim=-1)            # (n, M, M)
        cc = cc + torch.diag(torch.full((m,), float("inf"), device=cc.device, dtype=cc.dtype))
        k = min(2, m - 1)
        h = cc.topk(k, dim=-1, largest=False).values.mean(dim=-1)                  # (n, M)
        if self.standardize:
            sigma_bar = torch.ones((n, 1), dtype=h.dtype, device=h.device)
        else:
            sigma_bar = self.in_std.mean(dim=-1, keepdim=True).to(dtype=h.dtype, device=h.device)
        floor = _WIDTH_FLOOR * sigma_bar
        width0 = torch.maximum(self.width * h, floor)
        radius = torch.maximum((self.center_radius if self.center_radius is not None else 2.0) * h, floor)
        with torch.no_grad():
            self.centers0.copy_(centers.to(dtype=self.centers0.dtype))
            self.radius.copy_(radius.to(dtype=self.radius.dtype))
            self.center_shift.zero_()
            self.log_width.zero_()
            self.width0.copy_(width0.to(dtype=self.width0.dtype))

        if counts is None and u is not None:
            counts = torch.zeros((n, m), dtype=u.dtype, device=u.device)
            chunk = self._row_chunk(u.shape[1])
            for start in range(0, u.shape[1], chunk):
                ub = u[:, start:start + chunk]
                counts += self._cluster_sums(ub, self._assign(ub, centers), m)[1]
        size_txt = ""
        if counts is not None:
            c = counts.to(torch.float64)
            size_txt = (f"; cluster size min {int(c.min().item())}, median {int(c.median().item())}, "
                        f"{int((c == 0).sum().item())} empty of {n * m}")
        time_txt = f" in {time.perf_counter() - t0:.2f}s" if t0 is not None else ""
        logger.info("%s: %d centers per neuron placed by %s%s%s; width0 median %.3g, radius median %.3g",
                    self.get_short_name(), m, note, time_txt, size_txt,
                    width0.median().item(), radius.median().item())

    # ------------------------------------------------------------------
    # reports, prune, names
    # ------------------------------------------------------------------
    def fit_report(self) -> str | None:
        """One line after selection: how far the survivors' centers moved from
        their placed start (and how many sit at their radius) and where their
        width scales sit, so a fit that never moves a center, or pins
        everything at its bound, is visible."""
        if self.num_neurons == 0:
            return None
        with torch.no_grad():
            disp = self.displacements().to("cpu", torch.float32)
            r = self.radius.to("cpu", torch.float32)
            scale = self.width_scales().to("cpu", torch.float32)
        moved = disp.norm(dim=-1)
        # "at the bound": 99% of the radius / band used up.
        at_band = (torch.tanh(self.log_width.detach() / self._log_band).abs() >= 0.99).float().mean().item()
        if self.center_radius is None:
            radius_txt = f", unbounded ({(moved > 5.0).float().mean().item():.0%} beyond 5 std)"
        else:
            at_radius = (moved >= 0.99 * r).float().mean().item()
            radius_txt = f", {at_radius:.0%} at their radius (median radius {r.median().item():.3g})"
        return (f"{self.get_short_name()}: {self.num_neurons} survivors; center movement "
                f"median {moved.median().item():.3g}, max {moved.max().item():.3g} "
                f"(standardized units){radius_txt}; width scale min {scale.min().item():.2f}, "
                f"max {scale.max().item():.2f}, {at_band:.0%} at the band")

    def _prune_extra(self, idxs: torch.Tensor) -> None:
        dev = idxs.to(device=self.centers0.device)
        for name in ("center_shift", "log_width"):
            t = getattr(self, name)
            kept = t.detach().index_select(0, dev)
            if isinstance(t, nn.Parameter):
                setattr(self, name, nn.Parameter(kept))
            else:
                setattr(self, name, kept)  # rebinding replaces the registered buffer
        for name in ("centers0", "radius", "width0", "in_mean", "in_std"):
            setattr(self, name, getattr(self, name).index_select(0, dev))
        super()._prune_extra(idxs)

    def get_short_name(self) -> str:
        """Return e.g. 'RBF8', or 'RBF8x3' for 3 inputs."""
        suffix = f"x{self.dim}" if self.dim != 2 else ""
        return f"RBF{self.num_centers}{suffix}"

    def get_name(self) -> str:
        """Describe the centers, their placement, what is learnable, width and inputs."""
        learn = []
        if self.learn_centers:
            learn.append("centers, unbounded" if self.center_radius is None
                         else f"centers within {self.center_radius:g} x local spacing")
        if self.learn_widths:
            learn.append(f"widths within 1/{self.width_band:g}..{self.width_band:g} x start")
        learn_txt = "learnable " + ", ".join(learn) if learn else "fixed centers and widths"
        start = ("k-means from PCA quantiles" if self.seeding == "pca_quantiles" else "k-means++") \
            if self.placement == "kmeans" else "quantile-grid"
        parts = [f"Gaussian RBF ({self.num_centers} {start} centers, {learn_txt}, "
                 f"width {self.width:g} x local spacing"]
        if self.normalize:
            parts.append(", normalized")
        parts.append(f") over {'standardized ' if self.standardize else ''}{self.dim} inputs")
        if self.linear:
            parts.append(" + linear")
        return "".join(parts)
