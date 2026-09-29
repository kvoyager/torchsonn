"""RBF neuron: learnable Gaussian bumps over a neuron's input tuple, k-means
initialized.

Every other family fits a *global* basis over its inputs (monomials, orthogonal
polynomials): a weight multiplies a column that is nonzero everywhere. This
family fits a *local* one. Per neuron over `dim` inputs, `M` Gaussian bumps

    d_m(u) = |u - c_m|^2 / (2 s_m^2),   g_m(u) = exp(-d_m(u))

with the inputs `u` standardized per slot (mean / std from the layer's input
pass) unless `standardize=False`, the centers `c_m` and the widths `s_m`
parameters of the neuron (unless `learn_centers` / `learn_widths` are off, in
which case they are buffers). The width is carried in log form relative to its
initial value and bounded to a band by a smooth squash,
`s_m = width0_m * exp(L * tanh(rho_m / L))` with `L = ln band`, so it stays
positive and a bump can neither collapse onto one row nor blur into the linear
part. (A hard clamp was tried first: an LBFGS line-search step can jump the
raw parameter past the band in one move, and the clamp's zero gradient then
strands it there; tanh keeps a gradient everywhere.) The design row is

    [ phi_1(u) .. phi_M(u),  u_1 .. u_dim (linear),  1 (only if not normalize) ]

where `phi_m = softmax_m(-d)` when `normalize=True` (a partition of unity, so
the constant is in the span and dropped; the softmax form keeps a row far from
every center from underflowing to 0/0) and `phi_m = g_m` otherwise. With the
defaults (M=16, normalized, linear) a pair neuron has num_w = 18 weights plus
32 center coordinates and 16 log-widths, all fitted jointly by the trainer's
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
Widths start at `width` times the mean distance to the two nearest other
centers, floored so duplicated centers (tied or binary slots) never give a
zero width.

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

# Quantile grid of the `grid` placement: K points per slot, K = M^(1/dim).
_GRID_Q_LO, _GRID_Q_HI = 0.1, 0.9
# Width floor in units of the slot std (1 in standardized units).
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
                 width: float = 1.0,
                 learn_centers: bool = True,
                 learn_widths: bool = True,
                 width_band: float = 4.0,
                 normalize: bool = True,
                 linear: bool = True,
                 standardize: bool = True) -> None:
        if int(centers) < 2:
            raise ValueError(f"centers must be >= 2, got {centers}")
        if float(width) <= 0.0:
            raise ValueError(f"width must be > 0, got {width}")
        if float(width_band) <= 1.0:
            raise ValueError(f"width_band must be > 1, got {width_band}")
        placement = str(placement).lower()
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {list(PLACEMENTS)}, got {placement!r}")
        if int(dim) < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        self.num_centers = int(centers)
        self.placement = placement
        self.width = float(width)
        self.learn_centers = bool(learn_centers)
        self.learn_widths = bool(learn_widths)
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
        # them): centers on the diagonal of [-1.5, 1.5]^dim, unit widths,
        # identity stats. Correctly shaped so the module is usable as built.
        centers0 = torch.linspace(-1.5, 1.5, m).view(1, m, 1).expand(n, m, d).clone()
        if self.learn_centers:
            self.centers = nn.Parameter(centers0)
        else:
            self.register_buffer("centers", centers0)
        log_width0 = torch.zeros(n, m)
        if self.learn_widths:
            self.log_width = nn.Parameter(log_width0)
        else:
            self.register_buffer("log_width", log_width0)
        self.register_buffer("width0", torch.ones(n, m))
        self.register_buffer("in_mean", torch.zeros(n, d))
        self.register_buffer("in_std", torch.ones(n, d))
        # Where the centers started (after the input pass), for the survivor
        # report; plain CPU tensor, not state, pruned alongside the neurons.
        self._centers_start: torch.Tensor | None = None
        # Mini-batch k-means working state, only between fit_input_sample
        # (stream=True) and finish_input_stream.
        self._stream_counts: torch.Tensor | None = None
        self._stream_rows = 0
        self._stream_start: torch.Tensor | None = None

        self.params_metadata_names.extend([
            "num_centers", "placement", "width", "learn_centers", "learn_widths",
            "width_band", "normalize", "linear", "standardize",
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
            placement=metadata.get("placement", "kmeans"),
            width=metadata.get("width", 1.0),
            learn_centers=metadata.get("learn_centers", True),
            learn_widths=metadata.get("learn_widths", True),
            width_band=metadata.get("width_band", 4.0),
            normalize=metadata.get("normalize", True),
            linear=metadata.get("linear", True),
            standardize=metadata.get("standardize", True),
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

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-slot standardization of gathered slot inputs (B, num_neurons, dim)
        eager / (B, 1, dim) under vmap; identity when standardize is off."""
        if not self.standardize:
            return x
        return (x - self.in_mean) / self.in_std

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        u = self._standardize(x)
        # (B, n, 1, dim) - (n, M, dim) -> (B, n, M, dim); under vmap
        # (B, 1, 1, dim) - (M, dim) -> (B, 1, M, dim). Same broadcasting rule
        # as the orthogonal families' squash stats.
        diff = u.unsqueeze(-2) - self.centers
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
        # Always: the standardization, and the width floor in raw units.
        return True

    def fit_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
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
        return True

    @property
    def needs_input_stream(self) -> bool:
        return self.placement == "kmeans"

    def fit_input_sample(self, x_sample: torch.Tensor, stream: bool = False,
                         seed: int | None = None, iters: int = 20) -> None:
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
        centers = self._kmeans_pp(u, gen)
        if stream:
            # Start only; the trainer's streaming pass refines the centers.
            with torch.no_grad():
                self.centers.copy_(centers.to(dtype=self.centers.dtype))
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
        if self._stream_counts is None:
            return
        u = self._slots(x_batch)                                    # (n, b, dim)
        with torch.no_grad():
            c = self.centers.detach().to(dtype=u.dtype)
            assign = self._assign(u, c)                             # (n, b)
            sums, counts = self._cluster_sums(u, assign, self.num_centers)
            self._stream_counts += counts.to(torch.float64)
            self._stream_rows += u.shape[1]
            # Sculley's update with the per-center rate 1 / n_m:
            #   c_m += (1 / n_m) * sum_{i in batch, a_i = m} (u_i - c_m)
            step = (sums - counts.unsqueeze(-1) * c) / self._stream_counts.clamp(min=1.0).unsqueeze(-1).to(u.dtype)
            self.centers.add_(step.to(dtype=self.centers.dtype))

    def finish_input_stream(self) -> None:
        if self._stream_counts is None:
            return
        counts = self._stream_counts
        start = self._stream_start
        rows = self._stream_rows
        self._stream_counts = None
        self._stream_start = None
        self._stream_rows = 0
        centers = self.centers.detach().clone()
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
        n, b, d = u.shape
        sums = torch.zeros((n, m, d), dtype=u.dtype, device=u.device)
        counts = torch.zeros((n, m), dtype=u.dtype, device=u.device)
        sums.scatter_add_(1, assign.unsqueeze(-1).expand(n, b, d), u)
        counts.scatter_add_(1, assign, torch.ones_like(assign, dtype=u.dtype))
        return sums, counts

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
        """Write the placed centers, reset the widths from the local spacing,
        and log one line."""
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
        width0 = torch.maximum(self.width * h, _WIDTH_FLOOR * sigma_bar)
        with torch.no_grad():
            self.centers.copy_(centers.to(dtype=self.centers.dtype))
            self.log_width.zero_()
            self.width0.copy_(width0.to(dtype=self.width0.dtype))
        self._centers_start = centers.detach().to("cpu", torch.float32)

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
        logger.info("%s: %d centers per neuron placed by %s%s%s; width0 median %.3g",
                    self.get_short_name(), m, note, time_txt, size_txt, width0.median().item())

    # ------------------------------------------------------------------
    # reports, prune, names
    # ------------------------------------------------------------------
    def fit_report(self) -> str | None:
        """One line after selection: how far the survivors' centers moved from
        their placed start and where their width scales sit, so a fit that
        never moves a center, or pins widths at the band, is visible."""
        if self._centers_start is None or self.num_neurons == 0:
            return None
        moved = (self.centers.detach().to("cpu", torch.float32) - self._centers_start).norm(dim=-1)
        with torch.no_grad():
            scale = self.width_scales().to("cpu", torch.float32)
        # "at the band": the smooth bound is 99% saturated.
        at_band = (torch.tanh(self.log_width.detach() / self._log_band).abs() >= 0.99).float().mean().item()
        return (f"{self.get_short_name()}: {self.num_neurons} survivors; center movement "
                f"median {moved.median().item():.3g}, max {moved.max().item():.3g} "
                f"(standardized units); width scale min {scale.min().item():.2f}, "
                f"max {scale.max().item():.2f}, {at_band:.0%} at the band")

    def _prune_extra(self, idxs: torch.Tensor) -> None:
        dev = idxs.to(device=self.centers.device)
        for name in ("centers", "log_width"):
            t = getattr(self, name)
            kept = t.detach().index_select(0, dev)
            if isinstance(t, nn.Parameter):
                setattr(self, name, nn.Parameter(kept))
            else:
                setattr(self, name, kept)  # rebinding replaces the registered buffer
        self.width0 = self.width0.index_select(0, dev)
        self.in_mean = self.in_mean.index_select(0, dev)
        self.in_std = self.in_std.index_select(0, dev)
        if self._centers_start is not None:
            self._centers_start = self._centers_start.index_select(0, idxs.to("cpu"))
        super()._prune_extra(idxs)

    def get_short_name(self) -> str:
        suffix = f"x{self.dim}" if self.dim != 2 else ""
        return f"RBF{self.num_centers}{suffix}"

    def get_name(self) -> str:
        learn = []
        if self.learn_centers:
            learn.append("centers")
        if self.learn_widths:
            learn.append("widths")
        learn_txt = " + ".join(learn) if learn else "fixed"
        start = "k-means" if self.placement == "kmeans" else "quantile-grid"
        parts = [f"Gaussian RBF ({self.num_centers} {start} centers, learnable {learn_txt}, "
                 f"width {self.width:g} x local spacing"]
        if self.normalize:
            parts.append(", normalized")
        parts.append(f") over {'standardized ' if self.standardize else ''}{self.dim} inputs")
        if self.linear:
            parts.append(" + linear")
        return "".join(parts)
