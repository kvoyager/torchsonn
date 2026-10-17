"""MLP neuron: a one-hidden-layer network over a neuron's input tuple.

Every other family is linear in its weights over a fixed basis. This one fits
its own features: per neuron over `dim` inputs `u`, two optional per-slot input
transforms, in this order,

    standardize:  u_i -> (u_i - mean_i) / std_i        (stats from the input pass)
    squash:       sigma -> SigmaSquashNorm onto (-1, 1) (stats from the input pass)
                  tanh  -> tanh(.)                      (stateless)

give `z` (`z = u` with both off, the default). The squash is the one the
orthogonal families use. With `standardize` on, the sigma squash is calibrated
on the standardized slot (mean 0, std 1), so the composition equals the sigma
squash alone. The expanded input, `F` entries,

    f(z) = [ z_1 .. z_d,  z_i*z_j for i < j (cov),  z_i^2 (squares) ]

feeds one hidden layer of `H` units (`hidden`, default 2*dim + 1),

    h = sigma(W1 f + b1),    W1: (H, F),  b1: (H)

with `sigma` the `hidden_activation` (SiLU by default), and the output is

    y = act( w0 + v . f + a . h )

where `v . f` is the skip term (`skip`, on by default) and `act` the neuron's
`activation`. The readout `[w0, v, a]` is the module's `weight`, so
`num_w = 1 + F*skip + H` and `get_args` returns the row `[1, f, h]`: the base
`forward` (weighted sum, vmap squeeze, activation) applies unchanged. `W1` and
`b1` are the parameters `hidden_weight (num_neurons, H, F)` and
`hidden_bias (num_neurons, H)`; the trainer's vmapped candidate fit takes every
named parameter with in_dims=0, so they are fitted with the readout. Ridge
reaches the readout only.

With `skip` on and `a = 0` the neuron is the polynomial of `z` over the same
terms (the linear neuron, or with `cov` and `squares` the quadratic one), so
the hidden layer is a learned correction on top of it.
"""
import itertools

import torch
from torch import nn

from torchsonn.modules import SigmaSquashNorm
from torchsonn.neurons.base import _ACTIVATIONS, ActivationLike, BaseTupleNeuron

SQUASH_METHODS = ("sigma", "tanh")


class MLPNeuron(BaseTupleNeuron):
    """One-hidden-layer MLP neuron over a `dim`-tuple of inputs, see the module docstring."""

    def __init__(self,
                 num_feat: int,
                 num_src_feat: int,
                 activation: ActivationLike,
                 layer_index: int,
                 start_index: int,
                 max_neuron_models: int | None = None,
                 init_method: str = "xavier",
                 dim: int = 2,
                 hidden: int | None = None,
                 cov: bool = False,
                 squares: bool = False,
                 skip: bool = True,
                 hidden_activation: str = "silu",
                 standardize: bool = False,
                 squash: bool = False,
                 squash_method: str = "sigma",
                 squash_n_sigma: float = 2.0,
                 squash_core_range: float = 0.75) -> None:
        if int(dim) < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        self.dim = int(dim)
        self.hidden = 2 * self.dim + 1 if hidden is None else int(hidden)
        if self.hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        # A name, not a callable: the value round-trips through checkpoint
        # metadata, and a module instance would not rebuild from it.
        if not isinstance(hidden_activation, str) or hidden_activation.lower() not in _ACTIVATIONS:
            raise ValueError(
                f"hidden_activation must be one of {sorted(_ACTIVATIONS)}, got {hidden_activation!r}"
            )
        self.hidden_activation = hidden_activation.lower()
        self.cov = bool(cov)
        self.squares = bool(squares)
        self.skip = bool(skip)
        self.standardize = bool(standardize)
        self.squash = bool(squash)
        self.squash_method = str(squash_method).lower()
        if self.squash_method not in SQUASH_METHODS:
            raise ValueError(
                f"squash_method must be one of {list(SQUASH_METHODS)}, got {squash_method!r}"
            )
        self.squash_n_sigma = float(squash_n_sigma)
        self.squash_core_range = float(squash_core_range)
        # Slot pairs of the cov terms, fixed at construction so get_args has no
        # data-dependent control flow under vmap.
        self._cov_pairs = list(itertools.combinations(range(self.dim), 2)) if self.cov else []
        self.num_feat_expanded = self.dim + len(self._cov_pairs) + (self.dim if self.squares else 0)
        # Must be set before super().__init__ allocates `weight` from it.
        self.num_w = 1 + (self.num_feat_expanded if self.skip else 0) + self.hidden
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
        n = self.num_neurons
        self.hidden_act = _ACTIVATIONS[self.hidden_activation]()
        self.hidden_weight = nn.Parameter(torch.empty(n, self.hidden, self.num_feat_expanded))
        self.hidden_bias = nn.Parameter(torch.empty(n, self.hidden))
        self._reset_hidden()
        # Per-slot statistics carry the leading `num_neurons` axis, so they are
        # vmapped with the ensemble and index-selected by prune. Identity until
        # the input pass runs; only registered when their option is on.
        if self.standardize:
            self.register_buffer("in_mean", torch.zeros(n, self.dim))
            self.register_buffer("in_std", torch.ones(n, self.dim))
        self.squash_norm: SigmaSquashNorm | None = None
        if self.squash and self.squash_method == "sigma":
            self.squash_norm = SigmaSquashNorm(
                mean=torch.zeros(n, self.dim),
                std=torch.ones(n, self.dim),
                n_sigma=self.squash_n_sigma,
                core_range=self.squash_core_range,
            )

        self.params_metadata_names.extend([
            "hidden", "cov", "squares", "skip", "hidden_activation", "standardize",
            "squash", "squash_method", "squash_n_sigma", "squash_core_range",
        ])

    @classmethod
    def _construct_from_metadata(cls, metadata: dict) -> "MLPNeuron":
        return cls(
            num_feat=metadata["num_feat"],
            num_src_feat=metadata["num_src_feat"],
            activation=metadata["activation"],
            layer_index=metadata["layer_index"],
            start_index=metadata["start_index"],
            max_neuron_models=metadata["src_idxs"].shape[0],
            dim=metadata["dim"],
            hidden=metadata["hidden"],
            cov=metadata["cov"],
            squares=metadata["squares"],
            skip=metadata["skip"],
            hidden_activation=metadata["hidden_activation"],
            standardize=metadata["standardize"],
            squash=metadata["squash"],
            squash_method=metadata["squash_method"],
            squash_n_sigma=metadata["squash_n_sigma"],
            squash_core_range=metadata["squash_core_range"],
        )

    def reset_parameters(self) -> None:
        """Re-initialize the readout with `init_method` and the hidden layer
        with per-neuron xavier weights and zero biases."""
        super().reset_parameters()
        # The base __init__ calls this before the hidden layer exists.
        if hasattr(self, "hidden_weight"):
            self._reset_hidden()

    def _reset_hidden(self) -> None:
        # Xavier bound of one neuron's (H, F) matrix. nn.init.xavier_uniform_
        # on the stacked (num_neurons, H, F) tensor would treat the neuron axis
        # as fan-out and H as fan-in.
        bound = (6.0 / (self.hidden + self.num_feat_expanded)) ** 0.5
        with torch.no_grad():
            self.hidden_weight.uniform_(-bound, bound)
            self.hidden_bias.zero_()

    # ------------------------------------------------------------------
    # input pass
    # ------------------------------------------------------------------
    @property
    def needs_input_stats(self) -> bool:
        """True when the inputs are standardized or sigma-squashed."""
        return self.standardize or self.squash_norm is not None

    def fit_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set the per-slot statistics of the standardization and the sigma squash.

        `mean` / `std` are per layer-input feature; indexing them with
        `src_idxs` gives each candidate's slot statistics. A constant slot
        (std <= 1e-8) gets std 1. Under `standardize` the sigma squash sees
        standardized slots, so its statistics are mean 0, std 1.
        """
        idx = self.src_idxs.to(device=mean.device)
        slot_mean = mean[idx]
        slot_std = std[idx]
        if self.standardize:
            slot_std = torch.where(slot_std <= 1e-8, torch.ones_like(slot_std), slot_std)
            with torch.no_grad():
                self.in_mean.copy_(slot_mean.to(dtype=self.in_mean.dtype, device=self.in_mean.device))
                self.in_std.copy_(slot_std.to(dtype=self.in_std.dtype, device=self.in_std.device))
        if self.squash_norm is not None:
            if self.standardize:
                self.squash_norm.set_stats(torch.zeros_like(slot_mean), torch.ones_like(slot_std))
            else:
                self.squash_norm.set_stats(slot_mean, slot_std)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize and squash the gathered slot inputs, each when on."""
        if self.standardize:
            x = (x - self.in_mean) / self.in_std
        if self.squash:
            if self.squash_method == "sigma":
                x = self.squash_norm(x)
            else:
                x = torch.tanh(x)
        return x

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Expanded input `f(z)` (..., F) of gathered slot inputs (..., dim)."""
        z = self._transform(x)
        parts = [z]
        for i, j in self._cov_pairs:
            parts.append((z[..., i] * z[..., j]).unsqueeze(-1))
        if self.squares:
            parts.append(z * z)
        return torch.cat(parts, dim=-1)

    def hidden_units(self, f: torch.Tensor) -> torch.Tensor:
        """Hidden activations (B, num_neurons, H) of expanded inputs (B, num_neurons, F)."""
        w1, b1 = self.hidden_weight, self.hidden_bias
        # Inside the trainer's vmap the neuron axis is mapped away: W1 is (H, F)
        # and the slot inputs are (B, 1, F), so restore a singleton neuron axis.
        if w1.dim() == 2:
            w1, b1 = w1.unsqueeze(0), b1.unsqueeze(0)
        return self.hidden_act(torch.einsum("bnf,nhf->bnh", f, w1) + b1)

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        """Readout row `[1, f, h]` (`f` only with `skip`) of gathered slot
        inputs (B, num_neurons, dim)."""
        f = self.expand(x)
        parts = [torch.ones((*f.shape[:-1], 1), device=f.device, dtype=f.dtype)]
        if self.skip:
            parts.append(f)
        parts.append(self.hidden_units(f))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def fit_report(self) -> str | None:
        """One line after selection: how strongly the survivors read their
        hidden units against the skip path, by mean absolute readout weight."""
        if self.num_neurons == 0:
            return None
        w = self.weight.detach().abs().to("cpu", torch.float32)
        hidden_w = w[:, -self.hidden:].mean().item()
        txt = f"{self.get_short_name()}: {self.num_neurons} survivors; mean |hidden readout| {hidden_w:.3g}"
        if self.skip:
            skip_w = w[:, 1:1 + self.num_feat_expanded].mean().item()
            txt += f", mean |skip readout| {skip_w:.3g}"
        return txt

    def _prune_extra(self, idxs: torch.Tensor) -> None:
        dev = idxs.to(device=self.hidden_weight.device)
        self.hidden_weight = nn.Parameter(self.hidden_weight.detach().index_select(0, dev))
        self.hidden_bias = nn.Parameter(self.hidden_bias.detach().index_select(0, dev))
        if self.standardize:
            # Rebinding the attribute replaces the registered buffer.
            self.in_mean = self.in_mean.index_select(0, dev)
            self.in_std = self.in_std.index_select(0, dev)
        if self.squash_norm is not None:
            self.squash_norm.mean = self.squash_norm.mean.index_select(0, dev)
            self.squash_norm.std = self.squash_norm.std.index_select(0, dev)
        super()._prune_extra(idxs)

    def get_short_name(self) -> str:
        """Return e.g. 'MLP5', or 'MLP7x3' for 3 inputs."""
        suffix = f"x{self.dim}" if self.dim != 2 else ""
        return f"MLP{self.hidden}{suffix}"

    def get_name(self) -> str:
        """Describe the hidden layer, the expanded input, the skip path and the input transforms."""
        terms = ["inputs"]
        if self.cov:
            terms.append("pairwise products")
        if self.squares:
            terms.append("squares")
        transforms = []
        if self.standardize:
            transforms.append("standardized")
        if self.squash:
            transforms.append(f"{self.squash_method}-squashed")
        over = f"{' '.join(transforms)} " if transforms else ""
        skip = ", skip readout" if self.skip else ""
        return (f"MLP ({self.hidden} {self.hidden_activation} units) over {over}{self.dim} inputs, "
                f"expanded to {' + '.join(terms)}{skip}")
