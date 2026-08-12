"""Input normalization onto [-1, 1] for orthogonal-polynomial neurons.

Legendre P_k and Chebyshev T_k are orthogonal — and bounded by 1 — only on
[-1, 1]. Feed them a StandardScaler output and the tails blow the basis up
(T_k(x) ~ (2x)^k), wrecking the per-neuron least-squares conditioning. The
usual fix is `tanh` (what `squash=True` in neurons/orthopoly.py does), but tanh
spends its useful slope on the bulk of the data: at 1 sigma it is already at
0.76 and visibly curved, so the *typical* samples — the ones the fit should
resolve — get compressed together before the polynomial basis ever sees them.

`SigmaSquashNorm` splits the job in two:

  * Core, |x - mean| <= n_sigma * std: a straight linear rescale onto
    [-core_range, +core_range]. The bulk of the distribution passes through
    undistorted, so the Legendre expansion keeps its full resolution exactly
    where the data lives.
  * Tail, beyond n_sigma: a saturating map that carries [n_sigma*std, +inf)
    onto [core_range, 1) — outliers are bounded, never clipped, and never leave
    the orthogonality domain.

The tail is a plain rational function, not tanh: no exp, no transcendentals,
~6 flops and one divide, and it stays exact in float32 arbitrarily far out.

Tail construction
-----------------
Write s = (x - mean) / (n_sigma * std), so the core is |s| <= 1 and the map is
y = a*s there, with a = core_range. For the tail put t = |s| - 1 >= 0 and
define the *gap* to saturation g(t) = 1 - |y|:

    g(t) = (p + q*t) / (1 + r*t)^3,    p = 1 - a,  q = 2a,  r = a / (1 - a)

Those three coefficients are what you get by matching the core at the junction
to second order — g(0) = 1 - a, g'(0) = -a, g''(0) = 0 — which makes the whole
map C2: value, slope *and* curvature are continuous at +/-n_sigma, and the map
is odd so it is smooth through the origin too. C1 alone would be enough for
gradient descent, but torchsonn's Newton / Newton-LM / L-BFGS optimizers
consume curvature, and a jump in the second derivative shows up there as noise
in the Hessian; the cubic denominator buys C2 for one extra multiply.

The remaining properties fall out of p, q, r > 0:
  * g > 0 everywhere, so |y| < 1 and the output never leaves the orthogonality
    domain no matter how extreme the input.
  * g'(t) = -(a + 2qr*t) / (1 + r*t)^4 < 0, so g decreases monotonically and y
    is strictly increasing — the map is invertible and order-preserving, no
    two distinct inputs collapse onto the same value.
  * g = O(t^-2), a polynomial approach to saturation rather than tanh's
    exponential one. That is the practically important difference: the gap to
    1 stays representable far longer, so outliers keep their ordering instead
    of being flushed to a common +/-1. In float32 this map rounds to exactly
    1.0 only past ~2.7e3 sigma (gradient underflows past ~1.7e3); tanh gets
    there at 9 sigma. In float64 the figures are ~6e7 and ~4e7 sigma. Beyond
    those points |y| == 1 exactly and the gradient is 0 — outside any
    plausible data range, but it is a rounding limit, not a strict bound.

With the defaults (n_sigma=2, core_range=0.75) the map reads:

    0 sigma -> 0.000     3 sigma -> 0.936     6 sigma -> 0.991
    1 sigma -> 0.375     4 sigma -> 0.973    10 sigma -> 0.997
    2 sigma -> 0.750     5 sigma -> 0.984       inf   -> 1.000

Usage
-----
    norm = SigmaSquashNorm.from_data(x_train)      # per-feature mean/std
    u = norm(x)                                    # in (-1, 1), ready for P_k

Pair it with `squash=False` on the orthogonal neurons — the layer has already
done the squashing, and tanh on top would undo the linear core.
"""
from collections.abc import Sequence

import torch

from torchsonn.modules.base import SONNModule


class SigmaSquashNorm(SONNModule):
    """Standardize by mean/std, then squash onto (-1, 1) with a rational tail.

    Args:
        mean: Per-feature centers. Scalar, or any shape broadcastable against
            the trailing axes of the input (typically `[num_features]`).
        std: Per-feature scales, same shape as `mean`. Must be non-negative;
            entries at or below `eps` are treated as constant features and
            given a unit scale, so they map to a constant 0.
        n_sigma: Half-width of the linear core, in standard deviations.
            Inputs within `mean +/- n_sigma * std` are mapped linearly onto
            `+/-core_range`; everything beyond is squashed by the rational
            tail. Larger values keep more of the distribution linear at the
            cost of resolution in the core.
        core_range: Output magnitude reached at exactly `n_sigma`. Must lie in
            (0, 1) — at 1 the tail would have no room left to saturate into.
        eps: Threshold below which a `std` entry counts as a constant feature.
    """

    def __init__(self,
                 mean: float | Sequence[float] | torch.Tensor,
                 std: float | Sequence[float] | torch.Tensor,
                 n_sigma: float = 2.0,
                 core_range: float = 0.75,
                 eps: float = 1e-8) -> None:
        super().__init__()
        if n_sigma <= 0:
            raise ValueError(f"n_sigma must be > 0, got {n_sigma}")
        if not 0.0 < core_range < 1.0:
            raise ValueError(f"core_range must lie in (0, 1), got {core_range}")

        self.eps = float(eps)
        mean_t, std_t = self._sanitize_stats(mean, std, self.eps)

        # Buffers, not Parameters: these are dataset statistics, not something
        # to train. Registering them still moves them with .to(device) and
        # round-trips them through state_dict.
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

        self.n_sigma = float(n_sigma)
        self.core_range = float(core_range)
        # Both knobs change what the layer computes without changing any tensor
        # shape, so a checkpoint has to carry them — see SONNModule.state_dict.
        self.params_metadata_names.extend(["n_sigma", "core_range"])

    @staticmethod
    def _sanitize_stats(
        mean: float | Sequence[float] | torch.Tensor,
        std: float | Sequence[float] | torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate a (mean, std) pair and neutralize constant features."""
        mean_t = torch.as_tensor(mean, dtype=torch.get_default_dtype()).clone()
        std_t = torch.as_tensor(std, dtype=torch.get_default_dtype()).clone()
        if mean_t.shape != std_t.shape:
            raise ValueError(
                f"mean and std must have the same shape, got {tuple(mean_t.shape)} "
                f"and {tuple(std_t.shape)}"
            )
        if bool((std_t < 0).any()):
            raise ValueError("std must be non-negative")
        # A constant feature has std == 0, which `from_data` will hand us for
        # any degenerate column. Substituting 1.0 (sklearn's convention) rather
        # than a small floor is what keeps that harmless: the feature maps to
        # y = 0 for every sample, a constant column the neuron's own bias term
        # absorbs. A tiny floor would instead divide float noise by 1e-8 and
        # fling the column out to +/-1 at random.
        std_t = torch.where(std_t <= eps, torch.ones_like(std_t), std_t)
        return mean_t, std_t

    @torch.no_grad()
    def set_stats(self,
                  mean: float | Sequence[float] | torch.Tensor,
                  std: float | Sequence[float] | torch.Tensor) -> None:
        """Refit the statistics in place, keeping the buffers' shape and device.

        Used to calibrate an already-constructed layer once the data that will
        flow through it is known — see `BaseOrthogonalNeuron.fit_squash`, which
        calls this per layer during training. Copies in place rather than
        rebinding so any vmap/functional_call view of the buffers stays valid.
        """
        mean_t, std_t = self._sanitize_stats(mean, std, self.eps)
        if mean_t.shape != self.mean.shape:
            raise ValueError(
                f"expected stats of shape {tuple(self.mean.shape)}, "
                f"got {tuple(mean_t.shape)}"
            )
        self.mean.copy_(mean_t)
        self.std.copy_(std_t)

    @classmethod
    def from_data(cls,
                  x: torch.Tensor,
                  dim: int = 0,
                  **kwargs: float) -> "SigmaSquashNorm":
        """Build a layer from the per-feature statistics of `x`.

        Fit on the training split only — reusing it for validation/inference is
        the point, since the squash must be the same map every time.
        """
        x = torch.as_tensor(x)
        # torch.std defaults to the unbiased (ddof=1) estimate, unlike
        # numpy/sklearn's ddof=0; the gap is O(1/n) and irrelevant to the squash.
        return cls(mean=x.mean(dim=dim), std=x.std(dim=dim), **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.core_range
        # Tail coefficients, recomputed per call (three scalar ops, free next
        # to the tensor work) so they can never go stale against a core_range
        # restored from a checkpoint.
        p = 1.0 - a
        q = 2.0 * a
        r = a / (1.0 - a)

        s = (x - self.mean) / (self.std * self.n_sigma)
        abs_s = s.abs()

        t = (abs_s - 1.0).clamp(min=0.0)
        denom = 1.0 + r * t
        gap = (p + q * t) / (denom * denom * denom)

        # sign(s) * (1 - gap), written as s / |s| to keep the gradient right:
        # torch.sign has zero derivative, which would zero out the tail's
        # gradient contribution. |s| >= 1 wherever this branch is selected, and
        # the clamp keeps the *discarded* core-branch value finite — torch.where
        # evaluates both sides, and a 0/0 here would backpropagate NaN into the
        # core. The clamp is a no-op on the branch that actually survives.
        tail = s * (1.0 - gap) / abs_s.clamp(min=1.0)

        return torch.where(abs_s <= 1.0, a * s, tail)

    def extra_repr(self) -> str:
        return (f"n_sigma={self.n_sigma}, core_range={self.core_range}, "
                f"shape={tuple(self.mean.shape)}")
