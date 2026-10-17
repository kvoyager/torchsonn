"""Structured sparsity over the neurons of a pool: hard-concrete gates and group lasso.

Both remove whole neurons rather than single coefficients. Plain L1 on the
weights zeroes individual coefficients and leaves every neuron in place;
here each neuron is one unit that is either kept or dropped.

- `HardConcreteGate`: one learnable gate per neuron in [0, 1], with a
  differentiable expected-L0 penalty (Louizos, Welling and Kingma,
  "Learning Sparse Neural Networks through L0 Regularization", ICLR 2018).
- `group_lasso_penalty` / `group_soft_threshold`: the sum over neurons of
  the L2 norm of each neuron's parameters, and its proximal step, which
  sets whole groups to exactly zero.
- `diversity_loss`: the mean squared correlation between the outputs of
  the neurons that are kept, weighted by their gates, to push the kept
  neurons apart.
"""
import math
from collections.abc import Sequence

import torch
from torch import nn

# Floor under a squared group norm before the square root: the gradient of
# sqrt is infinite at 0, and a group the proximal step has zeroed would
# otherwise turn the penalty's gradient into NaN.
_TINY = 1e-24


class HardConcreteGate(nn.Module):
    """One gate per neuron in [0, 1], learned through an expected-L0 penalty.

    Each gate has one parameter, `log_alpha`. In training mode `forward`
    draws the stretched and clamped concrete gate, which is exactly 0 or
    exactly 1 with positive probability and continuous in between; in eval
    mode it returns the deterministic gate. `expected_l0` is the expected
    number of nonzero gates, a closed form differentiable in `log_alpha`, so
    `loss + lam * gate.expected_l0()` trades accuracy against the number of
    neurons in use.

    Parameters
    ----------
    num_gates : int
        Number of gates, one per neuron.
    beta : float
        Temperature of the concrete distribution. Default 0.66.
    gamma, zeta : float
        The interval (gamma, zeta) the sigmoid is stretched to before the
        clamp to [0, 1]; gamma < 0 < 1 < zeta. Defaults -0.1 and 1.1.
    init_open : float
        Starting probability, in (0, 1), of the underlying sigmoid sample;
        `log_alpha` starts at its logit. The default 0.9 starts most gates
        open, with a deterministic gate of 0.98.
    eps : float
        The uniform noise is kept in [eps, 1 - eps]. Default 1e-6.
    """

    def __init__(self, num_gates: int, beta: float = 0.66, gamma: float = -0.1,
                 zeta: float = 1.1, init_open: float = 0.9, eps: float = 1e-6) -> None:
        super().__init__()
        if not (gamma < 0.0 and zeta > 1.0):
            raise ValueError(f"need gamma < 0 and zeta > 1, got gamma={gamma}, zeta={zeta}")
        if not beta > 0.0:
            raise ValueError(f"need beta > 0, got {beta}")
        if not 0.0 < init_open < 1.0:
            raise ValueError(f"need 0 < init_open < 1, got {init_open}")
        self.beta, self.gamma, self.zeta, self.eps = float(beta), float(gamma), float(zeta), float(eps)
        init = math.log(init_open / (1.0 - init_open))
        self.log_alpha = nn.Parameter(torch.full((int(num_gates),), init))

    def _stretch_clamp(self, s: torch.Tensor) -> torch.Tensor:
        return (s * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def forward(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """The gate vector, shape (num_gates,).

        Training mode: a sample, drawn from `generator` (pass a seeded one so
        seeded runs repeat). Eval mode: the deterministic gate.
        """
        if not self.training:
            return self.deterministic()
        u = torch.rand(self.log_alpha.shape, generator=generator,
                       device=self.log_alpha.device, dtype=self.log_alpha.dtype)
        u = u.clamp(self.eps, 1.0 - self.eps)
        s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + self.log_alpha) / self.beta)
        return self._stretch_clamp(s)

    def deterministic(self) -> torch.Tensor:
        """The test-time gate, clamp(sigmoid(log_alpha) * (zeta - gamma) + gamma, 0, 1)."""
        return self._stretch_clamp(torch.sigmoid(self.log_alpha))

    def _shift(self) -> float:
        return self.beta * math.log(-self.gamma / self.zeta)

    def expected_l0(self) -> torch.Tensor:
        """Expected number of nonzero gates, sum_j P(z_j != 0); the sparsity penalty."""
        return torch.sigmoid(self.log_alpha - self._shift()).sum()

    @torch.no_grad()
    def prob_on(self) -> torch.Tensor:
        """P(z_j != 0) per gate, for logs and histograms."""
        return torch.sigmoid(self.log_alpha - self._shift())

    @torch.no_grad()
    def hard_gate(self, threshold: float = 0.05) -> torch.Tensor:
        """Boolean keep-mask: the deterministic gate is above `threshold`."""
        return self.deterministic() > threshold


def group_norms(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    """L2 norm of each group, over all the tensors that make it up.

    Every tensor in `parts` has the groups on dim 0, in the same order:
    group j is row j of each tensor (for example a neuron's coefficient row
    and its column of the head, transposed). Returns shape (G,).
    """
    sq = None
    for p in parts:
        s = p.reshape(p.shape[0], -1).pow(2).sum(dim=1)
        sq = s if sq is None else sq + s
    if sq is None:
        raise ValueError("group_norms needs at least one tensor")
    return sq.clamp_min(_TINY).sqrt()


def group_lasso_penalty(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    """sum_j ||w_j||_2 over the groups of `parts` (see `group_norms`)."""
    return group_norms(parts).sum()


def diversity_loss(h: torch.Tensor, g: torch.Tensor, eps: float = 1e-5, tau: float = 0.0) -> torch.Tensor:
    """Gate weighted mean squared correlation between neuron outputs.

    `h` is (B, K), one column per neuron; `g` is (K,), each neuron's gate
    (or another weight >= 0). The columns are standardized over the batch,
    so the result does not depend on their scale. Each off-diagonal
    correlation enters as max(0, |c_ij| - tau)^2 with the weight g_i g_j,
    so a pair counts only as much as both neurons are open, and the gates
    get a gradient that closes one of two correlated neurons. Returns the
    weighted mean over the pairs, in [0, 1].
    """
    z = (h - h.mean(0)) / (h.std(0) + eps)
    corr = z.T @ z / (h.shape[0] - 1)
    off = torch.relu((corr - torch.diag(torch.diag(corr))).abs() - tau)
    w = g[:, None] * g[None, :]
    w = w - torch.diag(torch.diag(w))
    return (w * off.pow(2)).sum() / (w.sum() + eps)


@torch.no_grad()
def group_soft_threshold(parts: Sequence[torch.Tensor], thresh: float) -> torch.Tensor:
    """The proximal step of `thresh * group_lasso_penalty`, in place.

    Scales every group by max(0, 1 - thresh / ||w_j||): a group whose norm
    is at most `thresh` becomes exactly zero, the others shrink toward zero
    by `thresh`. After a gradient step with learning rate `lr`, call it with
    `thresh = lam * lr`. Returns the group norms after the step.
    """
    norm = group_norms(parts)
    scale = (norm - thresh).clamp_min(0.0) / norm
    for p in parts:
        p.mul_(scale.view(-1, *([1] * (p.dim() - 1))).to(dtype=p.dtype))
    return norm * scale
