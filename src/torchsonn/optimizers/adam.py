import torch

from collections.abc import Iterable
from typing import Any

from torchsonn.optimizers.base import BaseOptimizer, LRLike


class BatchedAdam(BaseOptimizer):
    """Adam, batched over the candidate ensemble.

    Per member and parameter tensor, with g the clipped gradient and t the
    step count (shared by all members):

        m = b1*m + (1 - b1)*g,      v = b2*v + (1 - b2)*g^2
        theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
        m_hat = m / (1 - b1^t),     v_hat = v / (1 - b2^t)

    With `weight_decay` wd > 0 the decay is decoupled from the gradient, as
    in AdamW: theta = theta - lr * wd * theta - lr * m_hat / (sqrt(v_hat) + eps).

    Members with `active_mask` False in `step` keep their parameters; their
    moment buffers still advance.

    Parameters
    ----------
    params : dict of str -> (B, ...) tensor
        Initial parameters; they set the shapes of the moment buffers.
    shared_param_names : iterable of str
        See `BaseOptimizer`.
    lr : float or (B,) tensor
        See `BaseOptimizer`.
    clip_value : float or None
        See `BaseOptimizer`.
    clip_norm : float or None
        See `BaseOptimizer`.
    shared_param_lr_multiplier : float
        See `BaseOptimizer`.
    betas : (float, float)
        Decay rates b1, b2 of the moment estimates. Default (0.9, 0.999).
    eps : float
        Added to sqrt(v_hat) in the denominator. Default 1e-8.
    weight_decay : float
        Decoupled weight decay wd, applied to every parameter. Default 0.0.
    """

    def __init__(
        self,
        params: dict[str, torch.Tensor],
        shared_param_names: Iterable[str],
        lr: LRLike,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        clip_value: float | None = None,
        clip_norm: float | None = None,
        shared_param_lr_multiplier: float = 1.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(shared_param_names, lr, clip_value, clip_norm, shared_param_lr_multiplier)
        self.betas = betas
        self.eps = eps
        self.weight_decay = float(weight_decay)
        # now store as dict for consistency with torch.func style
        self.m: dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v: dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in params.items()}
        self.t: int = 0

    def step(
        self,
        params: dict[str, torch.Tensor],
        grads: dict[str, torch.Tensor],
        active_mask: torch.Tensor | None = None,
        b: Any = None,
    ) -> dict[str, torch.Tensor]:
        """Take one optimizer step for every active member.

        Parameters
        ----------
        params : dict of str -> (B, ...) tensor
            Current parameters, with the members on the leading dimension.
        grads : dict of str -> (B, ...) tensor
            Their gradients, with the same keys and shapes.
        active_mask : (B,) bool tensor, optional
            True for the members that step; None steps every member.
        b : any, optional
            Not used.

        Returns
        -------
        dict of str -> (B, ...) tensor
            The new parameters, with the same keys and shapes.
        """
        self.t += 1
        batch_size = next(iter(params.values())).shape[0]

        if active_mask is None:
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=next(iter(params.values())).device)

        new_params = {}
        for k in params.keys():
            p = params[k]
            g = grads[k]
            m = self.m[k]
            v = self.v[k]

            g = self.gradient_clipping(g)

            # Broadcast mask over param dims > 0
            update_mask = active_mask.view(-1, *([1] * (p.dim() - 1)))

            # Per-member lr broadcast over every trailing dim of the parameter
            # (2-D weights, 3-D RBF centers alike).
            lr = self.lr.view(-1, *([1] * (p.dim() - 1)))
            if k in self.shared_param_names:
                idx = torch.where(active_mask)[0]
                lr = lr.index_select(0, idx).mean() * self.shared_param_lr_multiplier
                g = g.index_select(0, idx).mean(dim=0)

            m.mul_(self.betas[0]).add_(g, alpha=1 - self.betas[0])
            v.mul_(self.betas[1]).addcmul_(g, g, value=1 - self.betas[1])

            m_hat = m / (1 - self.betas[0] ** self.t)
            v_hat = v / (1 - self.betas[1] ** self.t)

            update = lr * m_hat / (v_hat.sqrt() + self.eps)
            if self.weight_decay > 0:
                update = update + lr * self.weight_decay * p

            new_params[k] = p - update if k in self.shared_param_names else torch.where(update_mask, p - update, p)

        return new_params

    def state_dict(self) -> dict[str, Any]:
        """Return lr, betas, eps, weight_decay, the moment buffers, the step count and the shared-parameter settings."""
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "m": {k: v.clone() for k, v in self.m.items()},
            "v": {k: v.clone() for k, v in self.v.items()},
            "t": self.t,
            "shared_param_names": list(self.shared_param_names),
            "shared_param_lr_multiplier": self.shared_param_lr_multiplier,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the state written by `state_dict`; every field must be
        present, except `weight_decay`, read as 0.0 when missing."""
        self.lr = state_dict["lr"]
        self.betas = state_dict["betas"]
        self.eps = state_dict["eps"]
        self.weight_decay = float(state_dict.get("weight_decay", 0.0))
        self.m = {k: v.clone() for k, v in state_dict["m"].items()}
        self.v = {k: v.clone() for k, v in state_dict["v"].items()}
        self.t = state_dict["t"]
        self.shared_param_names = set(state_dict["shared_param_names"])
        self.shared_param_lr_multiplier = state_dict["shared_param_lr_multiplier"]