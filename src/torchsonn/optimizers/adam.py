import torch

from collections.abc import Iterable
from typing import Any

from torchsonn.optimizers.base import BaseOptimizer, LRLike


class BatchedAdam(BaseOptimizer):
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
    ) -> None:
        super().__init__(shared_param_names, lr, clip_value, clip_norm, shared_param_lr_multiplier)
        self.betas = betas
        self.eps = eps
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
        """
        params, grads: dicts of tensors with leading batch dimension
        active_mask: tensor of shape (batch,) bool, True if model active
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

            lr = self.lr.unsqueeze(dim=1)
            if k in self.shared_param_names:
                idx = torch.where(active_mask)[0]
                lr = lr.index_select(0, idx).mean() * self.shared_param_lr_multiplier
                g = g.index_select(0, idx).mean(dim=0)

            m.mul_(self.betas[0]).add_(g, alpha=1 - self.betas[0])
            v.mul_(self.betas[1]).addcmul_(g, g, value=1 - self.betas[1])

            m_hat = m / (1 - self.betas[0] ** self.t)
            v_hat = v / (1 - self.betas[1] ** self.t)

            update = lr * m_hat / (v_hat.sqrt() + self.eps)

            new_params[k] = p - update if k in self.shared_param_names else torch.where(update_mask, p - update, p)

        return new_params

    def state_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "m": {k: v.clone() for k, v in self.m.items()},
            "v": {k: v.clone() for k, v in self.v.items()},
            "t": self.t,
            "shared_param_names": list(self.shared_param_names),
            "shared_param_lr_multiplier": self.shared_param_lr_multiplier,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.lr = state_dict["lr"]
        self.betas = state_dict["betas"]
        self.eps = state_dict["eps"]
        self.m = {k: v.clone() for k, v in state_dict["m"].items()}
        self.v = {k: v.clone() for k, v in state_dict["v"].items()}
        self.t = state_dict["t"]
        self.shared_param_names = set(state_dict["shared_param_names"])
        self.shared_param_lr_multiplier = state_dict.get("shared_param_lr_multiplier", 1.0)