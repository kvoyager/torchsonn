import torch

from collections.abc import Iterable
from typing import Any

from torchsonn.optimizers.base import BaseOptimizer, LRLike


class BatchedSGD(BaseOptimizer):
    def __init__(
        self,
        params: dict[str, torch.Tensor],
        shared_param_names: Iterable[str],
        lr: LRLike,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        clip_value: float | None = None,
        clip_norm: float | None = None,
        shared_param_lr_multiplier: float = 1.0,
    ) -> None:
        super().__init__(shared_param_names, lr, clip_value, clip_norm, shared_param_lr_multiplier)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # Momentum buffer
        self.v: dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in params.items()}

    def step(
        self,
        params: dict[str, torch.Tensor],
        grads: dict[str, torch.Tensor],
        active_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        params, grads: dict of tensors with leading batch dimension
        active_mask: tensor of shape (batch,) bool, True if model is active
        """
        batch_size = next(iter(params.values())).shape[0]

        if active_mask is None:
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=next(iter(params.values())).device)

        new_params = {}
        for k in params.keys():
            p = params[k]
            g = grads[k]
            v = self.v[k]

            if self.weight_decay > 0:
                g = g + self.weight_decay * p

            g = self.gradient_clipping(g)

            update_mask = active_mask.view(-1, *([1] * (p.dim() - 1)))
            lr = self.lr.unsqueeze(dim=1)

            if k in self.shared_param_names:
                idx = torch.where(active_mask)[0]
                lr = lr.index_select(0, idx).mean() * self.shared_param_lr_multiplier
                g = g.index_select(0, idx).mean(dim=0)

            v.mul_(self.momentum).add_(g)

            if self.nesterov:
                update = lr * (self.momentum * v + g)
            else:
                update = lr * v

            if k in self.shared_param_names:
                new_params[k] = p - update
            else:
                new_params[k] = torch.where(update_mask, p - update, p)

        return new_params

    def state_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "v": {k: v.clone() for k, v in self.v.items()},
            "shared_param_names": list(self.shared_param_names),
            "shared_param_lr_multiplier": self.shared_param_lr_multiplier,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.lr = state_dict["lr"]
        self.weight_decay = state_dict["weight_decay"]
        self.momentum = state_dict["momentum"]
        self.nesterov = state_dict["nesterov"]
        self.v = {k: v.clone() for k, v in state_dict["v"].items()}
        self.shared_param_names = set(state_dict["shared_param_names"])
        self.shared_param_lr_multiplier = state_dict.get("shared_param_lr_multiplier", 1.0)