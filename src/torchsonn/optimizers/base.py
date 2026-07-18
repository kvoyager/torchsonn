import torch

from abc import ABC
from collections.abc import Iterable
from typing import Any

# `lr` can be either a single scalar (rare; only when the trainer hasn't yet
# broadcast per-ensemble rates) or a 1-D tensor of shape (ensemble_size,).
# All subclasses index into it as if it were a tensor.
LRLike = torch.Tensor | float


class BaseOptimizer(ABC):

    def __init__(
        self,
        shared_param_names: Iterable[str],
        lr: LRLike,
        clip_value: float | None = None,
        clip_norm: float | None = None,
        shared_param_lr_multiplier: float = 1.0,
    ) -> None:
        self.lr = lr
        self.clip_value = clip_value
        self.clip_norm = clip_norm
        self.shared_param_names: set[str] = set(shared_param_names)
        self.shared_param_lr_multiplier = shared_param_lr_multiplier

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raise NotImplementedError

    def gradient_clipping(self, g: torch.Tensor) -> torch.Tensor:
        if self.clip_value is not None:
            g = torch.clamp(g, -self.clip_value, self.clip_value)
        if self.clip_norm is not None:
            g_norm = g.flatten(1).norm(2, dim=1, keepdim=True)
            scale = torch.clamp(self.clip_norm / (g_norm + 1e-6), max=1.0)
            g = g * scale.view(-1, *([1] * (g.ndim - 1)))
        return g