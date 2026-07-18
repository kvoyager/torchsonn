import torch

from typing import Any

from torchsonn.optimizers.base import BaseOptimizer
from torchsonn.schedulers.base import BaseScheduler


class WarmupFlatScheduler(BaseScheduler):
    """Linear warmup followed by flat LR schedule."""

    def __init__(self, optimizer: BaseOptimizer, warmup_steps: int) -> None:
        self.opt = optimizer
        self.warmup_steps = warmup_steps
        # Snapshot the initial LR so it isn't mutated when opt.lr is later rewritten.
        base = optimizer.lr
        self.base_lr = base.detach().clone() if isinstance(base, torch.Tensor) else base
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            scale = self.step_num / float(max(1, self.warmup_steps))
        else:
            scale = 1.0  # flat after warmup

        self.opt.lr = self.base_lr * scale

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "base_lr": self.base_lr,
            "step_num": self.step_num,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.warmup_steps = state_dict["warmup_steps"]
        self.base_lr = state_dict["base_lr"]
        self.step_num = state_dict["step_num"]