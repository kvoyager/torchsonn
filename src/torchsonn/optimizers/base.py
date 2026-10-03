import torch

from abc import ABC
from collections.abc import Iterable
from typing import Any

# `lr` can be either a single scalar (rare; only when the trainer hasn't yet
# broadcast per-ensemble rates) or a 1-D tensor of shape (ensemble_size,).
# All subclasses index into it as if it were a tensor.
LRLike = torch.Tensor | float


class BaseOptimizer(ABC):
    """Base class of the ensemble-batched optimizers.

    Every parameter tensor carries the candidate ensemble on its leading
    dimension, one row per candidate neuron, and a subclass's `step` updates
    all rows at once. Parameters named in `shared_param_names` have no
    ensemble dimension; the subclasses update them once per step from the
    gradient averaged over the active members, at the mean of those members'
    learning rates times `shared_param_lr_multiplier`.

    Parameters
    ----------
    shared_param_names : iterable of str
        Names of the parameters shared by every ensemble member.
    lr : float or (ensemble_size,) tensor
        Learning rate, one per member when a tensor.
    clip_value : float, optional
        Clamp every gradient element to [-clip_value, clip_value].
    clip_norm : float, optional
        Rescale each member's gradient to at most this L2 norm.
    shared_param_lr_multiplier : float
        Factor on the learning rate of the shared parameters. Default 1.0.
    """

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
        """Return the optimizer state as a dict; implemented by subclasses."""
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the state written by `state_dict`; implemented by subclasses."""
        raise NotImplementedError

    def gradient_clipping(self, g: torch.Tensor) -> torch.Tensor:
        """Clip a batched gradient by value, then by per-member norm.

        Parameters
        ----------
        g : (B, ...) tensor
            Gradient with the ensemble on the leading dimension.

        Returns
        -------
        g : (B, ...) tensor
            `g` clamped to +-`clip_value` (when set), then each member's slice
            scaled down to L2 norm `clip_norm` (when set). Unchanged when both
            are None.
        """
        if self.clip_value is not None:
            g = torch.clamp(g, -self.clip_value, self.clip_value)
        if self.clip_norm is not None:
            g_norm = g.flatten(1).norm(2, dim=1, keepdim=True)
            scale = torch.clamp(self.clip_norm / (g_norm + 1e-6), max=1.0)
            g = g * scale.view(-1, *([1] * (g.ndim - 1)))
        return g