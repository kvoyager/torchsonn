"""Growth state of `Trainer.train`: the stop rule, the per-layer accumulator
and the step checkpoint snapshot."""
import sys
from dataclasses import dataclass, field, fields

import torch

from torchsonn.model import SONN
from torchsonn.optimizers import BaseOptimizer


@dataclass
class GrowthCriterion:
    """The layer-growth stop rule of `Trainer.train`, kept out of the loop so
    it can be tested on a scripted sequence of layer errors.

    A layer *improves* when it lowers the error of the last accepted
    improvement (`ref_error`) by at least `max(min_delta, epsilon * best)`,
    so several small steps down add up instead of each failing the margin
    on its own; the search stops after `width` consecutive layers without
    an improvement. `best_index` (the layer with the lowest error, margin or
    not) is what the trainer keeps; the window is counted from
    `last_improved_index`. Layer 0 always improves.
    """
    width: int
    epsilon: float
    min_delta: float = 0.0
    min_error: float = field(default=sys.float_info.max)
    ref_error: float = field(default=sys.float_info.max)
    best_index: int = 0
    last_improved_index: int = 0
    # Filled by the last `update` for the log.
    last_gain: float = 0.0
    last_margin: float = 0.0
    last_since: int = 0

    def update(self, layer_index: int, err: float) -> bool:
        """Record a layer's error; returns True when the search should stop."""
        err = float(err)
        if layer_index == 0 or self.min_error == sys.float_info.max:
            margin = 0.0
            gain = float("inf")
            improved = True
        else:
            margin = max(float(self.min_delta), float(self.epsilon) * self.min_error)
            gain = self.ref_error - err
            improved = gain >= margin - 1e-12
        if err < self.min_error:
            self.min_error = err
            self.best_index = layer_index
        if improved:
            self.last_improved_index = layer_index
            self.ref_error = err
        self.last_gain, self.last_margin = gain, margin
        self.last_since = layer_index - self.last_improved_index
        return layer_index > 0 and self.last_since >= int(self.width)

    def describe(self, layer_index: int, err: float) -> str:
        """Return a one-line log message on the last `update`: error, best layer, gain and margin, layers without improvement."""
        gain = "first layer" if self.last_gain == float("inf") else f"improved by {self.last_gain:+.4f} (margin {self.last_margin:.4f})"
        return (f"Layer #{layer_index}: error {float(err):.4f}, best {self.min_error:.4f} at layer "
                f"{self.best_index}; {gain}; {self.last_since} of {int(self.width)} layers without improvement")


@dataclass
class LayerAccumulator:
    """Mutable per-layer state accumulated across neuron-model training calls.

    Only lives in memory — never serialized directly.  train_layer creates one
    at the start and passes it through every train_model_ensemble call so the
    growing err / module_idxs lists are available when building StepCheckpoints.
    """
    err: list[torch.Tensor] = field(default_factory=list)
    module_idxs: list[torch.Tensor] = field(default_factory=list)
    # Per-neuron-model (C_i, N_val) raw scalar outputs on the dev split.
    # neuron_selection cats along dim 0 → (C_total, N_val) and uses it for
    # OMP-style decorrelation alongside err / module_idxs.
    z_val: list[torch.Tensor] = field(default_factory=list)
    # Dev split targets in the same order as z_val's columns, (N_val,) or
    # (N_val, O). The same for every neuron model, so it is kept once.
    y_val: torch.Tensor | None = None
    layer_completed: bool = False


@dataclass
class StepCheckpoint:
    """Immutable snapshot of all state needed to save or resume a training step.

    All fields are required so the type checker can catch a missing
    assignment at construction time rather than at serialization time.
    Built explicitly at each save site inside train_model_ensemble; never
    mutated after construction (use dataclasses.replace() for variants).
    """
    # serialization anchor — also used for checkpoint_dir / keep_last_n lookups
    model: SONN
    opt: BaseOptimizer
    # positional context
    layer_idx: int
    neuron_model_idx: int
    # step counters
    epoch: int
    step: int
    global_step: int
    # per-ensemble training state
    best_val_losses: torch.Tensor
    smoothed_val_losses: torch.Tensor
    last_steps_improve: torch.Tensor
    early_stop_flags: torch.Tensor
    lr: torch.Tensor
    neuron_model_completed: bool
    # layer-level state (needed to resume across neuron-model iterations)
    layer_completed: bool
    err: list[torch.Tensor]
    module_idxs: list[torch.Tensor]

    def to_dict(self) -> dict:
        """Return the snapshot as a dict, with `state_dict()` in place of any object that has one and None fields left out."""
        def safe_value(v):
            if hasattr(v, "state_dict"):
                return v.state_dict()
            return v
        return {
            f.name: safe_value(getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) is not None
        }
