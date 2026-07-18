"""Hydra-structured config schemas for SONN.

Importing this package triggers `ConfigStore.store(name="default", node=SONNConfig)`,
so any subsequent `@hydra.main` / `hydra.compose` call that references
`- default` in its `defaults:` list resolves to the typed dataclass below.
"""
from torchsonn.config.schemas import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    SONNConfig,
    TrainConfig,
)


__all__ = [
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "SONNConfig",
    "TrainConfig",
]