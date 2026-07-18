from torchsonn.schedulers.base import BaseScheduler
from torchsonn.schedulers.warmup_flat import WarmupFlatScheduler


scheduler_map = {
    "warmup_flat": WarmupFlatScheduler,
}


__all__ = [
    "BaseScheduler",
    "WarmupFlatScheduler",
    "scheduler_map",
]