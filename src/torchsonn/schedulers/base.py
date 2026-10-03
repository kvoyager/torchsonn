from abc import ABC
from typing import Any


class BaseScheduler(ABC):
    """Base class of the learning-rate schedulers.

    A scheduler holds an optimizer and rewrites its `lr` on every `step`;
    subclasses implement `step`, `state_dict` and `load_state_dict`.
    """

    def state_dict(self) -> dict[str, Any]:
        """Return the scheduler state as a dict; implemented by subclasses."""
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the state written by `state_dict`; implemented by subclasses."""
        raise NotImplementedError