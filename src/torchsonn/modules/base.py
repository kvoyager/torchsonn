from collections.abc import Mapping
from typing import Any

from torch import nn


class SONNModule(nn.Module):
    """`nn.Module` that saves plain-Python metadata next to its tensors.

    Subclasses list attribute names in `params_metadata_names`. `state_dict`
    stores their values under a `params_metadata` key, and `load_state_dict`
    restores them on every `SONNModule` in the tree, so settings that are not
    tensors travel with the checkpoint.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.params_metadata_names: list[str] = []

    def _set_metadata_dict(self, metadata_dict: Mapping[str, Any]) -> None:
        """Restore metadata fields from dict."""
        for name in self.params_metadata_names:
            if name in metadata_dict:
                setattr(self, name, metadata_dict[name])

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return the tensor state dict plus a `params_metadata` entry holding the attributes named in `params_metadata_names`."""
        base_dict = super().state_dict(*args, **kwargs)
        base_dict[f"{kwargs.get('prefix', '')}params_metadata"] = {
            name: getattr(self, name) for name in self.params_metadata_names
        }
        return base_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any) -> "SONNModule":
        """Load parameters and metadata recursively."""
        # Restore tensors
        super().load_state_dict(state_dict, *args, **kwargs)

        for name, module in self.named_modules():
            if isinstance(module, SONNModule):
                key = f"{name}.params_metadata" if name else "params_metadata"
                if key in state_dict:
                    module._set_metadata_dict(state_dict[key])

        return self