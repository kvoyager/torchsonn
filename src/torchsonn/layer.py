import sys
from typing import Any, Iterator, overload

import torch
from torch import nn

from torchsonn.modules import SONNModule
from torchsonn.neurons import BasePolynomNeuron


class NeuronModuleList(nn.ModuleList):
    """nn.ModuleList specialized to hold BasePolynomNeuron items.

    nn.ModuleList isn't generic in PyTorch, so the only way to give static
    checkers / IDEs a typed element is a thin subclass. Runtime behavior is
    unchanged — these methods just narrow the return type from
    `nn.Module | nn.ModuleList` to `BasePolynomNeuron` / `NeuronModuleList`.
    """

    def __iter__(self) -> Iterator[BasePolynomNeuron]:
        return super().__iter__()  # type: ignore[return-value]

    @overload
    def __getitem__(self, idx: int) -> BasePolynomNeuron: ...
    @overload
    def __getitem__(self, idx: slice) -> "NeuronModuleList": ...
    def __getitem__(self, idx):
        # nn.ModuleList.__getitem__ is stubbed with separate int/slice overloads
        # in torch's type stubs; passing the narrowed-but-still-union impl
        # parameter trips a false-positive without the ignore.
        return super().__getitem__(idx)  # type: ignore[arg-type]


#***********************************************************************************************************************
#   Network layer
#***********************************************************************************************************************
class SONNLayer(SONNModule):
    """Layer class
    """

    def __init__(
        self,
        d_model: int,
        nbest_neurons: int,
        layer_index: int,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        # pytorch modules
        self.neuron_models: NeuronModuleList = NeuronModuleList()

        # regular variables
        self.layer_index = layer_index
        self.nbest_neurons = nbest_neurons
        self.d_model = d_model
        self.err = sys.float_info.max
        self.err_values = None
        # self.err_idxs = None
        # self.topk_module_idxs: torch.Tensor | None = None
        self.module_idxs: torch.Tensor | None = None
        self.neuron_idxs = None  # map, absolute_neuron_idx <=> neuron_module idx, relative_neuron_idx
        self.neuron_models_names = []  # filled during state_dict() call

        # Optional per-layer LayerNorm, applied AFTER the optional shortcut
        # concat in SONN.forward so the raw input features (cat'd in by the
        # shortcut path) are folded into the normalization. Built lazily by
        # setup_layer_norm(dim) once we know the final concatenated width —
        # the caller passes d_model + d_orig when shortcut is on, d_model
        # alone otherwise. Stored explicitly so from_checkpoint_metadata can
        # reconstruct without re-deriving it from model config.
        self.use_layer_norm = use_layer_norm
        self.layer_norm_dim: int | None = None
        self.layer_norm: nn.LayerNorm | None = None

        self.params_metadata_names.extend([
            "layer_index",
            "nbest_neurons",
            "d_model",
            "err",
            "err_values",
            "module_idxs",
            "neuron_idxs",
            "neuron_models_names",
            "use_layer_norm",
            "layer_norm_dim",
        ])

    def setup_layer_norm(self, dim: int) -> None:
        """Build the per-layer LayerNorm with the supplied `dim` as its
        `normalized_shape`. Idempotent and a no-op when the flag is off.

        Called from `Trainer.neuron_selection` after pruning (passing
        `d_model + d_orig` when shortcut is on, `d_model` otherwise) and
        from `from_checkpoint_metadata` when restoring a model.
        """
        if not self.use_layer_norm or self.layer_norm is not None:
            return
        self.layer_norm_dim = int(dim)
        self.layer_norm = nn.LayerNorm(self.layer_norm_dim, elementwise_affine=False)
        # Mirror the neuron_models' device so SONN.forward can apply it
        # inline without an implicit cross-device dispatch.
        if len(self.neuron_models) > 0:
            self.layer_norm = self.layer_norm.to(self.neuron_models[0].weight.device)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.neuron_models_names = [neuron_model.__class__.__name__ for neuron_model in self.neuron_models]
        return super().state_dict(*args, **kwargs)

    @classmethod
    def from_checkpoint_metadata(cls, metadata: dict[str, Any]) -> "SONNLayer":
        layer = cls(
            d_model=metadata["d_model"],
            nbest_neurons=metadata["nbest_neurons"],
            layer_index=metadata["layer_index"],
            use_layer_norm=metadata.get("use_layer_norm", False),
        )
        # Restore the LayerNorm at its saved width. `layer_norm_dim` may be
        # missing in pre-LayerNorm checkpoints — fall back to d_model so
        # restore still succeeds, even though that case implies the flag
        # was off and setup_layer_norm will be a no-op anyway.
        ln_dim = metadata.get("layer_norm_dim") or metadata["d_model"]
        layer.setup_layer_norm(ln_dim)
        return layer

    def to(self, *args: Any, **kwargs: Any) -> "SONNLayer":
        # module_idxs / neuron_idxs / err_values are plain tensor attributes (not
        # registered buffers) so nn.Module.to() doesn't move them. Mirror the
        # BasePolynomNeuron.to() pattern so model.to('cuda') is consistent.
        for name in ("module_idxs", "neuron_idxs", "err_values"):
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor):
                setattr(self, name, t.to(*args, **kwargs))
        return super().to(*args, **kwargs)

    def __len__(self) -> int:
        return sum([item.num_neurons for item in self.neuron_models])

    def __getitem__(self, idx: int) -> BasePolynomNeuron:
        return self.neuron_models[idx]

    def __repr__(self) -> str:
        return 'Layer {0}'.format(self.layer_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        res = []
        for module in self.neuron_models:
            out = module(x)
            res.append(out)
        return torch.cat(res, dim=1)

    def describe(self, features: list[str], layers: "list[SONNLayer]") -> str:

        s = ['*' * 50,
             'Layer {0}'.format(self.layer_index),
             '*' * 50,
        ]
        for neuron in self:
            s.append(neuron.describe(features, layers))
        return '\n'.join(s)

    def get_parent_neron_module(self, idx: int) -> tuple[int, BasePolynomNeuron]:
        parent_neuron_idx = 0
        for parent_neuron in self.neuron_models:
            for _ in range(parent_neuron.num_neurons):
                if parent_neuron_idx == idx:
                    return parent_neuron_idx, parent_neuron
                parent_neuron_idx += 1
        raise ValueError

    def set_neuron_module(self) -> None:
        # set map, absolute_neuron_idx <=> neuron_module idx, relative_neuron_idx
        self.neuron_idxs = []
        neuron_idx = 0
        for neuron_module_idx, neuron in enumerate(self.neuron_models):
            for i in range(neuron.num_neurons):
                self.neuron_idxs.append([neuron_module_idx, i])
                neuron_idx += 1

        self.neuron_idxs = torch.tensor(self.neuron_idxs)
        self.d_model = self.neuron_idxs.shape[0]


