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

        # Optional per-layer LayerNorm over the input of the *next* layer,
        # applied AFTER the shortcut concat in SONN.forward so the raw input
        # features and older layers' outputs cat'd in by the shortcut are
        # folded into the normalization. Built lazily by setup_layer_norm(dim)
        # once we know the final concatenated width (SONN.next_input_width).
        # Stored explicitly so from_checkpoint_metadata can reconstruct
        # without re-deriving it from model config.
        self.use_layer_norm = use_layer_norm
        self.layer_norm_dim: int | None = None
        self.layer_norm: nn.LayerNorm | None = None

        # What this layer's input is made of, in concatenation order: the
        # outputs of the layers at `input_layers` (model positions, nearest
        # first), then the raw model inputs when `input_raw`. Kept here rather
        # than re-derived from `model.shortcut` because pruning can delete a
        # layer, after which "the last k layers" would name different ones.
        # Both start as None and are set right after construction: by
        # SONN.create_layer for a new layer, or from the saved metadata on
        # restore. Trainer.prune re-encodes them; SONN.layer_sources reads them.
        self.input_layers: list[int] | None = None
        self.input_raw: bool | None = None

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
            "input_layers",
            "input_raw",
        ])

    def setup_layer_norm(self, dim: int) -> None:
        """Build the per-layer LayerNorm with the supplied `dim` as its
        `normalized_shape`. Idempotent and a no-op when the flag is off.

        Called from `Trainer.neuron_selection` after pruning (passing
        the width of the next layer's input, `SONN.next_input_width`) and
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

    def fit_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Hand this layer's input statistics to every neuron model.

        `mean` / `std` are length-`num_feat` vectors over the features that
        feed this layer, measured on the training set. Called by
        `Trainer.fit_layer_inputs` after the layer is created and before it
        trains; a no-op for neuron families that calibrate nothing.
        """
        for neuron_model in self.neuron_models:
            neuron_model.fit_input_stats(mean, std)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Record the neuron class names, then return the module state dict."""
        self.neuron_models_names = [neuron_model.__class__.__name__ for neuron_model in self.neuron_models]
        return super().state_dict(*args, **kwargs)

    @classmethod
    def from_checkpoint_metadata(cls, metadata: dict[str, Any]) -> "SONNLayer":
        """
        Rebuild an empty layer from its checkpoint metadata.

        Restores the constructor arguments, the LayerNorm at its saved width
        and the input layout (`input_layers`, `input_raw`). The neuron
        modules are not created here; the model restores them separately.

        Parameters
        ----------
        metadata : dict
            The layer's entry in the checkpoint metadata.

        Returns
        -------
        layer : SONNLayer
            The restored layer, without neuron modules.
        """
        layer = cls(
            d_model=metadata["d_model"],
            nbest_neurons=metadata["nbest_neurons"],
            layer_index=metadata["layer_index"],
            use_layer_norm=metadata["use_layer_norm"],
        )
        # Restore the LayerNorm at its saved width. `layer_norm_dim` is None
        # when the layer has no LayerNorm; setup_layer_norm is then a no-op.
        layer.setup_layer_norm(metadata["layer_norm_dim"] or metadata["d_model"])
        layer.input_layers = metadata["input_layers"]
        layer.input_raw = metadata["input_raw"]
        return layer

    def to(self, *args: Any, **kwargs: Any) -> "SONNLayer":
        """Move the layer, including its plain tensor attributes, like `nn.Module.to`."""
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
        """
        Run every neuron module on the layer input.

        Parameters
        ----------
        x : (B, D_in) tensor
            The layer input.

        Returns
        -------
        out : (B, len(self)) tensor
            The outputs of all neuron modules, concatenated along dim 1 in
            module order.
        """

        res = []
        for module in self.neuron_models:
            out = module(x)
            res.append(out)
        return torch.cat(res, dim=1)

    def describe(self, features: list[str], layers: "list[SONNLayer]") -> str:
        """Return a text description of the layer: a header and one entry per neuron module."""

        s = ['*' * 50,
             'Layer {0}'.format(self.layer_index),
             '*' * 50,
        ]
        for neuron in self:
            s.append(neuron.describe(features, layers))
        return '\n'.join(s)

    def get_parent_neron_module(self, idx: int) -> tuple[int, BasePolynomNeuron]:
        """
        Find the neuron module that holds output column `idx`.

        Parameters
        ----------
        idx : int
            Absolute neuron (output column) index in the layer.

        Returns
        -------
        idx : int
            The same absolute index.
        module : BasePolynomNeuron
            The neuron module that produces that column.

        Raises
        ------
        ValueError
            If `idx` is not below the layer's neuron count.
        """
        parent_neuron_idx = 0
        for parent_neuron in self.neuron_models:
            for _ in range(parent_neuron.num_neurons):
                if parent_neuron_idx == idx:
                    return parent_neuron_idx, parent_neuron
                parent_neuron_idx += 1
        raise ValueError

    def set_neuron_module(self) -> None:
        """
        Build the column map of the layer.

        Sets `neuron_idxs`, an (N, 2) tensor whose row `i` holds the module
        index and the neuron index inside that module for output column `i`,
        and sets `d_model` to the number of columns N.
        """
        # set map, absolute_neuron_idx <=> neuron_module idx, relative_neuron_idx
        self.neuron_idxs = []
        neuron_idx = 0
        for neuron_module_idx, neuron in enumerate(self.neuron_models):
            for i in range(neuron.num_neurons):
                self.neuron_idxs.append([neuron_module_idx, i])
                neuron_idx += 1

        self.neuron_idxs = torch.tensor(self.neuron_idxs)
        self.d_model = self.neuron_idxs.shape[0]


