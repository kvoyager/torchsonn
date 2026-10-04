import numpy as np
import sys

import torch
from torch import nn

from collections.abc import Mapping
from typing import Any, Optional

from omegaconf import OmegaConf, DictConfig

# Side-effect import: registers SONNConfig with Hydra's ConfigStore so the
# tutorial YAMLs' `defaults: [default, _self_]` resolves to the typed schema.
from torchsonn.config import SONNConfig
from torch.func import vmap, grad, functional_call
from torchsonn.modules import SONNModule, SoftBinner

import torch.nn.functional as F
from torchsonn.layer import SONNLayer
from torchsonn.neurons import (
    BaseOrthogonalNeuron,
    BasePolynomNeuron,
    LinearPolynomNeuron,
    LinearCovPolynomNeuron,
    QuadraticPolynomNeuron,
    CubicPolynomNeuron,
    PolyQuadratic,
    LegendrePolynomNeuron,
    ChebyshevPolynomNeuron,
    RBFNeuron,
)
from torchsonn.loss import NormMSE
from torchsonn.types import RefFunctionType, CriterionType, LayerCreationError
import logging


def _parse_ref_function_entry(entry: Any) -> tuple[RefFunctionType, Optional[dict]]:
    """Parse a single entry from `model.ref_functions`.

    The list is intentionally heterogeneous (typed `List[Any]` in the config)
    because each ref-function family takes its own option set. Two shapes are
    accepted:

      • a bare name string  →  options = None
            - linear_cov
            - quadratic

      • a single-key mapping carrying options, e.g.
            - polyquad:
                squares: true
                dim: 3

    Returns (RefFunctionType, options_dict_or_None). The options dict is left
    untyped on purpose — each neuron class consumes its own kwargs (see
    `PolyQuadratic.__init__`'s `squares` / `dim`).
    """
    if isinstance(entry, (Mapping, DictConfig)):
        name = next(iter(entry))
        raw = entry[name]
        if raw is None:
            # `- polyquad:` (trailing colon, no value) parses to {name: None}.
            # Treat it as the bare-name form — use the ref function's own
            # default options.
            return RefFunctionType.get(name), None
        if not isinstance(raw, (Mapping, DictConfig)):
            raise TypeError(
                f"ref_functions[{name!r}] options must be a mapping, "
                f"got {type(raw).__name__}"
            )
        options = {k: v for k, v in raw.items()}
        return RefFunctionType.get(name), options
    # Bare-name form (string or RefFunctionType).
    return RefFunctionType.get(entry), None


def _parse_shortcut(value: Any) -> tuple[bool, int]:
    """Parse `model.shortcut` into (raw_features, prev_layers).

    Two shapes are accepted:

      • the bool shorthand `shortcut: true|false`, meaning
        `{raw_features: true|false, prev_layers: null}`;
      • a mapping with the keys `raw_features` (bool, default true) and
        `prev_layers` (int >= 0 | 'all' | null, default null).

    `prev_layers` counts the layers *before* the last one whose outputs also
    feed a new layer; the last layer always feeds it. null is returned as 0
    and 'all' as sys.maxsize, so callers clip it to the layers that exist.
    """
    if isinstance(value, bool):
        return value, 0
    if not isinstance(value, (Mapping, DictConfig)):
        raise ValueError(
            f"model.shortcut must be a bool or a mapping, got {type(value).__name__} {value!r}"
        )
    unknown = set(value.keys()) - {"raw_features", "prev_layers"}
    if unknown:
        raise ValueError(
            f"model.shortcut has unknown key(s) {sorted(unknown)}; "
            "expected 'raw_features' and 'prev_layers'"
        )
    raw = value.get("raw_features", True)
    if not isinstance(raw, bool):
        raise ValueError(f"model.shortcut.raw_features must be a bool, got {raw!r}")
    prev = value.get("prev_layers", None)
    if prev is None:
        prev = 0
    elif prev == "all":
        prev = sys.maxsize
    elif isinstance(prev, bool) or not isinstance(prev, int) or prev < 0:
        raise ValueError(
            f"model.shortcut.prev_layers must be null, a non-negative int or 'all', got {prev!r}"
        )
    return raw, prev


logger = logging.getLogger(__name__)


class SONN(SONNModule):
    """Self-organizing deep learning polynomial neural network (GMDH) as a PyTorch module.

    The model starts with no layers; `Trainer.train` grows them, and `infer`
    predicts with the trained network.

    Parameters
    ----------
    config : DictConfig or mapping
        The configuration. It is merged into `default_config()`, so it may
        hold only the keys that differ from the defaults.
    d_model : int
        Number of input features.
    feature_names : list of str or ndarray, optional
        Feature names, used in logs, by `get_selected_features` and in the
        network diagrams.
    preprocessing : nn.Module, optional
        A module applied to every input batch before the first layer.
    class_weights : tensor, optional
        Per-class loss weights for a multi-class model, or the weight of the
        positive class for a binary one. `Trainer` sets them when it is
        given class weights.
    """
    model_class = None

    def __init__(
        self,
        config: Any,
        d_model: int,
        feature_names: list[str] | np.ndarray | None = None,
        preprocessing: nn.Module | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.param = OmegaConf.merge(self.default_config(), config)  # parameters
        self.preprocessing = preprocessing

        if self.param.model.type == "binary":
            assert self.param.model.num_classes == 2
        if self.param.model.type == "multi-class":
            assert self.param.model.num_classes > 2

        # Resolve the heterogeneous `model.ref_functions` list (some entries
        # are bare names, some are single-key mappings carrying options) into
        # an ordered list of (RefFunctionType, options) pairs. A list rather
        # than a dict keyed by type, so the same family may appear more than
        # once with different options (e.g. a polyquad dim=4 *and* a polyquad
        # dim=10), and so per-layer neuron creation follows the YAML order.
        # See `_parse_ref_function_entry` for the accepted entry shapes.
        self.ref_functions: list[tuple[RefFunctionType, Optional[dict]]] = []
        for entry in self.param.model.ref_functions:
            ref_type, options = _parse_ref_function_entry(entry)
            self.ref_functions.append((ref_type, options))

        # What feeds every layer after the first besides the last layer's
        # outputs: the raw inputs and/or the outputs of `shortcut_prev` older
        # layers. See `_parse_shortcut` and `create_layer`.
        self.shortcut_raw, self.shortcut_prev = _parse_shortcut(self.param.model.shortcut)

        # Cache the parsed enum next to the (string-typed) config field. Don't
        # overwrite `self.param.train.criterion_type` — under structured-config
        # typing the field is `str`, so assigning the enum coerces to a string
        # like "CriterionType.cmpValidate" and silently breaks downstream
        # `== CriterionType.cmpValidate` comparisons.
        self.criterion_type = CriterionType.get(self.param.train.criterion_type)

        # Same reason as criterion_type: the schema types this `str`, so this is
        # the only place it gets validated. Cached as a bool because the trainer
        # reads it per evaluation.
        _err_norm = self.param.train.error_normalization
        if _err_norm not in ("variance", "energy"):
            raise ValueError(
                f"train.error_normalization must be 'variance' or 'energy', got {_err_norm!r}"
            )
        self.error_centered = _err_norm == "variance"

        self.feature_names = feature_names       # name of inputs, used to print model
        if isinstance(self.feature_names, np.ndarray):
            self.feature_names = self.feature_names.tolist()

        self.dtype = getattr(torch, self.param.train.dtype)

        self.nbest_neurons = config.model.nbest_neurons        # number of the best neurons to be selected
        assert self.nbest_neurons > 1
        self.layers: nn.ModuleList = nn.ModuleList()        #: :type: list of Layer
        self.d_model = d_model     # number of original features

        self.layer_err: list[float] = []          # array of layer's errors
        self.layer_val_err: list[float] = []      # same on the validation split, when one is given
        self.layer_val_err: list[float] = []      # same on the validation split, when one is given

        cw = class_weights.to(dtype=self.dtype) if class_weights is not None else None

        if self.param.model.type == "regressor":
            self.shared_proj = None
            self.soft_binner = None
            # NormMSE = sum((y - y_pred)^2) / sum((y - y_mean)^2), the same
            # normalization as the regularity criterion so the training loss and
            # the dev-set error live on one scale. Trainer.train replaces the
            # per-batch denominator with a fixed training-set variance; see
            # NormMSE's docstring. `error_normalization: energy` restores the
            # sum(y^2) form used by classical GMDH implementations (gmdhpy etc).
            # Returns a scalar; compute_loss's downstream .mean() is a no-op on
            # a scalar, so the rest of the pipeline is unchanged.
            self.loss_fn = NormMSE(centered=self.error_centered,
                                   censor_at=self.param.train.censor_target_at)

            # Regression out_proj is a Linear(num_out, 1) head — a global
            # linear combiner over the top-k survivor outputs of the last
            # layer. Built only when `use_output_projection: true`; otherwise
            # the model falls back to "best single neuron" inference (the
            # default GMDH path), capped at whatever a single polynomial
            # neuron can express.
            if self.param.model.use_output_projection:
                num_out = self.param.model.num_out_neurons
                if num_out is None:
                    num_out = self.param.model.max_neuron_models
                self.out_proj = nn.Linear(num_out, 1)
            else:
                self.out_proj = None
        elif self.param.model.type == "binary":
            self.shared_proj = None
            self.soft_binner = None
            # cw shape (1,) — pos_weight for the positive class
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=cw, reduction="none")
        elif self.param.model.type == "multi-class":
            assert not (self.param.model.soft_binner and self.param.model.use_neuron_proj), \
                "soft_binner and use_neuron_proj are mutually exclusive"

            num_classes = self.param.model.num_classes
            _scale = self.param.model.soft_binner_scale
            _centers = torch.linspace(0.05, 0.95, num_classes)

            # RBF-derived init: SoftBinner logit_k = -s*(x-c_k)^2 = 2sc*x - sc^2
            # The -sx^2 term cancels in softmax, so weight=2sc, bias=-sc^2.
            _col = (2 * _scale * _centers)  # (num_classes,)
            _bias_init = -_scale * _centers ** 2  # (num_classes,)

            def _make_proj(in_features: int) -> nn.Linear:
                proj = nn.Linear(in_features, num_classes)
                with torch.no_grad():
                    proj.weight.copy_(_col.unsqueeze(-1).expand(-1, in_features))
                    proj.bias.copy_(_bias_init)
                return proj

            if self.param.model.soft_binner:
                self.shared_proj = None
                self.soft_binner = SoftBinner(num_classes, scale=_scale)
            elif self.param.model.use_neuron_proj:
                # Per-neuron projection: each neuron carries its own proj_weight /
                # proj_bias (set via BasePolynomNeuron.set_proj in create_layer).
                # Picked up automatically by create_loss_functions with in_dims=0.
                self.shared_proj = None
                self.soft_binner = None
            else:
                # Default: single shared projection nn.Linear(1, num_classes)
                # trained across all ensemble members jointly (vmap in_dims=None).
                self.shared_proj = _make_proj(1)
                self.soft_binner = None

            # cw shape (num_classes,) — per-class loss weight
            self.loss_fn = nn.NLLLoss(weight=cw, reduction="none")

            if self.param.model.use_output_projection:
                num_out = self.param.model.num_out_neurons
                if num_out is None:
                    num_out = self.param.model.max_neuron_models
                self.out_proj = nn.Linear(num_out, num_classes)
            else:
                self.out_proj = None
        else:
            raise ValueError

        if not hasattr(self, "out_proj"):
            self.out_proj = None

        self.params_metadata_names.extend([
            "layer_err",
            "layer_names",
        ])

    def set_class_weights(self, class_weights: torch.Tensor) -> None:
        """Replace the loss function's class weights without rebuilding the model.

        Safe to call after model.to(device) — the new loss module is moved to
        the model's current device automatically.
        """
        cw = class_weights.to(dtype=self.dtype, device=self.device)
        if isinstance(self.loss_fn, nn.NLLLoss):
            self.loss_fn = nn.NLLLoss(weight=cw, reduction="none").to(device=self.device)
        elif isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=cw, reduction="none").to(device=self.device)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Record the layer class names, then return the module state dict."""
        self.layer_names = [layer.__class__.__name__ for layer in self.layers]
        return super().state_dict(*args, **kwargs)

    def restore_from_checkpoint_metadata(self, checkpoint_data: dict[str, Any]) -> None:
        """
        Rebuild the layer stack from checkpoint metadata.

        Replaces `self.layers` with new layers holding neuron modules of the
        saved classes and shapes. Only the architecture is restored: the
        weights are loaded afterwards with `load_state_dict` (see
        `Trainer.load_model_checkpoint`).

        Parameters
        ----------
        checkpoint_data : dict
            The model's state dict as saved in a checkpoint, including the
            `params_metadata` entries of the model, its layers and their
            neuron modules.
        """
        prefix = ""
        self.layers = nn.ModuleList()
        layer_names = checkpoint_data["params_metadata"]["layer_names"]
        for layer_idx, layer_name in enumerate(layer_names):
            key = f"{prefix}layers.{layer_idx}.params_metadata"
            layer_metadata = checkpoint_data[key]
            layer = SONNLayer(layer_metadata["d_model"], layer_metadata["nbest_neurons"], layer_metadata["layer_index"])
            for neuron_model_idx, neuron_model_name in enumerate(layer_metadata["neuron_models_names"]):
                key = f"{prefix}layers.{layer_idx}.neuron_models.{neuron_model_idx}.params_metadata"
                neuron_model_metadata = checkpoint_data[key]
                neuron_model = BasePolynomNeuron.from_checkpoint_metadata(neuron_model_metadata)
                layer.neuron_models.append(neuron_model)
            self.layers.append(layer)

    @classmethod
    def default_config(cls) -> Any:
        """Return the configuration schema with every default filled in.

        The result is `OmegaConf.structured(SONNConfig)`: a typed `DictConfig`
        with the sections `model` and `train` and a few top-level flags, a
        fresh copy on every call. Merge changes into it:

            config = OmegaConf.merge(SONN.default_config(), {"model": {"nbest_neurons": 8}})

        A key the schema does not have, or a value of the wrong type, is
        rejected when it is merged. `SONNConfig` and its sections document
        every key.

        Returns
        -------
        omegaconf.DictConfig
            The default configuration.
        """
        # Defaults live in the SONNConfig dataclass schema (src/config/
        # schemas.py). OmegaConf.structured turns the dataclass into a
        # type-checked DictConfig with the same shape the old YAML / literal
        # produced, and returns a fresh instance each call so a caller
        # mutating it doesn't poison subsequent SONN(...) constructions.
        return OmegaConf.structured(SONNConfig)

    def __str__(self) -> str:
        return "Self-organizing deep learning polynomial neural network"

    @property
    def device(self) -> torch.device:
        """Device of the first neuron module, or the configured `train.device` before any layer exists."""
        # Walk the layers until we find the first non-empty module list.
        # prune() can leave intermediate layers empty (or, with the
        # `continue`-on-empty guard, untouched) and the old "layers[0]
        # .neuron_models[0]" path raised IndexError in that case.
        for layer in self.layers:
            for nm in layer.neuron_models:
                return nm.device
        # No neuron modules anywhere — happens before train_layer has run
        # the first time. Fall back to the configured device.
        return torch.device(self.param.train.device)

    @property
    def retrain_required(self) -> bool:
        """True for the 'bias_retrain' criterion, which `Trainer.train_layer` does not implement."""
        return self.criterion_type == CriterionType.cmpComb_bias_retrain

    def _get_features_names_by_index(self, features_set: list[int] | set[int]) -> str:
        """Return names of features
        """
        if self.feature_names is None:
            return ', '.join(
                ['index=inp_{0} '.format(idx) for idx in features_set])
        else:
            return ', '.join(
                [self.feature_names[idx] for idx in features_set])

    def get_selected_features_indices(self) -> list[int]:
        """Return features that was selected as useful for neuron during fit
        """
        selected_features_set = set()
        for neuron in self.layers[0]:
            selected_features_set.update(neuron.src_idxs.view(-1).tolist())

        for layer_pos, layer in enumerate(self.layers[1:], start=1):
            if not self.layer_sources(layer_pos)[1]:
                continue
            # The block widths are the *actual* output counts of the source
            # layers, not their static nbest_neurons cap. Post-train_layer
            # they coincide, but after trainer.prune drops modules the cap is
            # stale and would misplace the raw block — silently
            # under-reporting the used features.
            for neuron in layer:
                for idx in neuron.src_idxs.view(-1).tolist():
                    source, local = self.locate(layer_pos, idx)
                    if source is None:
                        selected_features_set.add(local)
        return list(selected_features_set)

    def get_unselected_features_indices(self) -> list[int]:
        """Return features that was not selected as useful for neuron during fit
        """
        return list(set(np.arange(self.d_model).tolist()) -
                    set(self.get_selected_features_indices()))

    def get_unselected_features(self) -> str:
        """Return names of features that was not selected as useful for neuron during fit
        """
        unselected_features = self.get_unselected_features_indices()
        if len(unselected_features) == 0:
            return "No unselected features"
        else:
            return self._get_features_names_by_index(unselected_features)

    def get_selected_features(self) -> str:
        """Return names of features that was selected as useful for neuron during fit
        """
        return self._get_features_names_by_index(self.get_selected_features_indices())

    def plot_layer_error(self) -> None:
        """Plot layer error on validation set vs layer index

        matplotlib is imported here rather than at module scope so it can stay
        an optional dependency — this is the only place in the package that
        touches it, and a base install must still be able to import SONN.
        """
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on install extras
            raise ModuleNotFoundError(
                "SONN.plot_layer_error needs the optional 'matplotlib' "
                "dependency, which is not part of the base install. Add it "
                "with:\n"
                "    pip install \"torchsonn[viz]\""
            ) from exc

        fig = plt.figure()
        y = self.layer_err
        x = list(range(len(y)))
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y, 'b')
        ax1.set_title('Layer error on validate set')
        plt.xlabel('layer index')
        plt.ylabel('error')
        idx = len(self.layers) - 1
        plt.plot(x[idx], y[idx], 'rD')
        plt.show()

    @staticmethod
    def compute_loss(
        neuron_model: nn.Module,
        loss_fn: nn.Module | None,
        model_type: str,
        soft_binner: nn.Module | None,
        ridge_alpha: float,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss of one candidate neuron, written for `torch.func` transforms.

        Runs `neuron_model` with the given parameters and buffers through
        `functional_call`, so the trainer can `vmap` it over a candidate
        ensemble and take `grad` with respect to `params`. For multi-class
        models the scalar output is turned into logits by the soft binner,
        the shared projection or the neuron's own projection, whichever the
        parameters provide.

        Parameters
        ----------
        neuron_model : nn.Module
            The neuron module whose forward is called.
        loss_fn : nn.Module or None
            Per-sample loss. None returns the raw predictions (logits for
            multi-class) instead of a loss.
        model_type : str
            `model.type`; "multi-class" selects the logits path.
        soft_binner : nn.Module or None
            Maps the scalar output to class logits, if the model uses one.
        ridge_alpha : float
            L2 penalty on `params["weight"]`, added to the loss when > 0.
            Projection weights are not penalized.
        params : dict of str to tensor
            Parameters of one candidate (one slice of the ensemble under
            `vmap`).
        buffers : dict of str to tensor
            Buffers of the neuron module.
        x : (B, D) tensor
            Neuron input.
        y : (B,) tensor
            Targets (class indices for multi-class).

        Returns
        -------
        out : tensor
            Mean loss plus the ridge penalty as a scalar, or the predictions
            when `loss_fn` is None.
        """
        pred = functional_call(neuron_model, {**params, **buffers}, args=(x,))
        if model_type == "multi-class":
            if soft_binner:
                logits = soft_binner(pred)
            elif "shared_proj_weight" in params:
                # shared_proj: Linear(1, C), weight (C, 1) broadcast to all neurons.
                # pred (batch,) → (batch, 1) * (C,) = (batch, C).
                logits = (
                    pred.unsqueeze(-1) * params["shared_proj_weight"].squeeze(-1)
                    + params["shared_proj_bias"]
                )
            else:
                # neuron proj (proj_weight / proj_bias on the neuron module).
                # Each neuron has its own (C,) weight + bias (in_dims=0).
                # pred (batch,) → (batch, 1) * (C,) = (batch, C).
                logits = (
                    pred.unsqueeze(-1) * params["proj_weight"]
                    + params["proj_bias"]
                )
            if loss_fn is None:
                return logits
            out = loss_fn(F.log_softmax(logits, dim=-1), y).mean()
        else:
            # Regression / binary path: inside vmap pred is (batch,) (see the
            # squeeze in BasePolynomNeuron.forward — bug #17 fix). MSELoss and
            # BCEWithLogitsLoss take matching shapes, so compare directly
            # against y (batch,). .permute(1, 0) on a 1-D tensor errors with
            # "Dimension out of range".
            if loss_fn is None:
                return pred
            out = loss_fn(pred, y.to(pred.dtype)).mean()

        # Optional L2 / ridge penalty on the per-neuron polynomial weight,
        # added to the training-loss partial only. Under vmap `params["weight"]`
        # is the single-neuron slice (num_w,), so `(...**2).sum()` is a
        # per-ensemble scalar that broadcasts correctly through grad/vmap.
        # Projection weights (shared_proj / proj_weight / proj_bias) are
        # deliberately excluded — ridge here regularizes the OLS coefficient
        # fit, which is what the user controls via train.ridge_alpha.
        if ridge_alpha > 0.0 and "weight" in params:
            out = out + ridge_alpha * (params["weight"] ** 2).sum()
        return out

    @property
    def need_bias_err(self) -> bool:
        """True when the selection criterion uses the bias error ('bias', 'validate_bias', 'bias_retrain')."""
        return self.criterion_type in (
            CriterionType.cmpBias,
            CriterionType.cmpComb_validate_bias,
            CriterionType.cmpComb_bias_retrain,
        )

    @property
    def need_regularity_err(self) -> bool:
        """True when the selection criterion uses the regularity (validation) error ('validate', 'validate_bias')."""
        return self.criterion_type in (
            CriterionType.cmpValidate,
            CriterionType.cmpComb_validate_bias,
        )

    def get_error(
        self,
        criterion_type: CriterionType,
        regularity_err: torch.Tensor | None,
        bias_err: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute error of the neuron according to specified criterion
        """
        if criterion_type == CriterionType.cmpValidate:
            return regularity_err
        elif criterion_type == CriterionType.cmpBias:
            return bias_err
        elif criterion_type == CriterionType.cmpComb_validate_bias:
            alpha = self.param.train.error_alpha
            return (1.0 - alpha) * bias_err + alpha * regularity_err
        elif criterion_type == CriterionType.cmpComb_bias_retrain:
            return bias_err
        else:
            raise NotImplementedError

    # -- input layout -------------------------------------------------------
    #
    # The input of layer j >= 1 is the concatenation
    #     [h_{j-1} | h_{j-2} | ... | h_{j-1-k} | x_raw]
    # of the clamped outputs of the layers listed in its `input_layers`
    # (nearest first; the last layer always, plus `shortcut_prev` = k older
    # ones) followed by the raw model inputs when `input_raw`. Layer 0 reads
    # x_raw alone. With k = 0 this is the original [h_{j-1} | x_raw] layout.

    @property
    def concat_inputs(self) -> bool:
        """True when a layer's input is more than the previous layer's output.

        False only for `shortcut: false` (no raw features, no older layers),
        the configuration in which the last layer's LayerNorm also applies to
        its own output at inference — see `forward`.
        """
        return self.shortcut_raw or self.shortcut_prev > 0

    def new_layer_sources(self, position: int) -> tuple[list[int], bool]:
        """(input_layers, input_raw) the config gives a layer created at `position`."""
        if position == 0:
            return [], True
        oldest = max(0, position - 1 - self.shortcut_prev)
        return list(range(position - 1, oldest - 1, -1)), self.shortcut_raw

    def layer_sources(self, position: int) -> tuple[list[int], bool]:
        """(input_layers, input_raw) of the layer at `position`, as stored on it."""
        layer = self.layers[position]
        return list(layer.input_layers), bool(layer.input_raw)

    def input_blocks(self, position: int) -> list[tuple[Optional[int], int]]:
        """(source, width) per block of the input of the layer at `position`,
        in concatenation order. `source` is a layer position, or None for the
        raw model inputs."""
        input_layers, input_raw = self.layer_sources(position)
        blocks: list[tuple[Optional[int], int]] = [(s, len(self.layers[s])) for s in input_layers]
        if input_raw:
            blocks.append((None, self.d_model))
        return blocks

    def locate(self, position: int, flat_idx: int) -> tuple[Optional[int], int]:
        """Map an index into the input of the layer at `position` to
        (source, index inside that source's block); source None = raw inputs."""
        offset = 0
        for source, width in self.input_blocks(position):
            if flat_idx < offset + width:
                return source, flat_idx - offset
            offset += width
        raise IndexError(f"input index {flat_idx} out of range for layer at position {position} (width {offset})")

    def next_input_width(self) -> int:
        """Input width of a layer created next, at position len(self.layers)."""
        input_layers, input_raw = self.new_layer_sources(len(self.layers))
        return sum(len(self.layers[s]) for s in input_layers) + (self.d_model if input_raw else 0)

    def forward(self, x: torch.Tensor, skip_last_layer: bool = False) -> torch.Tensor:
        """
        Run the input through the layer stack.

        Applies the preprocessing module, if any, then each layer on the
        input its `layer_sources` describe (outputs of earlier layers and,
        under `shortcut.raw_features`, the raw features). Layer outputs are
        clamped to `model.output_clamp_value`. No output head is applied:
        `infer` does that.

        Parameters
        ----------
        x : (B, d_model) tensor
            Input features.
        skip_last_layer : bool
            Stop at the input of the last layer and return it. The trainer
            uses this while the last layer is being fitted. Default False.

        Returns
        -------
        out : tensor
            The last layer's output, shape (B, number of neurons in it), or
            its input when `skip_last_layer` is True. With no layers, the
            preprocessed input.
        """
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        x_inp = x
        clamp = self.param.model.output_clamp_value
        n_layers = len(self.layers)

        # During train_layer the last layer is the one being fitted, so we
        # stop the SONN forward at its input and let the candidate-neuron
        # ensemble (vmap'd over its own params) consume that feature map
        # directly. `skip_last_layer=True` is what the trainer passes in that
        # context; inference / .infer() leaves it False.
        sources = [self.layer_sources(j) for j in range(n_layers)]
        # Each output is kept until the last layer that reads it has its input.
        last_reader: dict[int, int] = {}
        for j, (input_layers, _) in enumerate(sources):
            for s in input_layers:
                last_reader[s] = j

        outs: dict[int, torch.Tensor] = {}
        for j, layer in enumerate(self.layers):
            if j > 0:
                input_layers, input_raw = sources[j]
                parts = [outs[s] for s in input_layers]
                if input_raw:
                    parts.append(x_inp)
                x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
                # Per-layer LayerNorm (if enabled) standardizes the feature
                # tensor that feeds the next layer — explicitly AFTER the
                # shortcut concat so the raw inputs and older layers' outputs
                # get folded into the normalization.
                prev = self.layers[j - 1]
                if prev.layer_norm is not None:
                    x = prev.layer_norm(x)
                for s in input_layers:
                    if last_reader[s] == j:
                        del outs[s]
            if skip_last_layer and j == n_layers - 1:
                return x
            outs[j] = torch.clamp(layer(x), -clamp, clamp)

        if n_layers == 0:
            return x
        x = outs[n_layers - 1]
        # The inference-time last layer feeds no further layer. Its LayerNorm
        # is sized for a next layer's input, which only equals its own output
        # when nothing is concatenated (`shortcut: false`); then it applies
        # here too, as it did in training. Otherwise out_proj consumes the raw
        # clamped output, matching how it trained.
        last = self.layers[-1]
        if last.layer_norm is not None and not self.concat_inputs:
            x = last.layer_norm(x)
        return x

    def get_best_neuron_model(self, layer: SONNLayer) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Locate the neuron with the smallest error in a layer.

        Parameters
        ----------
        layer : SONNLayer
            A trained layer, with `err_values` and `module_idxs` set.

        Returns
        -------
        module_idx : tensor
            Index of the neuron module that holds the best neuron.
        neuron_idx : tensor
            Index of the best neuron inside that module.
        """
        smallest_err_idx = layer.err_values.topk(1, largest=False)[1]
        best_module_idx, best_neuron_idx = layer.module_idxs[smallest_err_idx][0]
        return best_module_idx, best_neuron_idx

    def _best_neuron_column(self, layer: SONNLayer) -> int:
        """Cumulative column of the best neuron in the layer's concatenated output."""
        best_module_idx, best_neuron_idx = self.get_best_neuron_model(layer)
        col = 0
        for i, m in enumerate(layer.neuron_models):
            if i == int(best_module_idx):
                return col + int(best_neuron_idx)
            col += m.num_neurons
        return col

    def _best_column_cached(self, layer: SONNLayer) -> int:
        """`_best_neuron_column(layer)` as a plain int, cached under the same
        key as `_head_columns` (the headless readout). Computing it reads
        `err_values` and `module_idxs` on the host, a sync that is not
        permitted while a CUDA graph is being captured."""
        err = layer.err_values
        key = ("best", id(layer), err.data_ptr(), int(err.shape[0]),
               tuple(int(m.num_neurons) for m in layer.neuron_models))
        cached = getattr(self, "_best_column_cache", None)
        if cached is None or cached[0] != key:
            cached = (key, int(self._best_neuron_column(layer)))
            self._best_column_cache = cached
        return cached[1]

    def _head_columns(self, layer: SONNLayer, k: int) -> torch.Tensor:
        """`_best_neuron_columns(layer, k)` as a long tensor on the model's
        device, cached. Recomputing the list on every `infer` call cost a
        host-to-device copy per forward (and made the forward uncapturable
        in a CUDA graph). The cache key is everything the list derives from:
        the layer object, `k`, the identity and length of `layer.err_values`
        (replaced by selection and by `prune`), the per-module neuron counts
        (changed by `prune`) and the device.
        """
        err = layer.err_values
        key = (id(layer), int(k), err.data_ptr(), int(err.shape[0]),
               tuple(int(m.num_neurons) for m in layer.neuron_models), str(self.device))
        cached = getattr(self, "_head_columns_cache", None)
        if cached is None or cached[0] != key:
            cols = torch.as_tensor(self._best_neuron_columns(layer, k), dtype=torch.long, device=self.device)
            cached = (key, cols)
            self._head_columns_cache = cached
        return cached[1]

    def _readout_width(self, layer: SONNLayer) -> int:
        """How many of the last layer's columns `infer` reads.

        The head's inputs (`out_proj.in_features`, at most the layer's
        width); every column for a headless multi-class model with
        per-neuron projections (`use_neuron_proj`), whose prediction sums
        every neuron's projected output; otherwise the single best-error
        column. `Trainer.prune` keeps exactly this many columns.
        """
        width = len(layer)
        if self.out_proj is not None:
            return min(int(self.out_proj.in_features), width)
        if isinstance(self.loss_fn, nn.NLLLoss) and self.param.model.use_neuron_proj:
            return width
        return 1

    def _best_neuron_columns(self, layer: SONNLayer, k: int) -> list[int]:
        """Cumulative column indices of the top-k lowest-error neurons."""
        k = min(k, layer.err_values.shape[0])
        _, top_indices = layer.err_values.topk(k, largest=False)
        cols = []
        for idx in top_indices:
            module_idx = int(layer.module_idxs[idx, 0])
            neuron_idx = int(layer.module_idxs[idx, 1])
            col = 0
            for i, m in enumerate(layer.neuron_models):
                if i == module_idx:
                    col += neuron_idx
                    break
                col += m.num_neurons
            cols.append(col)
        return cols

    def create_layer(self, layer_index: int) -> SONNLayer:
        """Generate new layer with all possible neurons
        """
        logger.info(f"Creating layer #{layer_index}")
        layers_count = len(self.layers)
        layer = SONNLayer(
            self.d_model,
            self.nbest_neurons,
            layers_count,
            use_layer_norm=self.param.model.use_layer_norm,
        )

        # The first layer reads the original features. Every other layer reads
        # the selected neurons of the previous layer, of `shortcut.prev_layers`
        # older layers and, with `shortcut.raw_features`, the original
        # features again (see the input layout above `forward`).
        layer.input_layers, layer.input_raw = self.new_layer_sources(layers_count)
        n = self.next_input_width()

        # number of all possible combination of input pairs is N = (n * (n-1)) / 2
        # add all neurons to the layer
        #
        # Per-neuron-type activation: each ref_function entry can carry an
        # `activation:` key in its options dict. Entries that omit it fall
        # back to `None`, which BasePolynomNeuron.__init__ treats the same
        # as `""` — `nn.Identity()`. Example:
        #     ref_functions:
        #       - linear_cov                       # → Identity (default)
        #       - cubic: {activation: relu}        # → ReLU
        #       - polyquad:
        #           squares: true
        #           dim: 5
        #           activation: tanh               # → Tanh

        def _make_neuron_args(options: Optional[dict]) -> tuple[tuple, dict]:
            """Build (args, kwargs) for one ref-function entry.

            Pops `activation` from the entry's option dict; falls back to
            None (Identity) when unset. Everything else in options becomes a
            kwarg to the neuron constructor (e.g. polyquad's `dim` / `squares`).
            """
            # Copy so popping doesn't mutate the options dict cached on
            # self.ref_functions — it's parsed once in SONN.__init__ and reused
            # on every layer, so we'd otherwise drain it.
            options = dict(options) if isinstance(options, dict) else {}
            activation = options.pop("activation", None)
            args = (n, self.d_model, activation, layer_index, len(layer))
            kwargs = {"max_neuron_models": self.param.model.max_neuron_models, **options}
            return args, kwargs

        # RefFunctionType → neuron class. The loop walks self.ref_functions in
        # YAML order, so each entry — including repeated families such as two
        # polyquads with different `dim`s — builds its own neuron.
        neuron_cls_by_type = {
            RefFunctionType.rfLinear:        LinearPolynomNeuron,     # y = w0 + w1 x1 + w2 x2
            RefFunctionType.rfLinearCov:     LinearCovPolynomNeuron,  # + w3 x1 x2
            RefFunctionType.rfQuadratic:     QuadraticPolynomNeuron,  # full 2nd degree
            RefFunctionType.rfCubic:         CubicPolynomNeuron,      # full 3rd degree
            RefFunctionType.rfPolyQuadratic: PolyQuadratic,           # 2nd degree over `dim` inputs
            RefFunctionType.rfLegendre:      LegendrePolynomNeuron,   # Legendre basis, degree `degree`
            RefFunctionType.rfChebyshev:     ChebyshevPolynomNeuron,  # Chebyshev basis, degree `degree`
            RefFunctionType.rfRBF:           RBFNeuron,               # Gaussian bumps, `centers` per neuron
        }

        neuron_models = []
        for ref_type, options in self.ref_functions:
            neuron_cls = neuron_cls_by_type[ref_type]
            a, kw = _make_neuron_args(options)
            if issubclass(neuron_cls, BaseOrthogonalNeuron):
                # Only the orthogonal-polynomial families squash their inputs;
                # handing these kwargs to e.g. LinearPolynomNeuron would be a
                # TypeError. setdefault so a per-entry override in the YAML
                # (`- legendre: {squash_method: tanh}`) still wins over the
                # model-wide default.
                kw.setdefault("squash_method", self.param.model.squash_method)
                kw.setdefault("squash_n_sigma", self.param.model.squash_n_sigma)
                kw.setdefault("squash_core_range", self.param.model.squash_core_range)
            neuron_models.append(neuron_cls(*a, **kw))

        # Drop any family whose required input arity exceeds the number of
        # inputs available to this layer (n). A polyquad(dim=k) needs k distinct
        # inputs; the pair-based families need 2. When n is smaller its
        # candidate enumeration is empty (create_src_idxs → 0 index-tuples) and
        # the module holds 0 neurons, which would otherwise crash in forward on
        # an empty (float-typed) index tensor. Skipping lets a too-wide polyquad
        # sit out the early, narrow layers and join once `shortcut` widening
        # supplies enough inputs deeper in the network.
        for nm in neuron_models:
            if nm.num_neurons == 0:
                logger.warning(
                    "Layer #%d has %d input(s), fewer than the %d that %s "
                    "requires; skipping this neuron family for the layer.",
                    layer_index, n, nm.dim, nm.get_short_name(),
                )
        neuron_models = [nm for nm in neuron_models if nm.num_neurons > 0]

        neuron_models = [module.to(device=self.param.train.device) for module in neuron_models]
        if len(neuron_models) == 0:
            raise LayerCreationError(
                'Error creating layer. No functions were created',
                layer.layer_index,
            )

        if self.param.model.use_neuron_proj:
            num_classes = self.param.model.num_classes
            for nm in neuron_models:
                nm.set_proj(num_classes)

        layer.neuron_models.extend(neuron_models)

        return layer

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict with the trained model.

        Moves `x` to the model's device, runs `forward` and reads the
        prediction off the last layer: through the output head
        (`out_proj`) if the model has one, otherwise from the best-error
        neuron (regression and binary), the per-neuron projections or the
        shared projection / soft binner (multi-class).

        Parameters
        ----------
        x : (B, d_model) tensor
            Input features.

        Returns
        -------
        pred : tensor
            (B,) predictions for regression and binary models (logits for
            binary), or (B, C) log-probabilities for multi-class models.
        """
        out = self(x.to(device=self.device))
        if self.out_proj is not None:
            k = self.out_proj.in_features
            cols = self._head_columns(self.layers[-1], k)
            selected = out.index_select(1, cols)
            # pad with zeros if fewer neurons available than out_proj expects
            if cols.numel() < k:
                pad = torch.zeros(selected.shape[0], k - cols.numel(), device=selected.device, dtype=selected.dtype)
                selected = torch.cat([selected, pad], dim=-1)
            proj_out = self.out_proj(selected)
            if self.param.model.type == "multi-class":
                return F.log_softmax(proj_out, dim=-1)
            # Regression / binary: out_proj is Linear(num_out, 1) — squeeze
            # the trailing singleton so the result is (N,) matching the
            # raw-target shape that downstream report() / metrics expect.
            return proj_out.squeeze(-1)

        if not isinstance(self.loss_fn, nn.NLLLoss):
            # Headless regressor / binary: `out` is still the full nbest
            # ensemble, shape (N, num_neurons). The model's prediction is
            # the best-error neuron's output — the column the layer error
            # was scored on and the one `prune()` keeps — so return it as
            # (N,), matching the raw-target shape like the out_proj path.
            return out[:, self._best_column_cached(self.layers[-1])]

        if self.param.model.use_neuron_proj:
            # Collect per-neuron proj_weight / proj_bias from the final layer
            # and combine: out (batch, nbest) @ W (nbest, C) + b (C,). Every
            # column counts, so `prune` keeps them all (`_readout_width`).
            W = torch.cat([nm.proj_weight for nm in self.layers[-1].neuron_models], dim=0)
            b = torch.cat([nm.proj_bias   for nm in self.layers[-1].neuron_models], dim=0).mean(dim=0)
            return F.log_softmax(out @ W + b, dim=-1)

        # shared_proj / soft_binner: use single best-neuron scalar output.
        # After train_layer neurons are sorted by module idx (not error), so
        # column 0 is not always the best. After a full prune it collapses to 0.
        scalar = out[:, self._best_neuron_column(self.layers[-1])]
        if self.soft_binner:
            logits = self.soft_binner(scalar)
        else:
            logits = self.shared_proj(scalar.unsqueeze(dim=-1))
        return F.log_softmax(logits, dim=-1)





