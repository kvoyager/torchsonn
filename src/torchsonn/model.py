import numpy as np
import sys

import torch
from torch import nn

from collections.abc import Mapping
from typing import Any, Optional

from omegaconf import OmegaConf, DictConfig, ListConfig

# Side-effect import: registers SONNConfig with Hydra's ConfigStore so the
# tutorial YAMLs' `defaults: [default, _self_]` resolves to the typed schema.
from torchsonn.config import SONNConfig
from torch.func import vmap, grad, functional_call
from torchsonn.modules import SONNModule, SoftBinner

import torch.nn.functional as F
from torchsonn.layer import SONNLayer
from torchsonn.neurons import (
    BasePolynomNeuron,
    LinearPolynomNeuron,
    LinearCovPolynomNeuron,
    QuadraticPolynomNeuron,
    CubicPolynomNeuron,
    PolyQuadratic,
)
from torchsonn.loss import NormMSE
from torchsonn.types import RefFunctionType, CriterionType, LayerCreationError
import logging


def _parse_ref_function_entry(entry: Any) -> tuple[RefFunctionType, Optional[dict]]:
    """Parse a single entry from `model.ref_functions`.

    The list is intentionally heterogeneous (typed `List[Any]` in the config)
    because each ref-function family takes its own option set. Two shapes are
    accepted, both routinely emitted by hand-written YAML:

      • a bare name string  →  options = None
            - linear_cov
            - quadratic

      • a single-key mapping carrying options, e.g.
            - polyquad:
                squares: true
                dim: 3

        or the legacy list-of-single-key-dicts form (still in older YAMLs):
            - polyquad:
                - squares: true
                - dim: 3

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
        if isinstance(raw, (Mapping, DictConfig)):
            # Modern nested-mapping form.
            options = {k: v for k, v in raw.items()}
        elif isinstance(raw, (list, ListConfig)):
            # Legacy list-of-single-key-dicts form. Flatten into one dict.
            options = {k: v for item in raw for k, v in dict(item).items()}
        else:
            raise TypeError(
                f"ref_functions[{name!r}] must be a mapping or a list of "
                f"single-key mappings, got {type(raw).__name__}"
            )
        return RefFunctionType.get(name), options
    # Bare-name form (string or RefFunctionType).
    return RefFunctionType.get(entry), None


logger = logging.getLogger(__name__)


class SONN(SONNModule):
    """Base class for self-organizing deep learning polynomial neural network
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
        # a uniform dict keyed by RefFunctionType. See
        # `_parse_ref_function_entry` for the accepted shapes.
        self.ref_functions: dict[RefFunctionType, Optional[dict]] = {}
        for entry in self.param.model.ref_functions:
            ref_type, options = _parse_ref_function_entry(entry)
            self.ref_functions[ref_type] = options

        # Cache the parsed enum next to the (string-typed) config field. Don't
        # overwrite `self.param.train.criterion_type` — under structured-config
        # typing the field is `str`, so assigning the enum coerces to a string
        # like "CriterionType.cmpValidate" and silently breaks downstream
        # `== CriterionType.cmpValidate` comparisons.
        self.criterion_type = CriterionType.get(self.param.train.criterion_type)

        self.feature_names = feature_names       # name of inputs, used to print model
        if isinstance(self.feature_names, np.ndarray):
            self.feature_names = self.feature_names.tolist()

        self.dtype = getattr(torch, self.param.train.dtype)

        self.nbest_neurons = config.model.nbest_neurons        # number of the best neurons to be selected
        assert self.nbest_neurons > 1
        self.layers: nn.ModuleList = nn.ModuleList()        #: :type: list of Layer
        self.d_model = d_model     # number of original features

        self.layer_err: list[float] = []          # array of layer's errors

        cw = class_weights.to(dtype=self.dtype) if class_weights is not None else None

        if self.param.model.type == "regressor":
            self.shared_proj = None
            self.soft_binner = None
            # NormMSE = sum((y - y_pred)^2) / (sum(y^2) + eps) —
            # normalized criterion used by classical GMDH implementations
            # (gmdhpy etc). Returns a scalar; compute_loss's downstream .mean()
            # is a no-op on a scalar, so the rest of the pipeline is unchanged.
            self.loss_fn = NormMSE()

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
        self.layer_names = [layer.__class__.__name__ for layer in self.layers]
        return super().state_dict(*args, **kwargs)

    def restore_from_checkpoint_metadata(self, checkpoint_data: dict[str, Any]) -> None:
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
        """Parameters of self-organizing deep learning polynomial neural network
        ----------------------------
        shortcut - if set to true the original features will be added to the list of features of each layer
            default value is true

        criterion_type - criterion for selecting the best neurons
        the following criteria are possible:
            'validate': the default value,
                neurons are compared on the basis of validate error
            'bias': neurons are compared on the basis of bias error
            'validate_bias': combined criterion, neurons are compared on the basis of bias and validate errors
            'bias_retrain': firstly, neurons are compared on the basis of bias error, then neurons are retrain
                on the total data set (train and validate)
        example of using:
            model = Regressor(criterion_type='bias_retrain')

        max_layer_count - maximum number of layers,
            the default value is infinite (sys.maxsize)

        criterion_minimum_width - minimum number of layers at the right required to evaluate optimal number of layer
            (the optimal neuron) according to the minimum of criteria. For example, if it is found that
             criterion value has minimum at layer with index 10, the algorithm will proceed till the layer
             with index 15
             the default value is 5

        stop_train_epsilon_condition - the threshold to stop train. If the layer relative training error in compare
            with minimum layer error becomes smaller than stop_train_epsilon_condition the train is stopped. Default value is
            0.001

        manual_best_neurons_selection - if this value set to False, the number of best neurons to be
            selected is determined automatically and it is equal to the number of original features.
            Otherwise the number of best neurons to be selected is determined as
            max(original features, min_best_neurons_count) but not more than max_best_neurons_count.
            min_best_neurons_count (default 5) or max_best_neurons_count (default inf) has to be provided.
            For example, if you have N=10 features, the number of all generated neurons will be
            N*(N-1)/2=45, the number of selected best neurons will be 10, but you can increase this number to
            20 by setting manual_min_l_count_value = True and min_best_neurons_count = 20.
            If you have N=100 features, the number of all generated neurons will be
            N*(N-1)/2=4950, by default the number of partial neurons passed to the second layer is equal to the number of
            features = 100. If you want to reduce this number for some smaller number, 50 for example, set
            manual_best_neurons_selection=True and max_best_neurons_count=50.
            Note: if min_best_neurons_count is larger than number of generated neurons of the layer it will be reduced
            to that number
        example of using:
            model = Regressor(manual_best_neurons_selection=True, min_best_neurons_count=20)
            or
            model = Regressor(manual_best_neurons_selection=True, max_best_neurons_count=50)

        ref_function_types - set of reference functions, by default the set contains linear combination of two inputs
            and covariation: y = w0 + w1*x1 + w2*x2 + w3*x1*x2
            you can add other reference functions:
            'linear': y = w0 + w1*x1 + w2*x2
            'linear_cov': y = w0 + w1*x1 + w2*x2 + w3*x1*x2
            'quadratic': full polynom of the 2-nd degree
            'cubic': - full polynom of the 3-rd degree
            examples of using:
             - Regressor(ref_functions='linear')
             - Regressor(ref_functions=('linear_cov', 'quadratic', 'cubic', 'linear'))
             - Regressor(ref_functions=('quadratic', 'linear'))

        normalize - scale and normalize features if set to True. Default value is True

        layer_err_criterion - criterion of layer error calculation: 'top' - the topmost best neuron error is chosen
            as layer error; 'avg' - the layer error is the average error of the selected best neurons
            default value is 'top'

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

        if self.param.model.shortcut and len(self.layers) > 1:
            for layer_pos, layer in enumerate(self.layers[1:], start=1):
                # Use the *actual* prev-layer output count, not its static
                # nbest_neurons cap. Post-train_layer they coincide, but after
                # trainer.prune drops modules the cap is stale and would let
                # shortcut indices (which start at len(prev_layer)) slip past
                # the filter — silently under-reporting the used features.
                prev_layer = self.layers[layer_pos - 1]
                shortcut_offset = len(prev_layer)
                for neuron in layer:
                    src_idxs = [x for x in (neuron.src_idxs.view(-1) - shortcut_offset).tolist() if x >= 0]
                    selected_features_set.update(src_idxs)
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
        return self.criterion_type in (
            CriterionType.cmpBias,
            CriterionType.cmpComb_validate_bias,
            CriterionType.cmpComb_bias_retrain,
        )

    @property
    def need_regularity_err(self) -> bool:
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

    def forward(self, x: torch.Tensor, skip_last_layer: bool = False) -> torch.Tensor:
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        x_inp = x

        # During train_layer the last layer is the one being fitted, so we
        # stop the SONN forward before it and let the candidate-neuron
        # ensemble (vmap'd over its own params) consume the resulting feature
        # map directly. `skip_last_layer=True` is what the trainer passes in
        # that context; inference / .infer() leaves it False.
        layers = self.layers[: -1] if skip_last_layer else self.layers

        for idx, layer in enumerate(layers):
            x = layer(x)
            x = torch.clamp(x, -self.param.model.output_clamp_value, self.param.model.output_clamp_value)
            apply_shortcut = self.param.model.shortcut and (skip_last_layer or idx < len(self.layers) - 1)
            # if self.param.model.shortcut and idx < len(self.layers) - 1:
            if apply_shortcut:
                # if the model has shortcut the input for the layer will be [prev_layer_output, model_input]
                x = torch.cat([x, x_inp], dim=-1)
            # Per-layer LayerNorm (if enabled) standardizes the feature
            # tensor that feeds the next layer — explicitly AFTER the
            # shortcut concat so the raw input features get folded into
            # the normalization. Skipped on the inference-time last layer
            # (no shortcut applied there, so x has the un-cat'd width which
            # wouldn't match the LayerNorm's `normalized_shape`); out_proj
            # consumes the raw clamped output, matching how it trained.
            if layer.layer_norm is not None and (apply_shortcut or not self.param.model.shortcut):
                x = layer.layer_norm(x)
        return x

    def get_best_neuron_model(self, layer: SONNLayer) -> tuple[torch.Tensor, torch.Tensor]:
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

        if layers_count == 0:
            # the first layer, number of inputs equals to the number of the original features
            n = self.d_model
        else:
            # all other layers: number of inputs equals to the number of selected
            # neurons from the previous layer plus number of the original
            # features if param.shortcut is True
            n = self.layers[-1].d_model
            if self.param.model.shortcut:
                n += self.d_model

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

        def _make_neuron_args(ref_type: RefFunctionType) -> tuple[tuple, dict]:
            """Build (args, kwargs) for one ref-function family.

            Pops `activation` from the per-type options dict; falls back to
            None (Identity) when the entry doesn't set it. Everything else
            in options becomes a kwarg to the neuron constructor.
            """
            raw_options = self.ref_functions.get(ref_type) or {}
            # Copy so popping doesn't mutate the cached options dict on
            # subsequent layers — `_parse_ref_function_entry` caches it
            # once in SONN.__init__ and we'd otherwise drain it.
            options = dict(raw_options) if isinstance(raw_options, dict) else {}
            activation = options.pop("activation", None)
            args = (n, self.d_model, activation, layer_index, len(layer))
            kwargs = {"max_neuron_models": self.param.model.max_neuron_models, **options}
            return args, kwargs

        neuron_models = []

        # y = w0 + w1*x1 + w2*x2
        if RefFunctionType.rfLinear in self.ref_functions:
            a, kw = _make_neuron_args(RefFunctionType.rfLinear)
            neuron_models.append(LinearPolynomNeuron(*a, **kw))

        # y = w0 + w1*x1 + w2*x2 + w3*x1*x2
        if RefFunctionType.rfLinearCov in self.ref_functions:
            a, kw = _make_neuron_args(RefFunctionType.rfLinearCov)
            neuron_models.append(LinearCovPolynomNeuron(*a, **kw))

        # y = full polynom of the 2-nd degree
        if RefFunctionType.rfQuadratic in self.ref_functions:
            a, kw = _make_neuron_args(RefFunctionType.rfQuadratic)
            neuron_models.append(QuadraticPolynomNeuron(*a, **kw))

        # y = full polynom of the 3-rd degree
        if RefFunctionType.rfCubic in self.ref_functions:
            a, kw = _make_neuron_args(RefFunctionType.rfCubic)
            neuron_models.append(CubicPolynomNeuron(*a, **kw))

        # y = polynom of the 2-rd degree with n inputs (squares / dim from
        # the options dict; activation handled by `_make_neuron_args`).
        if RefFunctionType.rfPolyQuadratic in self.ref_functions:
            a, kw = _make_neuron_args(RefFunctionType.rfPolyQuadratic)
            neuron_models.append(PolyQuadratic(*a, **kw))

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
        out = self(x.to(device=self.device))
        if self.out_proj is not None:
            k = self.out_proj.in_features
            cols = self._best_neuron_columns(self.layers[-1], k)
            selected = out[:, cols]
            # pad with zeros if fewer neurons available than out_proj expects
            if len(cols) < k:
                pad = torch.zeros(selected.shape[0], k - len(cols), device=selected.device, dtype=selected.dtype)
                selected = torch.cat([selected, pad], dim=-1)
            proj_out = self.out_proj(selected)
            if self.param.model.type == "multi-class":
                return F.log_softmax(proj_out, dim=-1)
            # Regression / binary: out_proj is Linear(num_out, 1) — squeeze
            # the trailing singleton so the result is (N,) matching the
            # raw-target shape that downstream report() / metrics expect.
            return proj_out.squeeze(-1)

        if not isinstance(self.loss_fn, nn.NLLLoss):
            return out

        if self.param.model.use_neuron_proj:
            # Collect per-neuron proj_weight / proj_bias from the pruned final
            # layer and combine: out (batch, nbest) @ W (nbest, C) + b (C,).
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





