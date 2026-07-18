import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from torchsonn.config import SONNConfig
from torchsonn.layer import SONNLayer
from torchsonn.model import SONN, _parse_ref_function_entry
from torchsonn.neurons import LinearCovPolynomNeuron, LinearPolynomNeuron
from torchsonn.types import CriterionType, LayerCreationError, RefFunctionType


def _make_cfg(**overrides) -> OmegaConf:
    base = OmegaConf.structured(SONNConfig)
    merged = OmegaConf.merge(base, OmegaConf.create(overrides))
    return merged


class TestParseRefFunctionEntry:
    def test_bare_string(self):
        rt, opts = _parse_ref_function_entry("linear_cov")
        assert rt == RefFunctionType.rfLinearCov
        assert opts is None

    def test_enum_passthrough(self):
        rt, opts = _parse_ref_function_entry(RefFunctionType.rfQuadratic)
        assert rt == RefFunctionType.rfQuadratic
        assert opts is None

    def test_mapping_with_options(self):
        rt, opts = _parse_ref_function_entry({"polyquad": {"dim": 3, "squares": False}})
        assert rt == RefFunctionType.rfPolyQuadratic
        assert opts == {"dim": 3, "squares": False}

    def test_mapping_with_none(self):
        rt, opts = _parse_ref_function_entry({"polyquad": None})
        assert rt == RefFunctionType.rfPolyQuadratic
        assert opts is None

    def test_legacy_list_form(self):
        rt, opts = _parse_ref_function_entry(
            {"polyquad": [{"squares": True}, {"dim": 4}]}
        )
        assert rt == RefFunctionType.rfPolyQuadratic
        assert opts == {"squares": True, "dim": 4}

    def test_invalid_mapping_payload_raises(self):
        with pytest.raises(TypeError):
            _parse_ref_function_entry({"polyquad": 42})

    def test_omegaconf_dict_form(self):
        cfg = OmegaConf.create({"cubic": {"activation": "relu"}})
        # When iterated, an OmegaConf DictConfig becomes a dict-like mapping.
        entry = cfg
        rt, opts = _parse_ref_function_entry(entry)
        assert rt == RefFunctionType.rfCubic
        assert dict(opts) == {"activation": "relu"}


class TestSONNInit:
    def test_regressor_init(self):
        cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                                "soft_binner": False, "ref_functions": ["linear_cov"]})
        model = SONN(cfg, d_model=4)
        assert model.shared_proj is None
        assert model.soft_binner is None
        assert model.out_proj is None
        assert model.criterion_type == CriterionType.cmpValidate

    def test_binary_init(self):
        cfg = _make_cfg(model={"type": "binary", "num_classes": 2, "nbest_neurons": 3,
                                "soft_binner": False, "ref_functions": ["linear_cov"]})
        model = SONN(cfg, d_model=4)
        assert isinstance(model.loss_fn, nn.BCEWithLogitsLoss)

    def test_multiclass_soft_binner(self):
        cfg = _make_cfg(model={"type": "multi-class", "num_classes": 5, "nbest_neurons": 3,
                                "soft_binner": True, "ref_functions": ["linear_cov"]})
        model = SONN(cfg, d_model=4)
        assert model.soft_binner is not None
        assert model.shared_proj is None

    def test_multiclass_shared_proj_default(self):
        cfg = _make_cfg(model={"type": "multi-class", "num_classes": 4, "nbest_neurons": 3,
                                "soft_binner": False, "ref_functions": ["linear_cov"]})
        model = SONN(cfg, d_model=4)
        assert model.shared_proj is not None
        assert model.soft_binner is None

    def test_multiclass_neuron_proj(self):
        cfg = _make_cfg(model={"type": "multi-class", "num_classes": 4, "nbest_neurons": 3,
                                "soft_binner": False, "use_neuron_proj": True,
                                "ref_functions": ["linear_cov"]})
        model = SONN(cfg, d_model=4)
        assert model.shared_proj is None
        assert model.soft_binner is None

    def test_invalid_type_raises(self):
        cfg = _make_cfg(model={"type": "unknown-flavor", "num_classes": 3, "nbest_neurons": 3})
        with pytest.raises(ValueError):
            SONN(cfg, d_model=4)

    def test_regressor_with_output_projection(self):
        cfg = _make_cfg(
            model={
                "type": "regressor",
                "num_classes": 1,
                "nbest_neurons": 4,
                "soft_binner": False,
                "use_output_projection": True,
                "max_neuron_models": 6,
                "ref_functions": ["linear_cov"],
            }
        )
        model = SONN(cfg, d_model=4)
        assert isinstance(model.out_proj, nn.Linear)

    def test_feature_names_from_ndarray(self):
        import numpy as np
        cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                                "soft_binner": False, "ref_functions": ["linear_cov"]})
        names = np.array(["a", "b", "c"])
        model = SONN(cfg, d_model=3, feature_names=names)
        assert model.feature_names == ["a", "b", "c"]


class TestSONNProperties:
    def _model(self) -> SONN:
        cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                                "soft_binner": False, "ref_functions": ["linear_cov"]})
        return SONN(cfg, d_model=4)

    def test_str(self):
        assert "Self-organizing" in str(self._model())

    def test_device_fallback_without_layers(self):
        model = self._model()
        assert isinstance(model.device, torch.device)

    def test_retrain_required_false_by_default(self):
        assert self._model().retrain_required is False

    def test_need_bias_and_regularity_err_flags(self):
        m = self._model()
        assert m.need_regularity_err
        assert not m.need_bias_err


class TestCreateLayer:
    def _model(self, ref_functions: list, max_neuron_models: int = 6) -> SONN:
        cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                                "soft_binner": False, "max_neuron_models": max_neuron_models,
                                "ref_functions": ref_functions})
        return SONN(cfg, d_model=4)

    def test_first_layer_has_neurons(self):
        m = self._model(["linear"])
        layer = m.create_layer(0)
        assert len(layer.neuron_models) == 1
        assert isinstance(layer.neuron_models[0], LinearPolynomNeuron)

    def test_no_ref_functions_raises(self):
        m = self._model([])
        with pytest.raises(LayerCreationError) as excinfo:
            m.create_layer(0)
        assert excinfo.value.layer_index == 0

    def test_subsequent_layer_uses_prev(self):
        m = self._model(["linear_cov"])
        l0 = m.create_layer(0)
        l0.d_model = 5  # pretend the prev layer survived 5 neurons
        m.layers.append(l0)
        l1 = m.create_layer(1)
        assert len(l1.neuron_models) >= 1


class TestGetError:
    def _model(self, ct: str = "validate") -> SONN:
        cfg = _make_cfg(
            model={
                "type": "regressor",
                "num_classes": 1,
                "nbest_neurons": 3,
                "soft_binner": False,
                "ref_functions": ["linear_cov"],
            },
            train={"criterion_type": ct},
        )
        return SONN(cfg, d_model=4)

    def test_validate(self):
        m = self._model("validate")
        out = m.get_error(CriterionType.cmpValidate, torch.tensor([1.0]), None)
        assert out.item() == 1.0

    def test_bias(self):
        m = self._model("bias")
        out = m.get_error(CriterionType.cmpBias, None, torch.tensor([2.0]))
        assert out.item() == 2.0

    def test_combined(self):
        m = self._model("validate_bias")
        # alpha = 0.5 default → 0.5 * 1 + 0.5 * 3 = 2
        out = m.get_error(
            CriterionType.cmpComb_validate_bias,
            torch.tensor([1.0]),
            torch.tensor([3.0]),
        )
        assert out.item() == 2.0

    def test_bias_retrain(self):
        m = self._model("bias_retrain")
        out = m.get_error(
            CriterionType.cmpComb_bias_retrain, None, torch.tensor([7.0])
        )
        assert out.item() == 7.0


class TestForward:
    def _model(self, shortcut: bool = True, layer_norm: bool = False) -> SONN:
        cfg = _make_cfg(
            model={
                "type": "regressor",
                "num_classes": 1,
                "nbest_neurons": 3,
                "soft_binner": False,
                "ref_functions": ["linear_cov"],
                "shortcut": shortcut,
                "use_layer_norm": layer_norm,
            }
        )
        return SONN(cfg, d_model=4)

    def test_forward_no_layers_returns_input(self):
        m = self._model()
        x = torch.randn(3, 4)
        out = m(x)
        assert torch.allclose(out, x)

    def test_forward_with_layer(self):
        m = self._model(shortcut=False)
        layer = m.create_layer(0)
        m.layers.append(layer)
        x = torch.randn(2, 4)
        out = m(x)
        # 6 columns from LinearCov C(4,2)=6
        assert out.shape == (2, 6)

    def test_forward_skip_last_layer(self):
        m = self._model()
        # add a single layer; skip_last_layer means we run zero layers
        layer = m.create_layer(0)
        m.layers.append(layer)
        x = torch.randn(2, 4)
        out = m(x, skip_last_layer=True)
        assert torch.allclose(out, x)


def test_default_config_returns_fresh_instance():
    a = SONN.default_config()
    b = SONN.default_config()
    a.model.type = "regressor"
    assert b.model.type == "multi-class"


def test_set_class_weights_swap():
    cfg = _make_cfg(model={"type": "binary", "num_classes": 2, "nbest_neurons": 3,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)
    cw = torch.tensor([2.0])
    m.set_class_weights(cw)
    assert isinstance(m.loss_fn, nn.BCEWithLogitsLoss)
    assert torch.allclose(m.loss_fn.pos_weight, cw.to(m.dtype))


def test_set_class_weights_nll_swap():
    cfg = _make_cfg(model={"type": "multi-class", "num_classes": 3, "nbest_neurons": 3,
                            "soft_binner": True, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)
    cw = torch.tensor([1.0, 2.0, 3.0])
    m.set_class_weights(cw)
    assert isinstance(m.loss_fn, nn.NLLLoss)


def test_state_dict_records_layer_names():
    cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)
    m.layers.append(m.create_layer(0))
    sd = m.state_dict()
    assert m.layer_names == ["SONNLayer"]
    assert "params_metadata" in sd


def test_get_selected_and_unselected_features():
    cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4, feature_names=["a", "b", "c", "d"])
    layer = m.create_layer(0)
    m.layers.append(layer)
    sel = m.get_selected_features_indices()
    # Linear-cov first layer uses every original index in some pair
    assert set(sel) == {0, 1, 2, 3}
    assert m.get_unselected_features_indices() == []
    assert m.get_unselected_features() == "No unselected features"
    assert isinstance(m.get_selected_features(), str)


def test_compute_loss_multiclass_softbinner_vmap():
    """Exercise the soft-binner branch by using torch.func.vmap so pred is (batch,)."""
    from torch.func import vmap
    from functools import partial

    cfg = _make_cfg(model={"type": "multi-class", "num_classes": 4, "nbest_neurons": 3,
                            "soft_binner": True, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)
    neuron = LinearCovPolynomNeuron(4, 4, None, 0, 0)
    params = dict(neuron.named_parameters())
    buffers = {"src_idxs": neuron.src_idxs}

    fn = vmap(
        partial(SONN.compute_loss, neuron, None, "multi-class", m.soft_binner, 0.0),
        in_dims=({k: 0 for k in params}, {k: 0 for k in buffers}, None, None),
    )
    x = torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))
    out = fn(params, buffers, x, y)
    # (ensemble, batch, num_classes)
    assert out.shape[-1] == 4


def test_compute_loss_multiclass_shared_proj_vmap():
    from torch.func import vmap
    from functools import partial

    cfg = _make_cfg(model={"type": "multi-class", "num_classes": 3, "nbest_neurons": 3,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)
    neuron = LinearCovPolynomNeuron(4, 4, None, 0, 0)
    params = dict(neuron.named_parameters())
    params["shared_proj_weight"] = m.shared_proj.weight
    params["shared_proj_bias"] = m.shared_proj.bias
    buffers = {"src_idxs": neuron.src_idxs}

    in_dims_params = {k: 0 for k in params}
    in_dims_params["shared_proj_weight"] = None
    in_dims_params["shared_proj_bias"] = None
    in_dims_buffers = {k: 0 for k in buffers}
    fn = vmap(
        partial(SONN.compute_loss, neuron, None, "multi-class", None, 0.0),
        in_dims=(in_dims_params, in_dims_buffers, None, None),
    )
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    out = fn(params, buffers, x, y)
    assert out.shape[-1] == 3


def test_compute_loss_multiclass_neuron_proj_vmap():
    from torch.func import vmap
    from functools import partial

    cfg = _make_cfg(model={"type": "multi-class", "num_classes": 3, "nbest_neurons": 3,
                            "soft_binner": False, "use_neuron_proj": True,
                            "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)
    neuron = LinearCovPolynomNeuron(4, 4, None, 0, 0)
    neuron.set_proj(num_classes=3)
    params = dict(neuron.named_parameters())
    buffers = {"src_idxs": neuron.src_idxs}

    fn = vmap(
        partial(SONN.compute_loss, neuron, None, "multi-class", None, 0.0),
        in_dims=({k: 0 for k in params}, {k: 0 for k in buffers}, None, None),
    )
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    out = fn(params, buffers, x, y)
    assert out.shape[-1] == 3


def test_compute_loss_ridge_penalty_vmap():
    """Exercise the ridge_alpha > 0 branch under vmap."""
    from torch.func import vmap
    from functools import partial

    cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)
    neuron = LinearCovPolynomNeuron(4, 4, None, 0, 0)
    params = dict(neuron.named_parameters())
    buffers = {"src_idxs": neuron.src_idxs}
    x = torch.randn(8, 4)
    y = torch.randn(8)

    fn0 = vmap(
        partial(SONN.compute_loss, neuron, m.loss_fn, "regressor", None, 0.0),
        in_dims=({k: 0 for k in params}, {k: 0 for k in buffers}, None, None),
    )
    fn1 = vmap(
        partial(SONN.compute_loss, neuron, m.loss_fn, "regressor", None, 0.5),
        in_dims=({k: 0 for k in params}, {k: 0 for k in buffers}, None, None),
    )
    base = fn0(params, buffers, x, y)
    ridged = fn1(params, buffers, x, y)
    assert (ridged >= base).all()


def test_compute_loss_returns_pred_when_loss_fn_none():
    cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)
    neuron = LinearCovPolynomNeuron(4, 4, None, 0, 0)
    params = {k: v.detach() for k, v in neuron.named_parameters()}
    buffers = {}
    x = torch.randn(8, 4)
    y = torch.randn(8)
    # loss_fn=None hits the early-return pred path (regression branch)
    out = SONN.compute_loss(neuron, None, "regressor", None, 0.0, params, buffers, x, y)
    # neuron with full ensemble: pred shape (8, 6)
    assert out.shape == (8, 6)




def test_get_best_neuron_model_and_columns():
    cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 4,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)
    layer = m.create_layer(0)
    m.layers.append(layer)
    layer.err_values = torch.tensor([3.0, 1.0, 2.0])
    layer.module_idxs = torch.tensor([[0, 0], [0, 1], [0, 2]])
    bm, bn = m.get_best_neuron_model(layer)
    assert int(bm) == 0
    # smallest is at idx 1 → neuron_idx == 1
    assert int(bn) == 1
    col = m._best_neuron_column(layer)
    assert col == 1
    cols = m._best_neuron_columns(layer, k=2)
    assert len(cols) == 2


def test_infer_regression():
    cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)
    m.layers.append(m.create_layer(0))
    x = torch.randn(5, 4)
    out = m.infer(x)
    assert out.shape == (5, 6)


def test_get_selected_features_without_feature_names():
    cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)  # no feature_names
    m.layers.append(m.create_layer(0))
    out = m.get_selected_features()
    # without feature_names we get the "index=inp_X" form
    assert "index=inp_" in out


def test_get_selected_features_with_shortcut_layers():
    cfg = _make_cfg(
        model={
            "type": "regressor",
            "num_classes": 1,
            "nbest_neurons": 3,
            "soft_binner": False,
            "ref_functions": ["linear_cov"],
            "shortcut": True,
        }
    )
    m = SONN(cfg, d_model=4, feature_names=["a", "b", "c", "d"])
    l0 = m.create_layer(0)
    m.layers.append(l0)
    # Manually add a second layer with a manageable d_model
    l1 = m.create_layer(1)
    m.layers.append(l1)
    sel = m.get_selected_features_indices()
    assert all(0 <= s < 4 for s in sel)


def test_get_unselected_features_when_some_unselected():
    # Force a model where the first layer is so small no feature combination
    # uses index 3. Easier: directly tamper with src_idxs on a freshly-built model.
    cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=5, feature_names=["a", "b", "c", "d", "e"])
    l0 = m.create_layer(0)
    # Replace src_idxs to only reference {0,1} so {2,3,4} are unselected.
    l0.neuron_models[0].src_idxs = torch.tensor([[0, 1]])
    m.layers.append(l0)
    unsel = m.get_unselected_features_indices()
    assert set(unsel) == {2, 3, 4}
    s = m.get_unselected_features()
    assert "a" not in s.split(",")  # only unselected names appear


def test_plot_layer_error_does_not_raise(monkeypatch):
    # plt.show would block in non-interactive environments; replace with a no-op.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)

    cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)
    m.layers.append(m.create_layer(0))
    m.layer_err = [1.0]
    m.plot_layer_error()


def test_create_layer_with_all_ref_functions():
    cfg = _make_cfg(
        model={
            "type": "regressor",
            "num_classes": 1,
            "nbest_neurons": 3,
            "soft_binner": False,
            "max_neuron_models": 4,
            "ref_functions": [
                "linear",
                "linear_cov",
                "quadratic",
                "cubic",
                {"polyquad": {"dim": 2}},
            ],
        }
    )
    m = SONN(cfg, d_model=5)
    layer = m.create_layer(0)
    # five distinct neuron types
    assert len(layer.neuron_models) == 5


def test_create_layer_neuron_proj_sets_projection():
    cfg = _make_cfg(
        model={
            "type": "multi-class",
            "num_classes": 4,
            "nbest_neurons": 3,
            "soft_binner": False,
            "use_neuron_proj": True,
            "ref_functions": ["linear_cov"],
        }
    )
    m = SONN(cfg, d_model=4)
    layer = m.create_layer(0)
    nm = layer.neuron_models[0]
    assert nm.proj_weight is not None
    assert nm.proj_bias is not None


def test_infer_with_out_proj_regression():
    cfg = _make_cfg(
        model={
            "type": "regressor",
            "num_classes": 1,
            "nbest_neurons": 4,
            "soft_binner": False,
            "use_output_projection": True,
            "num_out_neurons": 2,
            "ref_functions": ["linear_cov"],
        }
    )
    m = SONN(cfg, d_model=4)
    layer = m.create_layer(0)
    m.layers.append(layer)
    layer.err_values = torch.arange(layer.neuron_models[0].num_neurons, dtype=torch.float)
    layer.module_idxs = torch.stack(
        [
            torch.zeros(layer.neuron_models[0].num_neurons, dtype=torch.long),
            torch.arange(layer.neuron_models[0].num_neurons, dtype=torch.long),
        ],
        dim=1,
    )
    x = torch.randn(3, 4)
    out = m.infer(x)
    # Regression with out_proj should produce shape (N,) after squeeze
    assert out.shape == (3,)


def test_infer_with_out_proj_multiclass():
    cfg = _make_cfg(
        model={
            "type": "multi-class",
            "num_classes": 3,
            "nbest_neurons": 4,
            "soft_binner": True,
            "use_output_projection": True,
            "num_out_neurons": 2,
            "ref_functions": ["linear_cov"],
        }
    )
    m = SONN(cfg, d_model=4)
    layer = m.create_layer(0)
    m.layers.append(layer)
    layer.err_values = torch.arange(layer.neuron_models[0].num_neurons, dtype=torch.float)
    layer.module_idxs = torch.stack(
        [
            torch.zeros(layer.neuron_models[0].num_neurons, dtype=torch.long),
            torch.arange(layer.neuron_models[0].num_neurons, dtype=torch.long),
        ],
        dim=1,
    )
    x = torch.randn(3, 4)
    out = m.infer(x)
    assert out.shape == (3, 3)


def test_infer_multiclass_softbinner_uses_best_neuron():
    cfg = _make_cfg(
        model={
            "type": "multi-class",
            "num_classes": 4,
            "nbest_neurons": 3,
            "soft_binner": True,
            "ref_functions": ["linear_cov"],
        }
    )
    m = SONN(cfg, d_model=4)
    layer = m.create_layer(0)
    m.layers.append(layer)
    layer.err_values = torch.arange(layer.neuron_models[0].num_neurons, dtype=torch.float)
    layer.module_idxs = torch.stack(
        [
            torch.zeros(layer.neuron_models[0].num_neurons, dtype=torch.long),
            torch.arange(layer.neuron_models[0].num_neurons, dtype=torch.long),
        ],
        dim=1,
    )
    out = m.infer(torch.randn(3, 4))
    assert out.shape == (3, 4)


def test_infer_multiclass_neuron_proj():
    cfg = _make_cfg(
        model={
            "type": "multi-class",
            "num_classes": 3,
            "nbest_neurons": 3,
            "soft_binner": False,
            "use_neuron_proj": True,
            "ref_functions": ["linear_cov"],
        }
    )
    m = SONN(cfg, d_model=4)
    layer = m.create_layer(0)
    m.layers.append(layer)
    out = m.infer(torch.randn(3, 4))
    assert out.shape == (3, 3)


def test_forward_with_shortcut_and_layer_norm():
    cfg = _make_cfg(
        model={
            "type": "regressor",
            "num_classes": 1,
            "nbest_neurons": 3,
            "soft_binner": False,
            "ref_functions": ["linear_cov"],
            "shortcut": True,
            "use_layer_norm": True,
        }
    )
    m = SONN(cfg, d_model=4)
    l0 = m.create_layer(0)
    l1 = m.create_layer(1)
    m.layers.extend([l0, l1])
    # Configure LayerNorm widths post-creation: l0 outputs C(4,2)=6 + 4 shortcut = 10
    l0.setup_layer_norm(10)
    # l1's input is l0.d_model + d_model when shortcut on
    l1.setup_layer_norm(l1.d_model + 4)
    x = torch.randn(2, 4)
    out = m(x)
    assert out.shape[0] == 2


def test_forward_with_preprocessing_module():
    cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    pre = nn.Identity()
    m = SONN(cfg, d_model=4, preprocessing=pre)
    x = torch.randn(3, 4)
    out = m(x)
    assert out.shape == x.shape


def test_restore_from_checkpoint_metadata_roundtrip():
    cfg = _make_cfg(model={"type": "regressor", "num_classes": 1, "nbest_neurons": 3,
                            "soft_binner": False, "ref_functions": ["linear_cov"]})
    m = SONN(cfg, d_model=4)
    m.layers.append(m.create_layer(0))
    sd = m.state_dict()

    m2 = SONN(cfg, d_model=4)
    m2.restore_from_checkpoint_metadata(sd)
    # one layer should be reconstructed
    assert len(m2.layers) == 1
    assert isinstance(m2.layers[0], SONNLayer)
