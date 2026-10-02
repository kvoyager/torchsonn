"""`model.shortcut`: raw features and the outputs of older layers as layer inputs.

The input of layer j >= 1 is [h_{j-1} | h_{j-2} | ... | h_{j-1-k} | x_raw]
with k = shortcut.prev_layers (null = 0) and x_raw present when
shortcut.raw_features. k = 0 is the original [h_{j-1} | x_raw] layout, which
the reference forward below reproduces from the code before the option.
"""
import sys

import pytest
import torch
from omegaconf import OmegaConf

from torchsonn.config import SONNConfig
from torchsonn.model import SONN, _parse_shortcut
from torchsonn.trainer import Trainer

from tests.test_trainer_smoke import _cfg, _make_dl

D_MODEL = 3
WIDTH = 3  # outputs kept per layer by _grow


def _model(shortcut, use_layer_norm: bool = False) -> SONN:
    base = OmegaConf.structured(SONNConfig)
    model_cfg = {
        "type": "regressor",
        "num_classes": 1,
        "nbest_neurons": 2,
        "soft_binner": False,
        "ref_functions": ["linear_cov"],
        "use_layer_norm": use_layer_norm,
    }
    if shortcut is not None:
        model_cfg["shortcut"] = shortcut
    return SONN(OmegaConf.merge(base, OmegaConf.create({"model": model_cfg})), d_model=D_MODEL)


def _grow(m: SONN, n_layers: int, width: int = WIDTH) -> SONN:
    """Append `n_layers` layers the way the trainer does: create at the next
    position, keep `width` neurons (selection), size the LayerNorm."""
    torch.manual_seed(0)
    for _ in range(n_layers):
        layer = m.create_layer(len(m.layers))
        m.layers.append(layer)
        # linear_cov enumerates every input pair: its widest index is the
        # input width - 1, recorded before selection drops pairs
        layer.test_input_width = int(layer.neuron_models[0].src_idxs.max()) + 1
        layer.neuron_models = layer.neuron_models[:1]
        layer.neuron_models[0].prune(torch.arange(width))
        layer.d_model = len(layer)
        layer.setup_layer_norm(m.next_input_width())
    return m


def _reference_forward(m: SONN, x: torch.Tensor, shortcut: bool, skip_last_layer: bool = False) -> torch.Tensor:
    """SONN.forward as it was before `shortcut` became a mapping."""
    x_inp = x
    layers = m.layers[:-1] if skip_last_layer else m.layers
    clamp = m.param.model.output_clamp_value
    for idx, layer in enumerate(layers):
        x = torch.clamp(layer(x), -clamp, clamp)
        apply_shortcut = shortcut and (skip_last_layer or idx < len(m.layers) - 1)
        if apply_shortcut:
            x = torch.cat([x, x_inp], dim=-1)
        if layer.layer_norm is not None and (apply_shortcut or not shortcut):
            x = layer.layer_norm(x)
    return x


class TestParseShortcut:
    def test_bool_shorthand(self):
        assert _parse_shortcut(True) == (True, 0)
        assert _parse_shortcut(False) == (False, 0)

    def test_mapping_defaults(self):
        assert _parse_shortcut({}) == (True, 0)
        assert _parse_shortcut({"raw_features": False}) == (False, 0)
        assert _parse_shortcut({"prev_layers": None}) == (True, 0)

    def test_prev_layers(self):
        assert _parse_shortcut({"prev_layers": 0}) == (True, 0)
        assert _parse_shortcut({"prev_layers": 2}) == (True, 2)
        assert _parse_shortcut({"prev_layers": "all"}) == (True, sys.maxsize)

    @pytest.mark.parametrize("value", [
        {"prev_layers": -1},
        {"prev_layers": "some"},
        {"prev_layers": 1.5},
        {"prev_layers": True},
        {"raw_features": "yes"},
        {"raw_feature": True},
        "true",
        1,
    ])
    def test_invalid(self, value):
        with pytest.raises(ValueError):
            _parse_shortcut(value)

    def test_config_forms(self):
        assert (_model(None).shortcut_raw, _model(None).shortcut_prev) == (True, 0)
        assert (_model(False).shortcut_raw, _model(False).shortcut_prev) == (False, 0)
        m = _model({"prev_layers": "all"})
        assert (m.shortcut_raw, m.shortcut_prev) == (True, sys.maxsize)
        m = _model({"raw_features": False, "prev_layers": 2})
        assert (m.shortcut_raw, m.shortcut_prev) == (False, 2)

    def test_dotlist_override(self):
        base = OmegaConf.structured(SONNConfig)
        cfg = OmegaConf.merge(
            base,
            OmegaConf.create({"model": {"type": "regressor", "num_classes": 1, "nbest_neurons": 2,
                                        "soft_binner": False}}),
            OmegaConf.from_dotlist(["model.shortcut.prev_layers=all"]),
        )
        m = SONN(cfg, d_model=D_MODEL)
        assert (m.shortcut_raw, m.shortcut_prev) == (True, sys.maxsize)


class TestUnchangedWithoutPrevLayers:
    """shortcut: true|false, an absent shortcut and prev_layers: null / 0 keep
    the original layout and forward bit for bit."""

    @pytest.mark.parametrize("shortcut, legacy", [
        (True, True), (False, False), (None, True),
        ({"prev_layers": None}, True), ({"prev_layers": 0}, True),
        ({"raw_features": False, "prev_layers": 0}, False),
    ])
    @pytest.mark.parametrize("use_layer_norm", [False, True])
    def test_forward_matches_reference(self, shortcut, legacy, use_layer_norm):
        m = _grow(_model(shortcut, use_layer_norm), 4)
        x = torch.randn(5, D_MODEL)
        for j in range(1, 4):
            assert m.layers[j].input_layers == [j - 1]
            assert m.layers[j].input_raw == legacy
        assert torch.equal(m(x), _reference_forward(m, x, legacy))
        assert torch.equal(m(x, skip_last_layer=True), _reference_forward(m, x, legacy, skip_last_layer=True))

    @pytest.mark.parametrize("shortcut, width", [(True, WIDTH + D_MODEL), (False, WIDTH)])
    def test_widths(self, shortcut, width):
        m = _grow(_model(shortcut), 3)
        assert m.next_input_width() == width
        if m.layers[0].layer_norm is not None:
            assert m.layers[0].layer_norm_dim == width


class TestPrevLayers:
    @pytest.mark.parametrize("k, expected", [
        (1, [[], [0], [1, 0], [2, 1], [3, 2]]),
        (2, [[], [0], [1, 0], [2, 1, 0], [3, 2, 1]]),
        ("all", [[], [0], [1, 0], [2, 1, 0], [3, 2, 1, 0]]),
    ])
    @pytest.mark.parametrize("raw", [True, False])
    def test_layout_and_width(self, k, expected, raw):
        m = _grow(_model({"raw_features": raw, "prev_layers": k}), 5)
        for j, layer in enumerate(m.layers):
            assert layer.input_layers == expected[j]
            assert m.layer_sources(j) == (expected[j], True if j == 0 else raw)
            width = WIDTH * len(expected[j]) + (D_MODEL if raw or j == 0 else 0)
            assert sum(w for _, w in m.input_blocks(j)) == width
            assert layer.test_input_width == width
        # a sixth layer would read layer 4 plus k older ones
        n_next = {1: 2, 2: 3, "all": 5}[k]
        assert m.next_input_width() == WIDTH * n_next + (D_MODEL if raw else 0)

    def test_locate(self):
        m = _grow(_model({"prev_layers": 1}), 3)
        # layer 2 reads [h_1 (3) | h_0 (3) | raw (3)]
        assert [m.locate(2, i) for i in range(9)] == [
            (1, 0), (1, 1), (1, 2), (0, 0), (0, 1), (0, 2), (None, 0), (None, 1), (None, 2)]
        with pytest.raises(IndexError):
            m.locate(2, 9)

    @pytest.mark.parametrize("use_layer_norm", [False, True])
    def test_forward_composes_blocks(self, use_layer_norm):
        m = _grow(_model({"prev_layers": "all"}, use_layer_norm), 4)
        clamp = m.param.model.output_clamp_value
        x = torch.randn(5, D_MODEL)
        h = [torch.clamp(m.layers[0](x), -clamp, clamp)]
        for j in range(1, 4):
            inp = torch.cat([h[s] for s in reversed(range(j))] + [x], dim=-1)
            if use_layer_norm:
                assert m.layers[j - 1].layer_norm_dim == inp.shape[-1]
                inp = m.layers[j - 1].layer_norm(inp)
            if j == 3:
                assert torch.equal(m(x, skip_last_layer=True), inp)
            h.append(torch.clamp(m.layers[j](inp), -clamp, clamp))
        # the inference-time last layer's output is not normalized: a next
        # layer's input would include more than it
        assert torch.equal(m(x), h[3])

    def test_layer_norm_without_raw_features(self):
        m = _grow(_model({"raw_features": False, "prev_layers": 1}, use_layer_norm=True), 3)
        # sized for a next layer's [h_2 | h_1]; not applied to the last output
        assert m.concat_inputs
        assert m.layers[-1].layer_norm_dim == 2 * WIDTH
        x = torch.randn(4, D_MODEL)
        assert m(x).shape == (4, WIDTH)

    def test_selected_features_read_raw_blocks(self):
        m = _grow(_model({"prev_layers": 1}), 3)
        m.layers[0].neuron_models[0].src_idxs = torch.tensor([[0, 1], [0, 1]])
        # layer 2: [h_1 (0-2) | h_0 (3-5) | raw (6-8)] -> raw feature 2 at 8
        m.layers[1].neuron_models[0].src_idxs = torch.tensor([[0, 1], [1, 2]])
        m.layers[2].neuron_models[0].src_idxs = torch.tensor([[0, 3], [4, 8]])
        assert sorted(m.get_selected_features_indices()) == [0, 1, 2]


class TestCheckpoint:
    def test_layout_round_trips(self):
        m = _grow(_model({"prev_layers": 1}), 3)
        sd = m.state_dict()
        m2 = _model({"prev_layers": 1})
        m2.restore_from_checkpoint_metadata(sd)
        m2.load_state_dict(sd, strict=False)
        assert [layer.input_layers for layer in m2.layers] == [[], [0], [1, 0]]
        assert [layer.input_raw for layer in m2.layers] == [True, True, True]
        x = torch.randn(4, D_MODEL)
        assert torch.equal(m2(x), m(x))

    @pytest.mark.parametrize("shortcut", [True, False])
    def test_legacy_checkpoint_falls_back_to_previous_layer(self, shortcut):
        m = _grow(_model(shortcut), 3)
        x = torch.randn(4, D_MODEL)
        expected = m(x)
        for layer in m.layers:
            layer.input_layers = None
            layer.input_raw = None
        assert m.layer_sources(0) == ([], True)
        assert m.layer_sources(2) == ([1], shortcut)
        assert torch.equal(m(x), expected)


def test_prune_refuses_older_layer_inputs(tmp_path):
    cfg = _cfg(tmp_path)
    m = _grow(_model({"prev_layers": 1}), 3)
    with pytest.raises(NotImplementedError):
        Trainer(config=cfg).prune(m)


def test_train_with_prev_layers(tmp_path):
    cfg = _cfg(tmp_path, max_layer_count=3, criterion_minimum_width=3)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"model": {"shortcut": {"prev_layers": "all"}}}))
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    trained = trainer.train(model, _make_dl(48), _make_dl(16), _make_dl(8), verbose=False)
    for j, layer in enumerate(trained.layers):
        assert layer.input_layers == list(range(j - 1, -1, -1))
    preds, targets = trainer.infer(trained, _make_dl(8), verbose=False)
    assert preds.shape[0] == targets.shape[0]
