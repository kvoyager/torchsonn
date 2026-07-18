import pytest
import torch

from torchsonn.layer import NeuronModuleList, SONNLayer
from torchsonn.neurons import LinearCovPolynomNeuron, LinearPolynomNeuron


def _make_layer_with_two_neurons(layer_index: int = 0, use_layer_norm: bool = False) -> SONNLayer:
    layer = SONNLayer(d_model=4, nbest_neurons=2, layer_index=layer_index, use_layer_norm=use_layer_norm)
    layer.neuron_models.append(LinearPolynomNeuron(4, 4, None, layer_index, 0))
    layer.neuron_models.append(LinearCovPolynomNeuron(4, 4, None, layer_index, 6))
    return layer


class TestNeuronModuleList:
    def test_iteration_and_indexing(self):
        ml = NeuronModuleList()
        a = LinearPolynomNeuron(3, 3, None, 0, 0)
        b = LinearCovPolynomNeuron(3, 3, None, 0, 0)
        ml.append(a)
        ml.append(b)
        # iteration yields the same items
        out = list(ml)
        assert out[0] is a and out[1] is b
        # __getitem__ int returns the typed item
        assert ml[0] is a
        # __getitem__ slice returns NeuronModuleList
        sub = ml[0:1]
        assert isinstance(sub, NeuronModuleList)
        assert len(sub) == 1


class TestSONNLayer:
    def test_len_repr(self):
        layer = _make_layer_with_two_neurons()
        # Each neuron has C(4,2)=6 → total 12
        assert len(layer) == 12
        assert "Layer 0" in repr(layer)

    def test_indexing(self):
        layer = _make_layer_with_two_neurons()
        assert isinstance(layer[0], LinearPolynomNeuron)
        assert isinstance(layer[1], LinearCovPolynomNeuron)

    def test_forward_concatenates(self):
        layer = _make_layer_with_two_neurons()
        x = torch.randn(5, 4)
        out = layer(x)
        # 6 from each → 12 columns
        assert out.shape == (5, 12)

    def test_setup_layer_norm_idempotent(self):
        layer = _make_layer_with_two_neurons(use_layer_norm=True)
        layer.setup_layer_norm(12)
        first = layer.layer_norm
        layer.setup_layer_norm(20)  # should be a no-op
        assert layer.layer_norm is first
        assert layer.layer_norm_dim == 12

    def test_setup_layer_norm_disabled(self):
        layer = _make_layer_with_two_neurons(use_layer_norm=False)
        layer.setup_layer_norm(12)
        assert layer.layer_norm is None

    def test_state_dict_populates_names(self):
        layer = _make_layer_with_two_neurons()
        sd = layer.state_dict()
        assert layer.neuron_models_names == [
            "LinearPolynomNeuron",
            "LinearCovPolynomNeuron",
        ]
        assert any(k.endswith("params_metadata") for k in sd.keys())

    def test_from_checkpoint_metadata_without_layer_norm(self):
        meta = {
            "d_model": 8,
            "nbest_neurons": 3,
            "layer_index": 2,
            "use_layer_norm": False,
        }
        layer = SONNLayer.from_checkpoint_metadata(meta)
        assert layer.layer_index == 2
        assert layer.layer_norm is None

    def test_from_checkpoint_metadata_with_layer_norm_dim(self):
        meta = {
            "d_model": 8,
            "nbest_neurons": 3,
            "layer_index": 2,
            "use_layer_norm": True,
            "layer_norm_dim": 12,
        }
        layer = SONNLayer.from_checkpoint_metadata(meta)
        assert layer.layer_norm is not None
        assert layer.layer_norm_dim == 12

    def test_from_checkpoint_metadata_missing_layer_norm_dim_falls_back(self):
        meta = {
            "d_model": 8,
            "nbest_neurons": 3,
            "layer_index": 2,
            "use_layer_norm": True,
        }
        # missing layer_norm_dim → falls back to d_model
        layer = SONNLayer.from_checkpoint_metadata(meta)
        assert layer.layer_norm_dim == 8

    def test_get_parent_neron_module_found(self):
        layer = _make_layer_with_two_neurons()
        # absolute idx 0 → first neuron (idx 0 inside first module)
        idx, parent = layer.get_parent_neron_module(0)
        assert idx == 0
        assert parent is layer.neuron_models[0]
        # absolute idx 7 → second module (offset by num_neurons of first)
        idx, parent = layer.get_parent_neron_module(7)
        assert parent is layer.neuron_models[1]

    def test_get_parent_neron_module_invalid_raises(self):
        layer = _make_layer_with_two_neurons()
        with pytest.raises(ValueError):
            layer.get_parent_neron_module(9999)

    def test_set_neuron_module_populates_idxs(self):
        layer = _make_layer_with_two_neurons()
        layer.set_neuron_module()
        # 12 neurons → neuron_idxs shape (12, 2)
        assert layer.neuron_idxs.shape == (12, 2)
        assert layer.d_model == 12

    def test_to_preserves_buffer_dtype(self):
        layer = _make_layer_with_two_neurons()
        layer.err_values = torch.tensor([1.0, 2.0])
        layer.module_idxs = torch.tensor([[0, 0], [0, 1]])
        layer.neuron_idxs = torch.tensor([[0, 0], [0, 1]])
        moved = layer.to(dtype=torch.float64)
        # err_values follows the dtype cast
        assert moved.err_values.dtype == torch.float64
