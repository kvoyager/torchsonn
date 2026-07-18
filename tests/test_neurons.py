import logging

import pytest
import torch
from torch import nn

from torchsonn.neurons import (
    BasePolynomNeuron,
    CubicPolynomNeuron,
    LinearCovPolynomNeuron,
    LinearPolynomNeuron,
    PolyQuadratic,
    QuadraticPolynomNeuron,
    generate_unique_combinations,
    generate_unique_pairs,
)
from torchsonn.neurons.base import _ACTIVATIONS, _max_unique_tuples
from torchsonn.types import CriterionType


class TestMaxUniqueTuples:
    @pytest.mark.parametrize(
        "n,k,allow_self,ordered,expected",
        [
            (0, 2, False, True, 0),
            (3, 0, False, True, 0),
            (4, 2, False, False, 6),     # C(4,2)
            (4, 2, False, True, 12),     # P(4,2)
            (3, 2, True, True, 9),       # 3^2
            (3, 2, True, False, 6),      # C(3+2-1,2)
            (2, 3, False, True, 0),      # k > n with replacement off
        ],
    )
    def test_table(self, n, k, allow_self, ordered, expected):
        assert _max_unique_tuples(n, k, allow_self, ordered) == expected


class TestGenerateUniquePairs:
    def test_default_no_replacement_unordered(self):
        pairs = generate_unique_pairs(5, 5, seed=42, ordered=False)
        assert len(pairs) == 5
        # No self-pairs, unordered (i <= j)
        for a, b in pairs:
            assert a != b
            assert a <= b
        assert len(set(pairs)) == 5

    def test_ordered_keeps_direction(self):
        pairs = generate_unique_pairs(4, 6, seed=1, ordered=True)
        # Some pair has order distinct from its reverse: only the ordered case
        # can produce both (a,b) and (b,a) — make sure that's even possible.
        seen = set(pairs)
        assert all(a != b for a, b in seen)

    def test_allow_self_returns_self_pairs(self):
        pairs = generate_unique_pairs(3, 9, seed=0, allow_self=True, ordered=True)
        assert any(a == b for a, b in pairs)

    def test_clamp_when_over_cap(self, caplog):
        caplog.set_level(logging.WARNING)
        pairs = generate_unique_pairs(3, 999, seed=0, ordered=False)
        assert len(pairs) == 3  # C(3,2) = 3
        assert any("clamping" in rec.message for rec in caplog.records)

    def test_zero_requests_returns_empty(self):
        assert generate_unique_pairs(5, 0, seed=0) == []


class TestGenerateUniqueCombinations:
    def test_triplets(self):
        out = generate_unique_combinations(5, 3, 4, seed=1, ordered=False)
        assert len(out) == 4
        for t in out:
            assert len(t) == 3
            assert list(t) == sorted(t)
            assert len(set(t)) == 3  # no replacement

    def test_clamp(self, caplog):
        caplog.set_level(logging.WARNING)
        out = generate_unique_combinations(4, 2, 1000, seed=0, ordered=False)
        assert len(out) == 6
        assert any("clamping" in rec.message for rec in caplog.records)

    def test_zero_request_empty(self):
        assert generate_unique_combinations(5, 2, 0) == []

    def test_allow_self(self):
        out = generate_unique_combinations(2, 3, 4, seed=0, allow_self=True, ordered=True)
        assert all(len(t) == 3 for t in out)
        # 2^3 = 8 distinct tuples, so 4 must succeed.
        assert len(out) == 4


class TestBinaryNeurons:
    @pytest.mark.parametrize(
        "cls,num_w,short",
        [
            (LinearPolynomNeuron, 3, "Linear"),
            (LinearCovPolynomNeuron, 4, "LinearCov"),
            (QuadraticPolynomNeuron, 6, "Quadratic"),
            (CubicPolynomNeuron, 8, "Cubic"),
        ],
    )
    def test_basic_forward_and_metadata(self, cls, num_w, short):
        neuron = cls(
            num_feat=4,
            num_src_feat=4,
            activation=None,
            layer_index=0,
            start_index=0,
        )
        assert cls.num_w == num_w
        x = torch.randn(7, 4)
        out = neuron(x)
        # First layer: enumerated pairs n*(n-1)/2 = 6
        assert out.shape == (7, 6)
        assert neuron.get_short_name() == short
        assert isinstance(neuron.get_name(), str)
        # num_neurons reflects weight rows
        assert neuron.num_neurons == 6
        assert neuron.ensemble_size == 6
        # device property mirrors weight
        assert neuron.device == neuron.weight.device

    def test_max_neuron_models_uses_random_subset(self):
        n = LinearPolynomNeuron(
            num_feat=6,
            num_src_feat=6,
            activation="relu",
            layer_index=0,
            start_index=0,
            max_neuron_models=4,
        )
        assert n.weight.shape[0] == 4
        assert n.src_idxs.shape == (4, 2)

    def test_init_uniform(self):
        n = LinearPolynomNeuron(
            num_feat=3,
            num_src_feat=3,
            activation=None,
            layer_index=0,
            start_index=0,
            init_method="uniform",
        )
        assert n.weight.abs().max() <= 0.1

    def test_init_unknown_raises(self):
        with pytest.raises(NotImplementedError):
            LinearPolynomNeuron(
                num_feat=3,
                num_src_feat=3,
                activation=None,
                layer_index=0,
                start_index=0,
                init_method="not-a-method",
            )

    def test_activation_string_lookup(self):
        n = LinearPolynomNeuron(3, 3, "tanh", 0, 0)
        assert isinstance(n.activation, nn.Tanh)

    def test_activation_unknown_string_raises(self):
        with pytest.raises(ValueError):
            LinearPolynomNeuron(3, 3, "swiglu", 0, 0)

    def test_activation_callable(self):
        f = lambda t: t * 2
        n = LinearPolynomNeuron(3, 3, f, 0, 0)
        assert n.activation is f

    def test_activation_invalid_type_raises(self):
        with pytest.raises(ValueError):
            LinearPolynomNeuron(3, 3, 42, 0, 0)  # int is not allowed

    def test_empty_activation_becomes_identity(self):
        n = LinearPolynomNeuron(3, 3, "", 0, 0)
        assert isinstance(n.activation, nn.Identity)

    def test_set_proj_attaches_params(self):
        n = LinearPolynomNeuron(3, 3, None, 0, 0)
        n.set_proj(num_classes=4)
        assert n.proj_weight.shape == (n.weight.shape[0], 4)
        assert torch.allclose(n.proj_bias, torch.zeros_like(n.proj_bias))
        # Calling twice still bookkeeps cleanly
        n.set_proj(num_classes=2)
        assert n.proj_weight.shape == (n.weight.shape[0], 2)
        assert "proj_num_classes" in n.params_metadata_names

    def test_prune_indices(self):
        n = LinearPolynomNeuron(4, 4, None, 0, 0)
        original_rows = n.weight.shape[0]
        idxs = torch.tensor([0, original_rows - 1])
        n.set_proj(num_classes=3)
        n.prune(idxs)
        assert n.weight.shape[0] == 2
        assert n.src_idxs.shape[0] == 2
        assert n.proj_weight.shape[0] == 2
        assert n.created_neuron_idxs.shape[0] == 2

    def test_need_bias_tools(self):
        n = LinearPolynomNeuron(3, 3, None, 0, 0)
        assert not n.need_bias_tools(CriterionType.cmpValidate)
        assert n.need_bias_tools(CriterionType.cmpBias)

    def test_to_moves_buffers(self):
        n = LinearPolynomNeuron(3, 3, None, 0, 0)
        moved = n.to(dtype=torch.float64)
        assert moved is n
        assert moved.weight.dtype == torch.float64
        # src_idxs is integer — `.to(dtype=float64)` would convert it too,
        # so use a device-only call to keep dtype invariant.

    def test_from_checkpoint_metadata_registry(self):
        n = LinearCovPolynomNeuron(3, 3, None, 0, 0)
        meta = {
            "cls": "LinearCovPolynomNeuron",
            "num_feat": 3,
            "num_src_feat": 3,
            "activation": None,
            "layer_index": 0,
            "start_index": 0,
            "dim": 2,
            "src_idxs": n.src_idxs,
        }
        restored = BasePolynomNeuron.from_checkpoint_metadata(meta)
        assert isinstance(restored, LinearCovPolynomNeuron)
        assert restored.weight.shape == n.weight.shape

    def test_from_checkpoint_metadata_with_projection(self):
        n = LinearPolynomNeuron(3, 3, None, 0, 0)
        n.set_proj(num_classes=2)
        meta = {
            "cls": "LinearPolynomNeuron",
            "num_feat": 3,
            "num_src_feat": 3,
            "activation": None,
            "layer_index": 0,
            "start_index": 0,
            "dim": 2,
            "src_idxs": n.src_idxs,
            "proj_num_classes": 2,
        }
        restored = BasePolynomNeuron.from_checkpoint_metadata(meta)
        assert restored.proj_weight.shape[1] == 2

class TestPolyQuadratic:
    def test_dim_must_be_at_least_two(self):
        with pytest.raises(AssertionError):
            PolyQuadratic(4, 4, None, 0, 0, dim=1)

    def test_forward_dim2(self):
        pq = PolyQuadratic(5, 5, None, 0, 0, dim=2)
        x = torch.randn(8, 5)
        out = pq(x)
        # n*(n-1)/2 = 10
        assert out.shape == (8, 10)
        assert "poly2" == pq.get_short_name()
        assert "full" in pq.get_name()

    def test_forward_dim3_with_max_models(self):
        pq = PolyQuadratic(6, 6, "tanh", 0, 0, dim=3, max_neuron_models=4)
        x = torch.randn(3, 6)
        out = pq(x)
        assert out.shape == (3, 4)
        assert pq.src_idxs.shape == (4, 3)

    def test_squares_excluded(self):
        pq = PolyQuadratic(4, 4, None, 0, 0, dim=2, squares=False)
        # num_w: 1 + dim + dim*(dim+1)/2 - dim
        # = 1 + 2 + 3 - 2 = 4
        assert pq.num_w == 4
        assert "covariance only" in pq.get_name()

    def test_create_src_idxs_dim_not_2_without_max_raises(self):
        with pytest.raises(NotImplementedError):
            PolyQuadratic(4, 4, None, 0, 0, dim=3, max_neuron_models=None)

    def test_get_args_raises(self):
        pq = PolyQuadratic(4, 4, None, 0, 0, dim=2)
        with pytest.raises(NotImplementedError):
            pq.get_args(torch.randn(2, 2, 2))

def test_all_known_activations_resolve():
    # Sanity: every registered name builds a fresh module.
    for name in _ACTIVATIONS:
        n = LinearPolynomNeuron(3, 3, name, 0, 0)
        assert isinstance(n.activation, nn.Module)
