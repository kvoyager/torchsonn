import math
import itertools
import logging

import pytest
import torch
from torch import nn

from torchsonn.neurons import (
    BasePolynomNeuron,
    ChebyshevPolynomNeuron,
    CubicPolynomNeuron,
    LegendrePolynomNeuron,
    LinearCovPolynomNeuron,
    LinearPolynomNeuron,
    PolyQuadratic,
    QuadraticPolynomNeuron,
    generate_unique_combinations,
    generate_unique_pairs,
)
from torchsonn.neurons.base import _ACTIVATIONS, _max_unique_tuples
from torchsonn.types import CriterionType, RefFunctionType


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

class TestOrthogonalNeurons:
    @pytest.mark.parametrize(
        "cls,short_prefix,name_word",
        [
            (LegendrePolynomNeuron, "Legendre", "Legendre"),
            (ChebyshevPolynomNeuron, "Chebyshev", "Chebyshev"),
        ],
    )
    def test_defaults_forward_and_metadata(self, cls, short_prefix, name_word):
        # Default degree=3, cross=True → num_w = 1 + 2*3 + 1 = 8, matching
        # CubicPolynomNeuron's width (same layout, orthogonal basis).
        n = cls(num_feat=4, num_src_feat=4, activation=None, layer_index=0, start_index=0)
        assert n.num_w == 8
        assert n.degree == 3 and n.cross is True and n.squash is True
        x = torch.randn(7, 4)
        out = n(x)
        # First layer enumerates all C(4,2) = 6 pairs.
        assert out.shape == (7, 6)
        assert n.get_short_name() == f"{short_prefix}3"
        assert name_word in n.get_name()

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_squash_method_defaults_to_sigma(self, cls):
        n = cls(4, 4, None, 0, 0)
        assert n.squash_method == "sigma"
        assert n.squash_norm is not None
        # Per-neuron-input-slot stats: the leading axis must match num_neurons
        # because Trainer.create_loss_functions vmaps every buffer at in_dims=0.
        assert n.squash_norm.mean.shape == (n.num_neurons, n.dim)
        assert "sigma-squashed" in n.get_name()

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_tanh_method_carries_no_statistics(self, cls):
        n = cls(4, 4, None, 0, 0, squash_method="tanh")
        assert n.squash_norm is None
        assert not n.needs_input_stats
        assert "tanh-squashed" in n.get_name()
        x = torch.randn(5, 6, 2) * 50
        assert torch.allclose(n._squash(x), torch.tanh(x))

    def test_unknown_squash_method_rejected(self):
        with pytest.raises(ValueError):
            LegendrePolynomNeuron(4, 4, None, 0, 0, squash_method="softsign")

    def test_sigma_method_without_stats_refuses_rather_than_using_tanh(self):
        """Silently falling back to tanh would swap the nonlinearity under the
        user without changing any reported config — fail loudly instead."""
        n = LegendrePolynomNeuron(4, 4, None, 0, 0)
        n.squash_norm = None
        with pytest.raises(RuntimeError, match="sigma"):
            n._squash(torch.randn(3, 6, 2))

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_fit_input_stats_gathers_per_input_slot_stats(self, cls):
        n = cls(4, 4, None, 0, 0)
        mean = torch.tensor([10.0, 20.0, 30.0, 40.0])
        std = torch.tensor([1.0, 2.0, 3.0, 4.0])
        n.fit_input_stats(mean, std)
        # neuron k's slot j must carry the stats of the feature column it reads
        for k, (i, j) in enumerate(n.src_idxs.tolist()):
            assert n.squash_norm.mean[k].tolist() == [mean[i], mean[j]]
            assert n.squash_norm.std[k].tolist() == [std[i], std[j]]

    def test_fit_input_stats_neutralizes_a_constant_feature(self):
        n = LegendrePolynomNeuron(3, 3, None, 0, 0)
        n.fit_input_stats(torch.tensor([0.0, 5.0, 0.0]), torch.tensor([1.0, 0.0, 1.0]))
        # std 0 → unit scale, so the constant column maps to 0 rather than
        # dividing float noise by ~0 and saturating at random.
        assert (n.squash_norm.std > 0).all()
        x = torch.full((4, n.num_neurons, 2), 5.0)
        assert torch.isfinite(n._squash(x)).all()

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_prune_keeps_squash_stats_aligned(self, cls):
        n = cls(5, 5, None, 0, 0)
        n.fit_input_stats(torch.arange(5.0), torch.ones(5))
        keep = torch.tensor([2, 0, 7])
        expected = n.squash_norm.mean[keep].clone()
        n.prune(keep)
        assert n.squash_norm.mean.shape == (n.num_neurons, n.dim)
        assert n.squash_norm.std.shape == (n.num_neurons, n.dim)
        # not just the right shape — the right rows, in the right order
        assert torch.equal(n.squash_norm.mean, expected)
        assert torch.isfinite(n(torch.randn(6, 5))).all()

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_sigma_squash_bounds_wildly_scaled_inputs(self, cls):
        n = cls(4, 4, None, 0, 0, degree=6)
        n.fit_input_stats(torch.full((4,), 100.0), torch.full((4,), 25.0))
        x = torch.randn(64, 4) * 25.0 + 100.0
        args = n.get_args(
            torch.index_select(x, 1, n.src_idxs.view(-1)).view(64, -1, n.dim)
        )
        # |P_k| <= 1 on [-1, 1]; unsquashed these columns would explode.
        assert args.abs().max() <= 1.0 + 1e-6
        assert torch.isfinite(n(x)).all()

    def test_squash_knobs_survive_checkpoint_metadata(self):
        n = LegendrePolynomNeuron(4, 4, None, 0, 0, squash_n_sigma=3.0,
                                  squash_core_range=0.5)
        n.fit_input_stats(torch.arange(4.0), torch.ones(4) * 2)
        restored = BasePolynomNeuron.from_checkpoint_metadata(
            {name: getattr(n, name) for name in n.params_metadata_names}
        )
        assert restored.squash_method == "sigma"
        assert restored.squash_norm.n_sigma == 3.0
        assert restored.squash_norm.core_range == 0.5
        # buffers must already have the right shape for load_state_dict
        assert restored.squash_norm.mean.shape == n.squash_norm.mean.shape
        restored.load_state_dict(n.state_dict(), strict=False)
        assert torch.equal(restored.squash_norm.mean, n.squash_norm.mean)

    @pytest.mark.parametrize("key", ["squash_method", "squash_n_sigma", "squash_core_range"])
    def test_metadata_without_a_squash_key_raises(self, key):
        n = LegendrePolynomNeuron(4, 4, None, 0, 0)
        meta = {name: getattr(n, name) for name in n.params_metadata_names}
        meta.pop(key)
        with pytest.raises(KeyError, match=key):
            BasePolynomNeuron.from_checkpoint_metadata(meta)

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    @pytest.mark.parametrize(
        "degree,cross,expected_num_w",
        [
            (1, False, 3),   # 1 + 2*1
            (1, True, 4),    # + bilinear
            (2, False, 5),   # 1 + 2*2
            (2, True, 6),
            (3, True, 8),    # == Cubic
            (5, True, 12),   # 1 + 2*5 + 1
        ],
    )
    def test_num_w_variants(self, cls, degree, cross, expected_num_w):
        n = cls(3, 3, None, 0, 0, degree=degree, cross=cross)
        assert n.num_w == expected_num_w
        assert n.weight.shape[1] == expected_num_w

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_degree_must_be_positive(self, cls):
        with pytest.raises(ValueError):
            cls(3, 3, None, 0, 0, degree=0)

    def test_chebyshev_basis_values(self):
        # squash=False so the raw closed forms apply; cross=True adds xi*xj.
        n = ChebyshevPolynomNeuron(2, 2, None, 0, 0, degree=3, cross=True, squash=False)
        x = torch.tensor([[[0.3, -0.7]]])  # [B=1, T=1, dim=2]
        args = n.get_args(x)
        xi, xj = 0.3, -0.7
        # T_0=1, T_1=x, T_2=2x^2-1, T_3=4x^3-3x
        T = lambda k, v: {0: 1.0, 1: v, 2: 2 * v * v - 1, 3: 4 * v**3 - 3 * v}[k]
        expected = torch.tensor([[[
            1.0,
            T(1, xi), T(1, xj),
            T(2, xi), T(2, xj),
            T(3, xi), T(3, xj),
            xi * xj,
        ]]])
        assert torch.allclose(args, expected, atol=1e-6)

    def test_legendre_basis_values(self):
        n = LegendrePolynomNeuron(2, 2, None, 0, 0, degree=3, cross=True, squash=False)
        x = torch.tensor([[[0.3, -0.7]]])
        args = n.get_args(x)
        xi, xj = 0.3, -0.7
        # P_0=1, P_1=x, P_2=(3x^2-1)/2, P_3=(5x^3-3x)/2
        P = lambda k, v: {0: 1.0, 1: v, 2: 0.5 * (3 * v * v - 1), 3: 0.5 * (5 * v**3 - 3 * v)}[k]
        expected = torch.tensor([[[
            1.0,
            P(1, xi), P(1, xj),
            P(2, xi), P(2, xj),
            P(3, xi), P(3, xj),
            xi * xj,
        ]]])
        assert torch.allclose(args, expected, atol=1e-6)

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_squash_bounds_basis_columns(self, cls):
        # |P_k(u)|, |T_k(u)| <= 1 on [-1, 1]; tanh maps any input into (-1, 1),
        # so with squash=True every column stays bounded even for huge inputs.
        n = cls(2, 2, None, 0, 0, degree=6, cross=True, squash=True)
        x = torch.full((1, 1, 2), 50.0)
        args = n.get_args(x)
        assert args.abs().max() <= 1.0 + 1e-6

    def test_no_squash_lets_columns_grow(self):
        # Without squashing, a raw input outside [-1, 1] blows the basis up —
        # exactly the conditioning problem squash=True exists to avoid.
        n = ChebyshevPolynomNeuron(2, 2, None, 0, 0, degree=5, cross=False, squash=False)
        x = torch.full((1, 1, 2), 3.0)
        args = n.get_args(x)
        assert args.abs().max() > 100.0  # T_5(3) = 3363

    def test_no_cross_drops_bilinear_column(self):
        n = ChebyshevPolynomNeuron(2, 2, None, 0, 0, degree=2, cross=False, squash=False)
        x = torch.tensor([[[0.5, 0.25]]])
        args = n.get_args(x)
        # [1, T1(xi), T1(xj), T2(xi), T2(xj)] — no xi*xj term.
        assert args.shape[-1] == 5
        assert args.shape[-1] == n.num_w

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_checkpoint_roundtrip_preserves_shape(self, cls):
        # from_checkpoint_metadata must rebuild the weight at the checkpoint's
        # width; num_w depends on degree/cross, not dim, so those must survive.
        n = cls(4, 4, None, 0, 0, degree=4, cross=False, squash=False)
        meta = {
            "cls": cls.__name__,
            "num_feat": 4,
            "num_src_feat": 4,
            "activation": None,
            "layer_index": 0,
            "start_index": 0,
            "dim": 2,
            "degree": 4,
            "cross": False,
            "squash": False,
            "squash_method": "sigma",
            "squash_n_sigma": 2.0,
            "squash_core_range": 0.75,
            "src_idxs": n.src_idxs,
        }
        restored = BasePolynomNeuron.from_checkpoint_metadata(meta)
        assert isinstance(restored, cls)
        assert restored.num_w == n.num_w
        assert restored.weight.shape == n.weight.shape
        assert restored.degree == 4
        assert restored.cross is False
        assert restored.squash is False

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_activation_and_max_neuron_models(self, cls):
        n = cls(6, 6, "tanh", 0, 0, max_neuron_models=4, degree=2)
        assert isinstance(n.activation, nn.Tanh)
        assert n.weight.shape[0] == 4
        assert n.src_idxs.shape == (4, 2)


class TestMultiInputOrthogonalNeurons:
    """dim > 2: additive univariate terms + pairwise (cross) interactions."""

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    @pytest.mark.parametrize(
        "dim,degree,cross,expected_num_w",
        [
            (3, 1, False, 4),    # 1 + 3*1
            (3, 1, True, 7),     # + C(3,2)=3 pairwise cross terms
            (3, 3, True, 13),    # 1 + 3*3 + 3
            (4, 3, True, 19),    # 1 + 4*3 + C(4,2)=6 — the worked example
            (4, 3, False, 13),   # 1 + 4*3
            (5, 2, True, 21),    # 1 + 5*2 + C(5,2)=10
        ],
    )
    def test_num_w(self, cls, dim, degree, cross, expected_num_w):
        n = cls(6, 6, None, 0, 0, max_neuron_models=4, dim=dim, degree=degree, cross=cross)
        assert n.num_w == expected_num_w
        assert n.weight.shape[1] == expected_num_w
        # One bilinear column per input pair, none when cross is off.
        assert len(n._cross_pairs) == (dim * (dim - 1) // 2 if cross else 0)

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_dim_must_be_at_least_two(self, cls):
        with pytest.raises(ValueError):
            cls(6, 6, None, 0, 0, dim=1)

    def test_cross_terms_are_pairwise_not_nway_product(self):
        # The critical dim > 2 correctness property: the cross block holds the
        # C(dim,2) pairwise products u_i*u_j, NOT the single dim-way product
        # u_0*u_1*...*u_{dim-1} that torch.prod(u, dim=-1) would give.
        n = LegendrePolynomNeuron(3, 3, None, 0, 0, max_neuron_models=1,
                                  dim=3, degree=1, cross=True, squash=False)
        a, b, c = 0.5, -0.3, 0.2
        x = torch.tensor([[[a, b, c]]])  # [B=1, T=1, dim=3]
        args = n.get_args(x)
        # degree 1 ⇒ P_1(v)=v, so: [1, a, b, c, a*b, a*c, b*c]
        expected = torch.tensor([[[1.0, a, b, c, a * b, a * c, b * c]]])
        assert args.shape[-1] == n.num_w == 7
        assert torch.allclose(args, expected, atol=1e-6)
        # Guard the exact bug: the 3-way product must be absent from the row.
        nway = a * b * c
        assert not torch.any((args - nway).abs() < 1e-6)

    def test_multi_input_univariate_block_ordering(self):
        # Full row for dim=3, degree=2, cross=True on the Chebyshev basis:
        # [1, T1(a),T1(b),T1(c), T2(a),T2(b),T2(c), a*b,a*c,b*c] — the univariate
        # columns are grouped by degree, then by input, then the pairwise block.
        n = ChebyshevPolynomNeuron(3, 3, None, 0, 0, max_neuron_models=1,
                                   dim=3, degree=2, cross=True, squash=False)
        a, b, c = 0.5, -0.3, 0.2
        x = torch.tensor([[[a, b, c]]])
        args = n.get_args(x)
        T2 = lambda v: 2 * v * v - 1  # T_2(x) = 2x^2 - 1
        expected = torch.tensor([[[
            1.0,
            a, b, c,
            T2(a), T2(b), T2(c),
            a * b, a * c, b * c,
        ]]])
        assert args.shape[-1] == n.num_w == 10
        assert torch.allclose(args, expected, atol=1e-6)

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_src_idxs_and_forward_shape(self, cls):
        # dim-ary src tuples and an end-to-end forward through the ensemble.
        n = cls(5, 5, None, 0, 0, max_neuron_models=4, dim=3, degree=3)
        assert n.src_idxs.shape == (4, 3)
        assert n.weight.shape[0] == 4
        out = n(torch.randn(7, 5))
        assert out.shape == (7, 4)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_exhaustive_enumeration_all_dim_tuples(self, cls):
        # max_neuron_models=None enumerates every unordered dim-tuple: C(4,3)=4.
        n = cls(4, 4, None, 0, 0, dim=3)
        assert n.num_neurons == 4
        assert n.src_idxs.shape == (4, 3)
        tuples = {tuple(row.tolist()) for row in n.src_idxs}
        assert tuples == {(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)}

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_zero_neurons_when_fewer_features_than_dim(self, cls):
        # num_feat < dim ⇒ no valid input tuple ⇒ empty ensemble, which
        # SONN.create_layer detects (num_neurons == 0) and skips for the layer.
        n = cls(2, 2, None, 0, 0, max_neuron_models=6, dim=4)
        assert n.num_neurons == 0

    def test_short_name_encodes_arity_only_when_non_default(self):
        # Pair neuron keeps the historical short name; multi-input appends xN.
        pair = LegendrePolynomNeuron(4, 4, None, 0, 0, degree=3, dim=2)
        multi = LegendrePolynomNeuron(6, 6, None, 0, 0, max_neuron_models=4, degree=3, dim=4)
        assert pair.get_short_name() == "Legendre3"
        assert multi.get_short_name() == "Legendre3x4"
        assert "4 inputs" in multi.get_name()

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_checkpoint_roundtrip_preserves_dim_and_width(self, cls):
        # num_w now depends on dim, so _construct_from_metadata must forward the
        # saved dim to rebuild the weight at the checkpoint's width.
        n = cls(6, 6, None, 0, 0, max_neuron_models=4, dim=4, degree=3, cross=True, squash=False)
        meta = {
            "cls": cls.__name__,
            "num_feat": 6,
            "num_src_feat": 6,
            "activation": None,
            "layer_index": 0,
            "start_index": 0,
            "dim": 4,
            "degree": 3,
            "cross": True,
            "squash": False,
            "squash_method": "sigma",
            "squash_n_sigma": 2.0,
            "squash_core_range": 0.75,
            "src_idxs": n.src_idxs,
        }
        restored = BasePolynomNeuron.from_checkpoint_metadata(meta)
        assert isinstance(restored, cls)
        assert restored.dim == 4
        assert restored.num_w == n.num_w == 19
        assert restored.weight.shape == n.weight.shape
        assert len(restored._cross_pairs) == 6

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_squash_bounds_multi_input_columns(self, cls):
        # tanh maps every input into (-1, 1), where |P_k|, |T_k| <= 1 and the
        # pairwise products are bounded too — so no column blows up even for
        # huge raw inputs, exactly as in the dim=2 case.
        n = cls(4, 4, None, 0, 0, dim=4, degree=6, cross=True, squash=True)
        x = torch.full((1, 1, 4), 50.0)
        args = n.get_args(x)
        assert args.abs().max() <= 1.0 + 1e-6


def test_all_known_activations_resolve():
    # Sanity: every registered name builds a fresh module.
    for name in _ACTIVATIONS:
        n = LinearPolynomNeuron(3, 3, name, 0, 0)
        assert isinstance(n.activation, nn.Module)


# --- BaseTupleNeuron and the input-pass hooks ----------------------------

class TestBaseTupleNeuron:
    # Captured from the code before the refactor: the orthogonal families'
    # state_dict keys and metadata names must not change, or checkpoints
    # written before it would not load.
    LEGENDRE_STATE_KEYS = ["params_metadata", "squash_norm.mean", "squash_norm.params_metadata",
                           "squash_norm.std", "weight"]
    LEGENDRE_METADATA = ["cls", "num_feat", "num_src_feat", "activation", "layer_index", "start_index",
                         "dim", "max_neuron_models", "src_idxs", "created_neuron_idxs", "degree",
                         "cross", "squash", "squash_method", "squash_n_sigma", "squash_core_range"]

    def test_orthogonal_families_keep_state_and_metadata(self):
        from torchsonn.neurons.base import BaseTupleNeuron
        for cls in (LegendrePolynomNeuron, ChebyshevPolynomNeuron):
            n = cls(5, 5, None, 0, 0, degree=3, max_neuron_models=4)
            assert isinstance(n, BaseTupleNeuron)
            assert sorted(n.state_dict()) == self.LEGENDRE_STATE_KEYS
            assert n.params_metadata_names == self.LEGENDRE_METADATA
            meta = {k: getattr(n, k) for k in n.params_metadata_names}
            back = BasePolynomNeuron.from_checkpoint_metadata(meta)
            back.load_state_dict(n.state_dict(), strict=False)
            x = torch.randn(7, 5)
            assert torch.equal(back(x), n(x))

    def test_input_stats_hooks(self):
        n = LegendrePolynomNeuron(4, 4, None, 0, 0, max_neuron_models=3)
        assert n.needs_input_stats is True
        n.fit_input_stats(torch.arange(4.0), torch.ones(4) * 2)
        assert torch.equal(n.squash_norm.mean, torch.arange(4.0)[n.src_idxs])
        tanh = LegendrePolynomNeuron(4, 4, None, 0, 0, max_neuron_models=3, squash_method="tanh")
        assert tanh.needs_input_stats is False
        # The new hooks default to "nothing wanted" on every family.
        for nm in (n, LinearCovPolynomNeuron(4, 4, None, 0, 0), PolyQuadratic(4, 4, None, 0, 0, dim=3, max_neuron_models=3)):
            assert nm.needs_input_sample is False and nm.needs_input_stream is False
            nm.fit_input_sample(torch.zeros(2, 4))
            nm.stream_input_batch(torch.zeros(2, 4))
            nm.finish_input_stream()

    def test_tuple_enumeration(self):
        from torchsonn.neurons.base import BaseTupleNeuron

        class Triple(BaseTupleNeuron):
            num_w = 4

            def get_args(self, x):
                return torch.cat([torch.ones_like(x[..., :1]), x], dim=-1)

            def get_name(self):
                return "triple"

            def get_short_name(self):
                return "T"

        with pytest.raises(ValueError, match="dim must be >= 2"):
            Triple(5, 5, None, 0, 0, dim=1)
        full = Triple(5, 5, None, 0, 0, dim=3)
        assert full.dim == 3 and full.num_neurons == 10           # C(5, 3)
        assert full.src_idxs.tolist() == [list(t) for t in itertools.combinations(range(5), 3)]
        capped = Triple(5, 5, None, 0, 0, dim=3, max_neuron_models=4)
        assert capped.num_neurons == 4 and capped.weight.shape == (4, 4)
        assert all(a < b < c for a, b, c in capped.src_idxs.tolist())
        clamped = Triple(5, 5, None, 0, 0, dim=3, max_neuron_models=50)
        assert clamped.num_neurons == 10
        empty = Triple(2, 2, None, 0, 0, dim=3)
        assert empty.num_neurons == 0
        assert full(torch.randn(6, 5)).shape == (6, 10)
        # dim round-trips through the base metadata path.
        meta = {k: getattr(full, k) for k in full.params_metadata_names}
        assert BasePolynomNeuron.from_checkpoint_metadata(meta).dim == 3


# --- The RBF family ----------------------------------------------------------

from torch.func import functional_call, grad as _fgrad, vmap as _fvmap

from torchsonn.neurons import RBFNeuron


def _rbf(num_feat=4, **kw):
    kw.setdefault("max_neuron_models", 3)
    return RBFNeuron(num_feat, num_feat, None, 0, 0, **kw)


def _slots(n, x):
    return torch.index_select(x, 1, n.src_idxs.view(-1)).view(x.shape[0], -1, n.dim)


def _design(n, x):
    """Design rows (N, num_w) of a single-candidate neuron."""
    assert n.num_neurons == 1
    return n.get_args(_slots(n, x))[:, 0, :]


def _two_clusters(n_rows=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n_rows // 2, 2, generator=g) * 0.3 + torch.tensor([3.0, 3.0])
    b = torch.randn(n_rows // 2, 2, generator=g) * 0.3 + torch.tensor([-3.0, -3.0])
    return torch.cat([a, b])[torch.randperm(n_rows, generator=g)]


def _calibrate(n, x, **kw):
    n.fit_input_stats(x.mean(0), x.std(0, unbiased=False))
    n.fit_input_sample(x, **kw)


class TestRBFNeuron:
    def test_num_w_rule_and_validation(self):
        for m, dim, norm, lin in itertools.product((2, 9, 16), (2, 3), (True, False), (True, False)):
            n = _rbf(5, dim=dim, centers=m, normalize=norm, linear=lin)
            assert n.num_w == m + (dim if lin else 0) + (0 if norm else 1)
            assert n.weight.shape == (3, n.num_w)
            assert n.centers().shape == (3, m, dim) and n.center_shift.shape == (3, m, dim)
            assert n.log_width.shape == (3, m) and n.radius.shape == (3, m)
            assert n.get_args(_slots(n, torch.randn(6, 5))).shape == (6, 3, n.num_w)
        for bad in ({"centers": 1}, {"width": 0.0}, {"width_band": 1.0}, {"center_radius": 0.0}, {"placement": "random"},
                    {"dim": 1}, {"placement": "grid", "centers": 10}):
            with pytest.raises(ValueError):
                _rbf(**bad)
        assert _rbf(placement="grid", centers=9).num_w == 11

    def test_normalized_bumps_are_a_partition_of_unity_even_far_away(self):
        n = _rbf()
        x = torch.randn(200, 4)
        _calibrate(n, x, seed=0)
        far = torch.cat([x, x * 1e3, x + 1e4])
        phi = n.get_args(_slots(n, far))[..., :n.num_centers]
        assert torch.isfinite(phi).all()
        assert torch.allclose(phi.sum(-1), torch.ones(phi.shape[:-1]), atol=1e-6)
        raw = _rbf(normalize=False)
        _calibrate(raw, x, seed=0)
        g = raw.get_args(_slots(raw, far))[..., :raw.num_centers]
        assert torch.isfinite(g).all() and (g <= 1.0).all() and (g >= 0.0).all()

    def test_constant_and_linear_targets_are_nested(self):
        n = _rbf(2, max_neuron_models=None)
        x = torch.randn(400, 2) * 2 + 1
        _calibrate(n, x, seed=0)
        phi = _design(n, x)
        for y in (torch.full((400,), 3.0), 2.0 * x[:, 0] - x[:, 1] + 0.5):
            sol = torch.linalg.lstsq(phi, y.unsqueeze(1)).solution
            assert ((phi @ sol).squeeze(1) - y).abs().max() < 1e-3

    def test_vmap_matches_eager_including_gradients(self):
        n = _rbf()
        x = torch.randn(64, 4)
        _calibrate(n, x, seed=0)
        params = dict(n.named_parameters())
        buffers = dict(n.named_buffers())
        buffers["src_idxs"] = n.src_idxs
        in_dims = ({k: 0 for k in params}, {k: 0 for k in buffers}, None)

        def loss(p, b, x):
            return functional_call(n, {**p, **b}, (x,)).pow(2).mean()

        eager = n(x)
        vm = _fvmap(lambda p, b, x: functional_call(n, {**p, **b}, (x,)), in_dims=in_dims)(params, buffers, x)
        assert torch.allclose(vm.T, eager, atol=1e-5)
        g_vm = _fvmap(_fgrad(loss), in_dims=in_dims)(params, buffers, x)
        n.zero_grad()
        for k in range(n.num_neurons):
            eager[:, k].pow(2).mean().backward(retain_graph=True)
        for name in ("weight", "center_shift", "log_width"):
            assert torch.allclose(g_vm[name], getattr(n, name).grad, atol=1e-5), name

    @pytest.mark.parametrize("learn", [True, False])
    def test_checkpoint_round_trip(self, learn):
        n = _rbf(learn_centers=learn, learn_widths=learn, placement="grid", centers=9, width=0.8)
        x = torch.randn(300, 4) * 4 - 2
        _calibrate(n, x, seed=0)
        with torch.no_grad():
            n.log_width.add_(0.3)
        assert isinstance(n.center_shift, nn.Parameter) is learn
        assert {"centers0", "radius", "center_shift", "width0"} <= set(n.state_dict())
        meta = {k: getattr(n, k) for k in n.params_metadata_names}
        back = BasePolynomNeuron.from_checkpoint_metadata(meta)
        assert isinstance(back, RBFNeuron) and back.num_centers == 9 and back.placement == "grid"
        assert isinstance(back.center_shift, nn.Parameter) is learn
        back.load_state_dict(n.state_dict(), strict=False)
        assert torch.equal(back(x), n(x))
        assert back.get_name() == n.get_name()

    def test_prune_keeps_every_per_neuron_tensor_aligned(self):
        n = _rbf(6, max_neuron_models=5)
        x = torch.randn(200, 6)
        _calibrate(n, x, seed=0)
        with torch.no_grad():
            n.center_shift.add_(torch.randn_like(n.center_shift) * 0.5)
        before = n(x)
        keep = torch.tensor([4, 1])
        expect = {name: getattr(n, name).detach()[keep].clone()
                  for name in ("centers0", "radius", "center_shift", "log_width", "width0", "in_mean", "in_std")}
        start = n.centers().detach()[keep].clone()
        n.prune(keep)
        assert isinstance(n.center_shift, nn.Parameter) and isinstance(n.log_width, nn.Parameter)
        for name, t in expect.items():
            assert torch.equal(getattr(n, name).detach(), t), name
        assert torch.equal(n.centers().detach(), start)
        assert dict(n.named_buffers()).keys() >= {"width0", "in_mean", "in_std"}
        assert torch.equal(n(x), before[:, keep])

    def test_kmeans_finds_two_clusters_deterministically(self):
        x = _two_clusters()
        n = _rbf(2, centers=2, max_neuron_models=None)
        _calibrate(n, x, seed=3)
        u_means = torch.stack([((x[x[:, 0] > 0]).mean(0) - n.in_mean[0]) / n.in_std[0],
                               ((x[x[:, 0] < 0]).mean(0) - n.in_mean[0]) / n.in_std[0]])
        c = n.centers().detach()[0]
        for target in u_means:
            assert (c - target).norm(dim=-1).min() < 0.1
        again = _rbf(2, centers=2, max_neuron_models=None)
        _calibrate(again, x, seed=3)
        assert torch.equal(again.centers().detach(), n.centers().detach())
        other = _rbf(2, centers=2, max_neuron_models=None)
        _calibrate(other, x, seed=4)
        # same clusters, possibly the other order
        assert (other.centers().detach()[0].flip(0) - c).abs().max() < 1e-4 or \
            (other.centers().detach()[0] - c).abs().max() < 1e-4

    def test_constant_slot_gives_floored_widths_and_finite_output(self):
        x = torch.randn(300, 3)
        x[:, 1] = 7.0
        n = _rbf(3, centers=4, max_neuron_models=None)
        _calibrate(n, x, seed=0)
        assert (n.width0 >= 0.05 - 1e-6).all()
        out = n(x)
        assert torch.isfinite(out).all() and torch.isfinite(n.get_args(_slots(n, x))).all()

    def test_stream_mode_matches_exact_lloyd(self):
        x = _two_clusters(n_rows=4096, seed=1)
        exact = _rbf(2, centers=2, max_neuron_models=None)
        _calibrate(exact, x, seed=5)
        streamed = _rbf(2, centers=2, max_neuron_models=None)
        streamed.fit_input_stats(x.mean(0), x.std(0, unbiased=False))
        streamed.fit_input_sample(x[:512], stream=True, seed=5)
        assert streamed.needs_input_stream and streamed._stream_counts is not None
        for i in range(0, 4096, 256):
            streamed.stream_input_batch(x[i:i + 256])
        after_one = streamed.centers0.detach().clone()
        for i in range(0, 4096, 256):
            streamed.stream_input_batch(x[i:i + 256])
        streamed.finish_input_stream()
        assert streamed._stream_counts is None
        c_exact = exact.centers().detach()[0]
        c_stream = streamed.centers().detach()[0]
        for row in c_exact:
            assert (c_stream - row).norm(dim=-1).min() < 0.05
        assert (streamed.centers0.detach() - after_one).abs().max() < 0.01
        assert streamed.fit_report() is not None
        # a grid family does not stream, and the stream hooks are no-ops on it
        grid = _rbf(2, centers=4, placement="grid", max_neuron_models=None)
        assert not grid.needs_input_stream
        grid.stream_input_batch(x[:8])
        grid.finish_input_stream()

    @pytest.mark.parametrize("seeding, start", [("pca_quantiles", "PCA-quantile start"),
                                                ("kmeans++", "k-means++ start")])
    def test_stream_start_log_names_the_seeding(self, caplog, seeding, start):
        x = _two_clusters(n_rows=512, seed=1)
        n = _rbf(2, centers=2, max_neuron_models=None, seeding=seeding)
        n.fit_input_stats(x.mean(0), x.std(0, unbiased=False))
        caplog.set_level(logging.INFO, logger="torchsonn.neurons.rbf")
        n.fit_input_sample(x, stream=True, seed=5)
        lines = [rec.getMessage() for rec in caplog.records if "streaming pass follows" in rec.getMessage()]
        assert len(lines) == 1
        assert f": {start} on 512 sampled rows" in lines[0]

    def test_local_bump_beats_legendre_design(self):
        """A fixed 16-center k-means basis resolves a bump comparable to its
        center spacing (radius ~0.7 std here: measured ratio 0.26 of the
        Legendre-3 residual; 0.07 with 25 centers). A bump much narrower
        than the spacing is what the learnable centers are for, see the
        joint-fit test."""
        g = torch.Generator().manual_seed(0)
        u = torch.randn(3000, 2, generator=g)
        y = torch.exp(-((u[:, 0] - 0.4) ** 2 + (u[:, 1] + 0.3) ** 2) / 0.5)
        rbf = _rbf(2, max_neuron_models=None)
        _calibrate(rbf, u, seed=0)
        leg = LegendrePolynomNeuron(2, 2, None, 0, 0, degree=3)
        leg.fit_input_stats(u.mean(0), u.std(0, unbiased=False))

        def resid(phi):
            sol = torch.linalg.lstsq(phi, y.unsqueeze(1)).solution
            return ((phi @ sol).squeeze(1) - y).pow(2).mean().item()

        r_rbf = resid(_design(rbf, u))
        r_leg = resid(leg.get_args(_slots(leg, u))[:, 0, :])
        assert r_rbf < 0.4 * r_leg, (r_rbf, r_leg)

    def _fit_joint(self, n, u, y, steps=150):
        opt = torch.optim.LBFGS([p for p in n.parameters()], lr=0.5, max_iter=20, history_size=20,
                                line_search_fn="strong_wolfe")
        for _ in range(steps // 20):
            def closure():
                opt.zero_grad()
                loss = (n(u)[:, 0] - y).pow(2).mean()
                loss.backward()
                return loss
            opt.step(closure)
        with torch.no_grad():
            return (n(u)[:, 0] - y).pow(2).mean().item()

    def test_joint_fit_moves_a_center_onto_the_bump(self):
        g = torch.Generator().manual_seed(1)
        u = torch.randn(3000, 2, generator=g)
        target = torch.tensor([0.4, -0.3])
        y = torch.exp(-((u - target) ** 2).sum(1) / 0.05)
        n = _rbf(2, max_neuron_models=None)
        _calibrate(n, u, seed=0)
        # standardized coordinates of the bump
        t_std = (target - n.in_mean[0]) / n.in_std[0]
        fixed = torch.linalg.lstsq(_design(n, u), y.unsqueeze(1)).solution
        r_fixed = ((_design(n, u) @ fixed).squeeze(1) - y).pow(2).mean().item()
        start = n.centers().detach().clone()
        r_joint = self._fit_joint(n, u, y)
        assert r_joint < r_fixed
        assert (n.centers().detach()[0] - t_std).norm(dim=-1).min() < 0.1
        assert not torch.equal(n.centers().detach(), start)
        # every center stays inside its radius around the k-means start
        assert (n.displacements().detach().norm(dim=-1) < n.radius).all()
        with torch.no_grad():
            scale = n.width_scales()
        assert (scale <= 4.0).all() and (scale >= 0.25).all()
        assert torch.allclose(n.widths(), n.width0 * scale)
        assert "center movement" in n.fit_report()

    def test_frozen_centers_stay_bit_identical_under_the_same_fit(self):
        g = torch.Generator().manual_seed(1)
        u = torch.randn(1000, 2, generator=g)
        y = torch.exp(-((u - torch.tensor([0.4, -0.3])) ** 2).sum(1) / 0.05)
        n = _rbf(2, max_neuron_models=None, learn_centers=False, learn_widths=False)
        _calibrate(n, u, seed=0)
        c0, w0 = n.center_shift.clone(), n.log_width.clone()
        assert [name for name, _ in n.named_parameters()] == ["weight"]
        self._fit_joint(n, u, y, steps=40)
        assert torch.equal(n.center_shift, c0) and torch.equal(n.log_width, w0)

    def test_dim_three_builds_places_and_runs(self):
        n = _rbf(5, dim=3, centers=8, max_neuron_models=4)
        x = torch.randn(500, 5)
        _calibrate(n, x, seed=0)
        assert n.get_short_name() == "RBF8x3" and n.num_w == 8 + 3
        assert n(x).shape == (500, 4) and torch.isfinite(n(x)).all()
        assert n.centers().shape == (4, 8, 3)

    def test_standardize_off_uses_raw_units(self):
        x = torch.randn(300, 4) * 50 + 200
        n = _rbf(standardize=False)
        _calibrate(n, x, seed=0)
        assert n.centers().detach().abs().mean() > 50        # raw units, not z-scores
        assert (n.width0 >= 0.05 * n.in_std.mean(-1, keepdim=True) - 1e-4).all()
        assert torch.isfinite(n(x)).all()
        assert "standardized" not in n.get_name()

    def test_create_layer_builds_rbf_from_yaml(self):
        from omegaconf import OmegaConf
        from torchsonn.config import SONNConfig
        from torchsonn.model import SONN
        cfg = OmegaConf.merge(OmegaConf.structured(SONNConfig), OmegaConf.create({
            "model": {"type": "regressor", "num_classes": 1, "nbest_neurons": 2, "soft_binner": False,
                      "max_neuron_models": 3, "shortcut": False,
                      "ref_functions": [{"rbf": {"centers": 9, "placement": "grid", "learn_widths": False}},
                                        "rbf"]},
            "train": {"device": "cpu"},
        }))
        model = SONN(cfg, d_model=4)
        layer = model.create_layer(0)
        a, b = layer.neuron_models
        assert isinstance(a, RBFNeuron) and a.num_centers == 9 and a.placement == "grid"
        assert isinstance(a.log_width, torch.Tensor) and not isinstance(a.log_width, nn.Parameter)
        assert isinstance(b, RBFNeuron) and b.num_centers == 16 and b.placement == "kmeans"
        assert RefFunctionType.get("rbf") is RefFunctionType.rfRBF
        assert RefFunctionType.get_name(RefFunctionType.rfRBF) == "RBF"

    def test_center_radius_null_is_unbounded(self):
        n = _rbf(2, max_neuron_models=None, center_radius=None)
        x = torch.randn(300, 2)
        _calibrate(n, x, seed=0)
        assert n.center_radius is None and "unbounded" in n.get_name()
        with torch.no_grad():
            n.center_shift.add_(torch.full_like(n.center_shift, 7.0))
        assert torch.equal(n.displacements(), n.center_shift)
        assert (n.displacements().norm(dim=-1) > n.radius).all()
        assert "unbounded" in n.fit_report() and "beyond 5 std" in n.fit_report()
        meta = {k: getattr(n, k) for k in n.params_metadata_names}
        back = BasePolynomNeuron.from_checkpoint_metadata(meta)
        assert back.center_radius is None
        back.load_state_dict(n.state_dict(), strict=False)
        assert torch.equal(back(x), n(x))
        with pytest.raises(ValueError, match="center_radius"):
            _rbf(center_radius=0.0)



    # --- deterministic k-means: one-hot sums and PCA-quantile seeding ---------

    def test_one_hot_sums_match_scatter_add(self):
        torch.manual_seed(0)
        u = torch.randn(3, 50, 2)
        assign = torch.randint(0, 4, (3, 50))
        sums, counts = RBFNeuron._cluster_sums(u, assign, 4)
        ref_s = torch.zeros(3, 4, 2).scatter_add_(1, assign.unsqueeze(-1).expand(3, 50, 2), u)
        ref_c = torch.zeros(3, 4).scatter_add_(1, assign, torch.ones(3, 50))
        assert torch.allclose(sums, ref_s, atol=1e-5) and torch.equal(counts, ref_c)

    def test_pca_quantile_seeds_are_rows_at_quantiles_of_the_main_axis(self):
        g = torch.Generator().manual_seed(0)
        base = torch.randn(400, 2, generator=g)
        u = torch.stack([base @ torch.tensor([[3.0, 0.0], [0.0, 0.5]]),          # spread along x
                         base @ torch.tensor([[1.0, 1.0], [-0.2, 0.2]])])         # spread along the diagonal
        seeds = RBFNeuron._pca_quantile_seeds(u, 8)
        assert seeds.shape == (2, 8, 2)
        for n in range(2):
            for s in seeds[n]:
                assert ((u[n] - s).abs().sum(-1) < 1e-6).any()                   # a real row
        # candidate 0: the axis is x, so the seeds are ordered in x and span it
        xs = seeds[0, :, 0]
        assert torch.all(xs[1:] > xs[:-1]) and xs[0] < -2.0 and xs[-1] > 2.0
        # candidate 1: ordered along the diagonal
        diag = seeds[1] @ torch.tensor([1.0, 1.0])
        assert torch.all(diag[1:] > diag[:-1])
        # no randomness: the same input gives the same seeds without any seed
        assert torch.equal(RBFNeuron._pca_quantile_seeds(u, 8), seeds)

    def test_default_seeding_is_deterministic_and_seed_independent(self):
        x = _two_clusters(n_rows=1000, seed=2)
        a = _rbf(2, centers=4, max_neuron_models=None)
        b = _rbf(2, centers=4, max_neuron_models=None)
        _calibrate(a, x, seed=1)
        _calibrate(b, x, seed=99)
        assert a.seeding == "pca_quantiles"
        assert torch.equal(a.centers().detach(), b.centers().detach())
        assert torch.equal(a.width0, b.width0)
        # both clusters get centers
        c = a.centers().detach()[0]
        for sign in (1.0, -1.0):
            target = ((x[torch.sign(x[:, 0]) == sign]).mean(0) - a.in_mean[0]) / a.in_std[0]
            assert (c - target).norm(dim=-1).min() < 0.5

    def test_kmeanspp_seeding_is_still_available_and_seeded(self):
        x = _two_clusters(n_rows=1000, seed=2)
        a = _rbf(2, centers=4, max_neuron_models=None, seeding="kmeans++")
        b = _rbf(2, centers=4, max_neuron_models=None, seeding="kmeans++")
        c = _rbf(2, centers=4, max_neuron_models=None, seeding="kmeans++")
        _calibrate(a, x, seed=1)
        _calibrate(b, x, seed=1)
        _calibrate(c, x, seed=2)
        assert torch.equal(a.centers().detach(), b.centers().detach())
        assert "k-means++" in a.get_name() and "PCA" in _rbf(2, max_neuron_models=None).get_name()
        meta = {k: getattr(a, k) for k in a.params_metadata_names}
        assert BasePolynomNeuron.from_checkpoint_metadata(meta).seeding == "kmeans++"
        missing = dict(meta)
        missing.pop("seeding")
        with pytest.raises(KeyError, match="seeding"):
            BasePolynomNeuron.from_checkpoint_metadata(missing)
        with pytest.raises(ValueError, match="seeding"):
            _rbf(seeding="random")
