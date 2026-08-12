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
        assert not n.needs_squash_stats
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
    def test_fit_squash_gathers_per_input_slot_stats(self, cls):
        n = cls(4, 4, None, 0, 0)
        mean = torch.tensor([10.0, 20.0, 30.0, 40.0])
        std = torch.tensor([1.0, 2.0, 3.0, 4.0])
        n.fit_squash(mean, std)
        # neuron k's slot j must carry the stats of the feature column it reads
        for k, (i, j) in enumerate(n.src_idxs.tolist()):
            assert n.squash_norm.mean[k].tolist() == [mean[i], mean[j]]
            assert n.squash_norm.std[k].tolist() == [std[i], std[j]]

    def test_fit_squash_neutralizes_a_constant_feature(self):
        n = LegendrePolynomNeuron(3, 3, None, 0, 0)
        n.fit_squash(torch.tensor([0.0, 5.0, 0.0]), torch.tensor([1.0, 0.0, 1.0]))
        # std 0 → unit scale, so the constant column maps to 0 rather than
        # dividing float noise by ~0 and saturating at random.
        assert (n.squash_norm.std > 0).all()
        x = torch.full((4, n.num_neurons, 2), 5.0)
        assert torch.isfinite(n._squash(x)).all()

    @pytest.mark.parametrize("cls", [LegendrePolynomNeuron, ChebyshevPolynomNeuron])
    def test_prune_keeps_squash_stats_aligned(self, cls):
        n = cls(5, 5, None, 0, 0)
        n.fit_squash(torch.arange(5.0), torch.ones(5))
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
        n.fit_squash(torch.full((4,), 100.0), torch.full((4,), 25.0))
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
        n.fit_squash(torch.arange(4.0), torch.ones(4) * 2)
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

    def test_legacy_metadata_without_squash_keys_restores_as_tanh(self):
        """Checkpoints predating the configurable squash were tanh-squashed;
        defaulting them to the current 'sigma' default would restore them with
        a different — and uncalibrated — nonlinearity."""
        n = LegendrePolynomNeuron(4, 4, None, 0, 0)
        meta = {name: getattr(n, name) for name in n.params_metadata_names}
        for key in ("squash_method", "squash_n_sigma", "squash_core_range"):
            meta.pop(key)
        restored = BasePolynomNeuron.from_checkpoint_metadata(meta)
        assert restored.squash_method == "tanh"
        assert restored.squash_norm is None

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
