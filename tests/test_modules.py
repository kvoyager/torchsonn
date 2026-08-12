import pytest
import torch
from torch import nn

from torchsonn.modules import SigmaSquashNorm, SONNModule, SoftBinner


class _Demo(SONNModule):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.alpha = 0.5
        self.params_metadata_names.append("alpha")


class TestSONNModule:
    def test_state_dict_includes_params_metadata(self):
        m = _Demo()
        sd = m.state_dict()
        assert "params_metadata" in sd
        assert sd["params_metadata"] == {"alpha": 0.5}

    def test_state_dict_prefix_is_respected(self):
        m = _Demo()
        sd = m.state_dict(prefix="root.")
        assert "root.params_metadata" in sd

    def test_load_state_dict_restores_metadata(self):
        m1 = _Demo()
        m1.alpha = 9.0
        sd = m1.state_dict()

        m2 = _Demo()
        out = m2.load_state_dict(sd, strict=False)
        assert out is m2  # returns self
        assert m2.alpha == 9.0

    def test_set_metadata_dict_skips_unknown_keys(self):
        m = _Demo()
        m._set_metadata_dict({"alpha": 7.0, "ignore_me": 100})
        assert m.alpha == 7.0
        assert not hasattr(m, "ignore_me")


class TestSoftBinner:
    def test_output_shape_2d_input(self):
        sb = SoftBinner(n_bins=5, scale=10.0)
        x = torch.tensor([0.1, 0.5, 0.9])
        out = sb(x)
        assert out.shape == (3, 5)

    def test_centers_evenly_spread(self):
        sb = SoftBinner(n_bins=4)
        # Centers go from 0.05 to 0.95 inclusive
        assert torch.allclose(sb.centers[0], torch.tensor(0.05))
        assert torch.allclose(sb.centers[-1], torch.tensor(0.95))

    def test_logit_is_max_at_nearest_center(self):
        sb = SoftBinner(n_bins=10, scale=100.0)
        x = torch.tensor([0.05])  # exactly at first center
        out = sb(x)
        assert int(out.argmax(dim=-1).item()) == 0

    def test_higher_dim_input_transposes(self):
        sb = SoftBinner(n_bins=3, scale=5.0)
        # input with extra trailing dim (e.g. seq dim) — should transpose internally
        x = torch.rand(4, 6)  # interpreted as (B, T)
        out = sb(x)
        # logits axis moves: from (B, T, n_bins) → (B, n_bins, T)
        assert out.shape == (4, 3, 6)


def _unit_norm(**kwargs) -> SigmaSquashNorm:
    """Standard-normal layer in float64 — the smoothness assertions below
    compare derivative jumps down near 1e-6, well under float32 resolution."""
    return SigmaSquashNorm(mean=0.0, std=1.0, **kwargs).double()


class TestSigmaSquashNorm:
    def test_core_is_exactly_linear_within_n_sigma(self):
        n = _unit_norm()  # n_sigma=2, core_range=0.75
        z = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(n(z), 0.375 * z)

    def test_n_sigma_lands_on_core_range(self):
        for n_sigma, core_range in [(2.0, 0.75), (3.0, 0.75), (1.5, 0.9)]:
            n = _unit_norm(n_sigma=n_sigma, core_range=core_range)
            edge = n(torch.tensor([n_sigma, -n_sigma], dtype=torch.float64))
            assert torch.allclose(
                edge, torch.tensor([core_range, -core_range], dtype=torch.float64)
            )

    def test_output_never_leaves_the_unit_interval(self):
        """Hard bound: nothing, however extreme, escapes [-1, 1]."""
        n = _unit_norm()
        z = torch.tensor([-1e30, -1e9, -50.0, 0.0, 50.0, 1e9, 1e30],
                         dtype=torch.float64)
        y = n(z)
        assert torch.isfinite(y).all()
        assert (y.abs() <= 1.0).all()

    def test_saturation_is_not_reached_over_the_plausible_range(self):
        """The gap to 1 stays representable far past any realistic outlier — a
        polynomial tail rather than tanh's exponential one. tanh in float32 is
        already at exactly 1.0 by 9 sigma; this map holds out to ~2.7e3 (~6e7
        in float64), past which it does round to 1.0."""
        n = _unit_norm()
        z = torch.logspace(0, 4, 5000, dtype=torch.float64)
        assert (n(z).abs() < 1.0).all()

    def test_odd_and_strictly_increasing(self):
        n = _unit_norm()
        z = torch.linspace(-40, 40, 20001, dtype=torch.float64)
        y = n(z)
        assert torch.allclose(n(-z), -y, atol=1e-14)
        assert (y.diff() > 0).all()

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_smooth_through_the_junction(self, order):
        """Value, slope and curvature are all continuous at +/-n_sigma.

        Probing at 2±h always moves by O(h) even for a perfectly smooth map, so
        the test is that the gap shrinks with h — a real discontinuity would not.
        """
        n = _unit_norm()

        def nth_derivative(z0: float) -> float:
            z = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
            out = n(z)
            for _ in range(order):
                out, = torch.autograd.grad(out, z, create_graph=True)
            return out.item()

        for h in (1e-6, 1e-7, 1e-8):
            jump = abs(nth_derivative(2.0 + h) - nth_derivative(2.0 - h))
            assert jump < 100 * h

    def test_gradient_is_finite_and_positive_everywhere(self):
        """Guards the two `torch.where` traps: a NaN from the unselected branch,
        and the dead gradient at z == 0 that `torch.sign` would introduce."""
        n = _unit_norm()
        # 1e30 is well past the point where the gradient underflows to 0 (~4e7
        # sigma in float64); it is here to prove nothing turns into NaN.
        z = torch.tensor([-1e30, -1e4, -2.0, 0.0, 2.0, 1e4, 1e30],
                         dtype=torch.float64, requires_grad=True)
        grad, = torch.autograd.grad(n(z).sum(), z)
        assert torch.isfinite(grad).all()
        assert (grad >= 0).all()
        # strictly positive across the range that can actually carry signal
        assert (grad[1:-1] > 0).all()
        # at the origin the map is its linear core: slope = core_range / n_sigma
        assert grad[3].item() == pytest.approx(0.375)

    def test_per_feature_stats_from_data(self):
        torch.manual_seed(0)
        scale = torch.tensor([1.0, 10.0, 0.1, 100.0])
        x = torch.randn(256, 4) * scale + torch.tensor([5.0, -2.0, 0.0, 7.0])
        n = SigmaSquashNorm.from_data(x)

        assert torch.allclose(n.mean, x.mean(dim=0))
        u = n(x)
        assert u.shape == x.shape
        assert (u.abs() < 1.0).all()
        # each feature is centered by its own stats, so all four share a scale
        assert u.abs().max(dim=0).values.std() < 0.2

    def test_constant_feature_maps_to_zero(self):
        x = torch.cat([torch.randn(32, 1), torch.full((32, 1), 1234.5)], dim=1)
        u = SigmaSquashNorm.from_data(x)(x)
        assert torch.isfinite(u).all()
        assert (u[:, 1] == 0.0).all()

    def test_knobs_survive_a_checkpoint_round_trip(self):
        x = torch.randn(16, 3)
        src = SigmaSquashNorm.from_data(x, n_sigma=3.0, core_range=0.5)

        dst = SigmaSquashNorm(mean=torch.zeros(3), std=torch.ones(3))
        dst.load_state_dict(src.state_dict(), strict=False)

        assert dst.n_sigma == 3.0
        assert dst.core_range == 0.5
        assert torch.allclose(dst(x), src(x))

    @pytest.mark.parametrize("kwargs", [
        {"n_sigma": 0.0},
        {"n_sigma": -1.0},
        {"core_range": 0.0},
        {"core_range": 1.0},   # no room left for the tail to saturate into
        {"core_range": 1.5},
    ])
    def test_rejects_out_of_range_knobs(self, kwargs):
        with pytest.raises(ValueError):
            SigmaSquashNorm(mean=0.0, std=1.0, **kwargs)

    def test_rejects_negative_std_and_mismatched_shapes(self):
        with pytest.raises(ValueError):
            SigmaSquashNorm(mean=0.0, std=-1.0)
        with pytest.raises(ValueError):
            SigmaSquashNorm(mean=torch.zeros(3), std=torch.ones(4))
