import math

import pytest
import torch

from torchsonn.loss import (
    NormMSE,
    bias_error,
    bias_error_js,
    bias_error_l2,
    regularity_error,
    regularity_error_ce,
)


# A target whose mean dominates its spread — the regime where Sum(y^2) and
# Sum((y - ybar)^2) diverge. Mean 100, sd ~1.6, so the energy denominator is
# ~4000x the centered one.
SHIFTED_Y = torch.tensor([98.0, 99.0, 100.0, 101.0, 102.0])
OFFSETS = [10.0, 100.0, 1000.0]


class TestNormMSE:
    def test_zero_loss_on_exact_match(self):
        loss = NormMSE()
        y = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(loss(y, y), torch.tensor(0.0))

    def test_value_energy(self):
        loss = NormMSE(eps=0.0, centered=False)
        y_true = torch.tensor([1.0, 2.0])
        y_pred = torch.tensor([0.0, 0.0])
        # ||y - ŷ||^2 / ||y||^2 = 5/5 = 1
        assert torch.allclose(loss(y_pred, y_true), torch.tensor(1.0))

    def test_mean_predictor_is_one(self):
        loss = NormMSE(eps=0.0)
        y_pred = SHIFTED_Y.mean().expand_as(SHIFTED_Y)
        assert torch.allclose(loss(y_pred, SHIFTED_Y), torch.tensor(1.0))

    def test_offset_invariance(self):
        loss = NormMSE()
        y_pred = torch.tensor([98.5, 99.5, 99.5, 101.5, 101.0])
        base = loss(y_pred, SHIFTED_Y)
        for c in OFFSETS:
            assert torch.allclose(loss(y_pred + c, SHIFTED_Y + c), base, atol=1e-5)

    def test_energy_is_not_offset_invariant(self):
        # Guards the flag itself: `energy` must still reproduce the old,
        # offset-dependent behaviour for gmdhpy parity.
        loss = NormMSE(centered=False)
        y_pred = torch.tensor([98.5, 99.5, 99.5, 101.5, 101.0])
        base = loss(y_pred, SHIFTED_Y)
        shifted = loss(y_pred + 1000.0, SHIFTED_Y + 1000.0)
        assert shifted.item() < base.item() / 50

    def test_fixed_scale_is_batch_size_independent(self):
        # Per-batch normalization makes the loss depend on how the eval set was
        # chunked; a fixed scale does not.
        y = SHIFTED_Y
        y_hat = y + torch.tensor([0.5, -0.5, 0.5, -0.5, 0.5])
        var = y.var(unbiased=False).item()
        loss = NormMSE(eps=0.0, scale=var)

        whole = loss(y_hat, y)
        # weight each chunk by its sample count to undo the per-call 1/n
        chunks = [(y_hat[:2], y[:2]), (y_hat[2:], y[2:])]
        pooled = sum(loss(p, t) * t.numel() for p, t in chunks) / y.numel()
        assert torch.allclose(whole, pooled, atol=1e-6)

    def test_constant_target_is_finite(self):
        loss = NormMSE()
        y = torch.full((4,), 3.0)
        out = loss(y + 0.1, y)
        assert torch.isfinite(out)


class TestRegularityError:
    def test_perfect_prediction_is_zero(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        out = regularity_error(y, y)
        assert torch.allclose(out, torch.tensor(0.0))

    def test_with_candidate_batch_energy(self):
        # 2 candidates, 4 samples
        y_hat = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = regularity_error(y_hat, y, centered=False)
        assert out.shape == (2,)
        assert torch.allclose(out[0], torch.tensor(0.0))
        # zero predictor: sum((y-0)^2)/sum(y^2) = 1
        assert torch.allclose(out[1], torch.tensor(1.0))

    def test_mean_predictor_is_one(self):
        # The centered baseline is the mean predictor, not the zero predictor.
        y_hat = torch.stack([
            SHIFTED_Y,                                # perfect
            SHIFTED_Y.mean().expand_as(SHIFTED_Y),    # no-skill
        ])
        out = regularity_error(y_hat, SHIFTED_Y)
        assert torch.allclose(out[0], torch.tensor(0.0))
        assert torch.allclose(out[1], torch.tensor(1.0))

    def test_offset_invariance(self):
        # A linear model with intercept is translation-invariant, so the
        # criterion measuring it must be too.
        y_hat = torch.tensor([[98.5, 99.5, 99.5, 101.5, 101.0]])
        base = regularity_error(y_hat, SHIFTED_Y)
        for c in OFFSETS:
            assert torch.allclose(regularity_error(y_hat + c, SHIFTED_Y + c), base, atol=1e-4)

    def test_ranking_unchanged_by_normalization(self):
        y_hat = torch.tensor([
            [98.2, 99.1, 100.1, 100.8, 102.2],
            [97.0, 99.9, 100.0, 102.0, 101.0],
            [99.0, 99.0, 99.0, 102.0, 103.0],
        ])
        centered = regularity_error(y_hat, SHIFTED_Y)
        energy = regularity_error(y_hat, SHIFTED_Y, centered=False)
        assert torch.equal(centered.argsort(), energy.argsort())


class TestBiasError:
    def test_identical_fits_zero(self):
        y_hat = torch.randn(3, 5)
        y = torch.randn(5)
        out = bias_error(y_hat, y_hat, y)
        assert torch.allclose(out, torch.zeros(3), atol=1e-6)

    def test_basic_shape(self):
        y_hat_a = torch.tensor([[1.0, 2.0, 3.0]])
        y_hat_b = torch.tensor([[1.0, 2.0, 4.0]])
        y = torch.tensor([1.0, 2.0, 3.0])
        out = bias_error(y_hat_a, y_hat_b, y)
        assert out.shape == (1,)
        assert out[0].item() > 0

    def test_energy_matches_legacy(self):
        y_hat_a = torch.tensor([[1.0, 2.0, 3.0]])
        y_hat_b = torch.tensor([[1.0, 2.0, 4.0]])
        y = torch.tensor([1.0, 2.0, 3.0])
        out = bias_error(y_hat_a, y_hat_b, y, centered=False)
        # 1 / sum(y^2) = 1/14
        assert torch.allclose(out, torch.tensor([1.0 / 14]), atol=1e-6)

    def test_offset_invariance(self):
        a = torch.tensor([[98.5, 99.5, 99.5, 101.5, 101.0]])
        b = torch.tensor([[98.0, 99.0, 100.0, 101.0, 102.5]])
        base = bias_error(a, b, SHIFTED_Y)
        for c in OFFSETS:
            assert torch.allclose(bias_error(a, b, SHIFTED_Y + c), base, atol=1e-4)


class TestRegularityErrorCE:
    def test_perfect_prediction(self):
        # 4 samples × 3 classes, the model assigns logits massively to the
        # correct class for each sample.
        K = 3
        y = torch.tensor([0, 1, 2, 0])
        logits = torch.full((4, K), -100.0)
        for i, c in enumerate(y.tolist()):
            logits[i, c] = 100.0
        out = regularity_error_ce(logits, y)
        # CE → 0 so NCE → 0
        assert out.item() < 1e-3

    def test_random_logits_finite(self):
        logits = torch.randn(6, 4)
        y = torch.tensor([0, 1, 2, 3, 1, 0])
        out = regularity_error_ce(logits, y)
        assert torch.isfinite(out)
        assert out.item() > 0

    def test_candidate_batch_shape(self):
        # candidate-batch leading dim
        logits = torch.randn(2, 6, 4)
        y = torch.tensor([0, 1, 2, 3, 1, 0])
        out = regularity_error_ce(logits, y)
        assert out.shape == (2,)


class TestBiasErrorL2:
    def test_identical_is_zero(self):
        z = torch.randn(2, 5)
        out = bias_error_l2(z, z)
        assert torch.allclose(out, torch.zeros(2))

    def test_with_explicit_y(self):
        z_a = torch.tensor([[1.0, 2.0, 3.0]])
        z_b = torch.tensor([[1.0, 2.0, 4.0]])
        y = torch.tensor([1.0, 2.0, 3.0])
        out = bias_error_l2(z_a, z_b, y=y)
        # ((1-1)^2 + (2-2)^2 + (3-4)^2) / sum(y^2) = 1/14
        assert torch.allclose(out, torch.tensor([1.0 / 14]), atol=1e-6)

    def test_default_denominator_is_N(self):
        z_a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        z_b = torch.tensor([[1.0, 2.0, 3.0, 5.0]])
        out = bias_error_l2(z_a, z_b)
        # diff = 1, denom = N = 4
        assert torch.allclose(out, torch.tensor([0.25]))

    # --- logits form: what Trainer.bias_err passes for multi-class ---

    def test_logits_identical_is_zero(self):
        z = torch.randn(3, 5, 4)  # 3 candidates, 5 samples, 4 classes
        out = bias_error_l2(z, z, logits=True)
        assert torch.allclose(out, torch.zeros(3))

    def test_logits_reduces_to_candidate_dim(self):
        # Must collapse to (ensemble,), not (ensemble, N): summing over the
        # class axis alone leaves a per-sample tensor that cannot be mixed
        # with the regularity criterion.
        out = bias_error_l2(torch.randn(2, 6, 4), torch.randn(2, 6, 4), logits=True)
        assert out.shape == (2,)

    def test_logits_denominator_is_N_not_K(self):
        z_a = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        z_b = torch.tensor([[[1.0, 2.0, 4.0], [4.0, 7.0, 6.0]]])
        out = bias_error_l2(z_a, z_b, logits=True)
        # ||.||^2 per sample = 1 and 4 -> 5, denom = N = 2 (not K = 3)
        assert torch.allclose(out, torch.tensor([2.5]))

    def test_logits_one_hot_y_matches_default(self):
        # Sum_i ||y_i||^2 = N for one-hot, so explicit y must agree with y=None
        z_a = torch.randn(2, 5, 3)
        z_b = torch.randn(2, 5, 3)
        y = torch.eye(3)[torch.tensor([0, 1, 2, 1, 0])]
        assert torch.allclose(
            bias_error_l2(z_a, z_b, logits=True),
            bias_error_l2(z_a, z_b, y=y, logits=True),
        )

    def test_logits_combines_with_regularity_ce(self):
        # The failure this fix addresses: SONN.get_error under
        # cmpComb_validate_bias broadcast a (E, N) bias against an (E,)
        # regularity and raised.
        logits_a = torch.randn(4, 7, 3)
        logits_b = torch.randn(4, 7, 3)
        y = torch.tensor([0, 1, 2, 1, 0, 2, 1])
        bias = bias_error_l2(logits_a, logits_b, logits=True)
        reg = regularity_error_ce(logits_a, y)
        assert (0.5 * bias + 0.5 * reg).shape == (4,)


class TestBiasErrorJS:
    def test_identical_logits_zero(self):
        logits = torch.randn(3, 5, 4)  # 3 candidates, 5 samples, 4 classes
        out = bias_error_js(logits, logits)
        assert torch.allclose(out, torch.zeros(3), atol=1e-6)

    def test_value_in_unit_interval(self):
        a = torch.randn(2, 6, 3)
        b = torch.randn(2, 6, 3)
        out = bias_error_js(a, b)
        assert (out >= 0).all()
        assert (out <= 1 + 1e-5).all()

    def test_max_at_disjoint_supports(self):
        # one-hot logits at disjoint classes should approach the upper bound
        N, K = 5, 3
        a = torch.full((N, K), -1e3)
        b = torch.full((N, K), -1e3)
        for i in range(N):
            a[i, 0] = 1e3
            b[i, 1] = 1e3
        out = bias_error_js(a, b)
        # JS at fully disjoint one-hots = ln 2 per sample, divided by ln 2 → 1
        assert math.isclose(out.item(), 1.0, abs_tol=1e-3)
