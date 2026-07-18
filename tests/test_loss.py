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


class TestNormMSE:
    def test_zero_loss_on_exact_match(self):
        loss = NormMSE()
        y = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(loss(y, y), torch.tensor(0.0))

    def test_value(self):
        loss = NormMSE(eps=0.0)
        y_true = torch.tensor([1.0, 2.0])
        y_pred = torch.tensor([0.0, 0.0])
        # ||y - ŷ||^2 / ||y||^2 = 5/5 = 1
        assert torch.allclose(loss(y_pred, y_true), torch.tensor(1.0))


class TestRegularityError:
    def test_perfect_prediction_is_zero(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        out = regularity_error(y, y)
        assert torch.allclose(out, torch.tensor(0.0))

    def test_with_candidate_batch(self):
        # 2 candidates, 4 samples
        y_hat = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = regularity_error(y_hat, y)
        assert out.shape == (2,)
        assert torch.allclose(out[0], torch.tensor(0.0))
        # zero predictor: sum((y-0)^2)/sum(y^2) = 1
        assert torch.allclose(out[1], torch.tensor(1.0))


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
