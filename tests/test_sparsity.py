"""Hard-concrete gates and the group-lasso penalty (torchsonn.modules.sparsity)."""
import math

import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from torchsonn.config import SONNConfig
from torchsonn.modules import (HardConcreteGate, diversity_loss, group_lasso_penalty, group_norms,
                               group_soft_threshold)
from torchsonn.trainer import Trainer


class TestHardConcreteGate:
    def test_starts_nearly_open(self):
        gate = HardConcreteGate(5)
        assert torch.allclose(gate.log_alpha, torch.full((5,), math.log(9.0)))
        # clamp(0.9 * 1.2 - 0.1, 0, 1) = 0.98
        assert torch.allclose(gate.deterministic(), torch.full((5,), 0.98))
        assert gate.hard_gate().all()
        assert 0.9 < gate.expected_l0().item() / 5 < 1.0

    def test_eval_mode_is_the_deterministic_gate(self):
        gate = HardConcreteGate(4).eval()
        assert torch.equal(gate(), gate.deterministic())

    def test_samples_repeat_with_a_seeded_generator_and_lie_in_the_unit_interval(self):
        gate = HardConcreteGate(1000, init_open=0.5)
        a = gate(torch.Generator().manual_seed(3))
        b = gate(torch.Generator().manual_seed(3))
        c = gate(torch.Generator().manual_seed(4))
        assert torch.equal(a, b)
        assert not torch.equal(a, c)
        assert a.min() >= 0.0 and a.max() <= 1.0
        # The stretch puts probability mass on exactly 0 and exactly 1.
        assert (a == 0.0).any() and (a == 1.0).any()

    def test_expected_l0_matches_the_sampled_rate_of_nonzero_gates(self):
        gate = HardConcreteGate(20000, init_open=0.3)
        z = gate(torch.Generator().manual_seed(0))
        rate = (z > 0).float().mean().item()
        assert rate == pytest.approx(gate.expected_l0().item() / 20000, abs=0.01)
        assert torch.allclose(gate.prob_on().sum(), gate.expected_l0())

    def test_penalty_gradient_closes_gates(self):
        gate = HardConcreteGate(3)
        gate.expected_l0().backward()
        assert (gate.log_alpha.grad > 0).all()
        with torch.no_grad():
            gate.log_alpha.fill_(-10.0)
        assert torch.equal(gate.deterministic(), torch.zeros(3))
        assert not gate.hard_gate().any()
        assert gate.expected_l0().item() < 1e-3

    def test_hard_gate_threshold(self):
        gate = HardConcreteGate(3)
        with torch.no_grad():
            # deterministic gates 0, ~0.04, 0.98
            gate.log_alpha.copy_(torch.tensor([-5.0, math.log(0.12 / 0.88), math.log(9.0)]))
        z = gate.deterministic()
        assert z[0] == 0.0 and 0.0 < z[1] < 0.05
        assert gate.hard_gate(0.05).tolist() == [False, False, True]
        assert gate.hard_gate(0.01).tolist() == [False, True, True]

    @pytest.mark.parametrize("kwargs", [dict(gamma=0.1), dict(zeta=0.9), dict(beta=0.0),
                                        dict(init_open=1.0), dict(init_open=0.0)])
    def test_rejects_bad_shape_parameters(self, kwargs):
        with pytest.raises(ValueError):
            HardConcreteGate(2, **kwargs)


class TestGroupLasso:
    def test_norms_span_every_part(self):
        w = torch.tensor([[3.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        h = torch.tensor([[4.0], [0.0], [1.0]])
        n = group_norms([w, h])
        assert torch.allclose(n, torch.tensor([5.0, 0.0, math.sqrt(3.0)]), atol=1e-6)
        assert group_lasso_penalty([w, h]).item() == pytest.approx(5.0 + math.sqrt(3.0), abs=1e-6)

    def test_penalty_gradient_is_finite_at_a_zero_group(self):
        w = torch.tensor([[0.0, 0.0], [1.0, 2.0]], requires_grad=True)
        group_lasso_penalty([w]).backward()
        assert torch.isfinite(w.grad).all()
        assert torch.equal(w.grad[0], torch.zeros(2))

    def test_soft_threshold_zeroes_small_groups_and_shrinks_the_rest(self):
        w = torch.tensor([[3.0, 0.0], [0.1, 0.0], [0.0, 2.0]])
        h = torch.tensor([[4.0], [0.0], [0.0]])
        after = group_soft_threshold([w, h], 1.0)
        assert torch.equal(w[1], torch.zeros(2))
        # group 0: norm 5 -> 4, direction kept
        assert torch.allclose(w[0], torch.tensor([2.4, 0.0])) and torch.allclose(h[0], torch.tensor([3.2]))
        # group 2: norm 2 -> 1
        assert torch.allclose(w[2], torch.tensor([0.0, 1.0]))
        assert torch.allclose(after, torch.tensor([4.0, 0.0, 1.0]))

    def test_soft_threshold_writes_through_a_transposed_head(self):
        head = torch.tensor([[1.0, 0.01, 2.0]])          # (C=1, k=3): one column per neuron
        w = torch.tensor([[0.0], [0.0], [0.0]])
        group_soft_threshold([w, head.T], 0.5)
        assert torch.allclose(head, torch.tensor([[0.5, 0.0, 1.5]]))


def _cfg(**joint):
    return OmegaConf.merge(OmegaConf.structured(SONNConfig), {"train": {"joint_sparsity": joint}})


class TestDiversityLoss:
    def test_copies_score_one_and_independent_columns_near_zero(self):
        torch.manual_seed(0)
        a = torch.randn(4000, 1)
        assert torch.isclose(diversity_loss(torch.cat([a, 3 * a + 1], 1), torch.ones(2)),
                             torch.tensor(1.0), atol=1e-3)
        assert diversity_loss(torch.randn(4000, 4), torch.ones(4)) < 1e-2

    def test_a_closed_gate_drops_its_pairs(self):
        torch.manual_seed(0)
        a, b = torch.randn(500, 1), torch.randn(500, 1)
        h = torch.cat([a, a, b], 1)
        # With the copy's gate closed only the near-independent pair (a, b) is left.
        assert diversity_loss(h, torch.tensor([1.0, 0.0, 1.0])) < 0.05
        assert diversity_loss(h, torch.tensor([1.0, 1.0, 1.0])) > 0.3

    def test_tau_tolerates_mild_correlation(self):
        torch.manual_seed(0)
        a = torch.randn(4000, 1)
        h = torch.cat([a, a + 2.0 * torch.randn(4000, 1)], 1)  # |corr| about 0.45
        assert diversity_loss(h, torch.ones(2)) > 0.1
        assert diversity_loss(h, torch.ones(2), tau=0.6) == 0.0

    def test_the_gates_get_a_gradient_that_closes_a_copy(self):
        torch.manual_seed(0)
        a, b = torch.randn(500, 1), torch.randn(500, 1)
        g = torch.ones(3, requires_grad=True)
        diversity_loss(torch.cat([a, a, b], 1), g).backward()
        assert g.grad[0] > 0 and g.grad[1] > 0 and g.grad[2] < g.grad[0]


class TestJointSparsityConfig:
    def test_defaults(self):
        js = OmegaConf.structured(SONNConfig).train.joint_sparsity
        assert (js.kind, js.beta, js.gamma, js.zeta, js.init_open) == ("gates", 0.66, -0.1, 1.1, 0.9)
        Trainer(config=_cfg())

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ConfigKeyError):
            _cfg(lambda_=1.0)

    @pytest.mark.parametrize("joint", [dict(kind="l1"), dict(gamma=0.0), dict(zeta=1.0),
                                       dict(beta=-1.0), dict(init_open=1.5)])
    def test_trainer_rejects_bad_settings(self, joint):
        with pytest.raises(ValueError, match="joint_sparsity"):
            Trainer(config=_cfg(**joint))

    def test_group_lasso_is_accepted(self):
        Trainer(config=_cfg(kind="group_lasso"))
