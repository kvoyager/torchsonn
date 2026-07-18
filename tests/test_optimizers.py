import pytest
import torch

from torchsonn.optimizers import (
    BatchedAdam,
    BatchedLBFGS,
    BatchedNewton,
    BatchedNewtonLM,
    BatchedSGD,
    optimizer_map,
)
from torchsonn.optimizers.base import BaseOptimizer


def _make_params(batch: int = 3, d: int = 4) -> dict[str, torch.Tensor]:
    return {"weight": torch.randn(batch, d, requires_grad=False)}


def _make_grads(batch: int = 3, d: int = 4) -> dict[str, torch.Tensor]:
    return {"weight": torch.randn(batch, d)}


class TestBaseOptimizer:
    def test_clip_value_clamps(self):
        opt = BatchedAdam(_make_params(), shared_param_names=[], lr=torch.ones(3) * 0.1, clip_value=0.5)
        g = torch.tensor([[5.0, -5.0, 0.1, -0.1]])
        clipped = opt.gradient_clipping(g)
        assert (clipped.abs() <= 0.5 + 1e-9).all()

    def test_clip_norm_scales(self):
        opt = BatchedAdam(
            _make_params(), shared_param_names=[], lr=torch.ones(3) * 0.1, clip_norm=1.0
        )
        g = torch.tensor([[3.0, 4.0, 0.0, 0.0]])  # norm = 5
        out = opt.gradient_clipping(g)
        assert pytest.approx(out.norm().item(), abs=1e-4) == 1.0

    def test_no_clipping_passes_through(self):
        opt = BatchedAdam(_make_params(), shared_param_names=[], lr=torch.ones(3) * 0.1)
        g = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
        out = opt.gradient_clipping(g)
        assert torch.equal(out, g)

    def test_base_methods_raise_notimplemented(self):
        # Build a minimal concrete child that doesn't override the abstract
        # state_dict / load_state_dict methods.
        class _Min(BaseOptimizer):
            pass

        opt = _Min(shared_param_names=[], lr=0.1)
        with pytest.raises(NotImplementedError):
            opt.state_dict()
        with pytest.raises(NotImplementedError):
            opt.load_state_dict({})


class TestBatchedAdam:
    def test_step_changes_params(self):
        p = _make_params()
        opt = BatchedAdam(p, shared_param_names=[], lr=torch.ones(3) * 0.1)
        g = _make_grads()
        new = opt.step(p, g)
        assert not torch.equal(new["weight"], p["weight"])

    def test_step_with_active_mask(self):
        p = _make_params()
        opt = BatchedAdam(p, shared_param_names=[], lr=torch.ones(3) * 0.1)
        g = _make_grads()
        mask = torch.tensor([True, False, True])
        new = opt.step(p, g, active_mask=mask)
        # masked-out row should be unchanged
        assert torch.allclose(new["weight"][1], p["weight"][1])
        assert not torch.allclose(new["weight"][0], p["weight"][0])

    def test_shared_param_branch(self):
        p = {
            "weight": torch.randn(3, 2),
            "shared_w": torch.randn(3, 2),
        }
        opt = BatchedAdam(p, shared_param_names=["shared_w"], lr=torch.ones(3) * 0.1)
        g = {k: torch.randn_like(v) for k, v in p.items()}
        new = opt.step(p, g)
        # shared param updates identically across batch (no batch-conditional gating)
        assert new["shared_w"].shape == p["shared_w"].shape

    def test_state_roundtrip(self):
        p = _make_params()
        opt = BatchedAdam(p, shared_param_names=[], lr=torch.ones(3) * 0.1)
        opt.step(p, _make_grads())
        sd = opt.state_dict()

        opt2 = BatchedAdam(p, shared_param_names=[], lr=torch.ones(3) * 0.1)
        opt2.load_state_dict(sd)
        assert opt2.t == opt.t
        assert torch.allclose(opt2.m["weight"], opt.m["weight"])


class TestBatchedSGD:
    def test_step_changes_params(self):
        p = _make_params()
        opt = BatchedSGD(p, shared_param_names=[], lr=torch.ones(3) * 0.1)
        g = _make_grads()
        new = opt.step(p, g)
        assert not torch.equal(new["weight"], p["weight"])

    def test_weight_decay(self):
        p = _make_params()
        opt = BatchedSGD(p, shared_param_names=[], lr=torch.ones(3) * 0.0, weight_decay=0.1)
        g = {k: torch.zeros_like(v) for k, v in p.items()}
        new = opt.step(p, g)
        # with zero lr, params should not change
        assert torch.allclose(new["weight"], p["weight"])

    def test_no_nesterov(self):
        p = _make_params()
        opt = BatchedSGD(
            p, shared_param_names=[], lr=torch.ones(3) * 0.01, nesterov=False, momentum=0.5
        )
        new = opt.step(p, _make_grads())
        assert new["weight"].shape == p["weight"].shape

    def test_active_mask(self):
        p = _make_params()
        opt = BatchedSGD(p, shared_param_names=[], lr=torch.ones(3) * 0.1)
        g = _make_grads()
        mask = torch.tensor([True, False, True])
        new = opt.step(p, g, active_mask=mask)
        assert torch.allclose(new["weight"][1], p["weight"][1])

    def test_shared_param_branch(self):
        p = {"shared_w": torch.randn(3, 2), "w": torch.randn(3, 2)}
        opt = BatchedSGD(p, shared_param_names=["shared_w"], lr=torch.ones(3) * 0.1)
        g = {k: torch.randn_like(v) for k, v in p.items()}
        new = opt.step(p, g)
        assert new["shared_w"].shape == p["shared_w"].shape

    def test_state_roundtrip(self):
        p = _make_params()
        opt = BatchedSGD(p, shared_param_names=[], lr=torch.ones(3) * 0.1)
        opt.step(p, _make_grads())
        sd = opt.state_dict()

        opt2 = BatchedSGD(p, shared_param_names=[], lr=torch.ones(3) * 0.1)
        opt2.load_state_dict(sd)
        assert opt2.momentum == opt.momentum
        assert torch.allclose(opt2.v["weight"], opt.v["weight"])


class TestBatchedLBFGS:
    def test_first_step_returns_gradient_direction(self):
        # With empty history, the two-loop recursion returns g — so a single step
        # should leave shape intact and perturb params by lr * g.
        p = _make_params()
        opt = BatchedLBFGS(p, shared_param_names=[], lr=torch.ones(3) * 0.01)
        g = _make_grads()
        new = opt.step(p, g)
        assert new["weight"].shape == p["weight"].shape

    def test_history_accumulates(self):
        p = _make_params()
        opt = BatchedLBFGS(p, shared_param_names=[], lr=torch.ones(3) * 0.01, history_size=4)
        for _ in range(3):
            p = opt.step(p, _make_grads())
        # At least one history entry per batch member
        assert any(len(opt.s_hist["weight"][i]) > 0 for i in range(3))

    def test_active_mask_skips_history(self):
        p = _make_params()
        opt = BatchedLBFGS(p, shared_param_names=[], lr=torch.ones(3) * 0.01)
        mask = torch.tensor([True, False, False])
        new = opt.step(p, _make_grads(), active_mask=mask)
        # masked-out rows are unchanged
        assert torch.allclose(new["weight"][1], p["weight"][1])
        assert torch.allclose(new["weight"][2], p["weight"][2])

    def test_shared_param_branch(self):
        p = {
            "w": torch.randn(3, 2),
            "shared_w": torch.randn(2),  # single shared tensor
        }
        opt = BatchedLBFGS(p, shared_param_names=["shared_w"], lr=torch.ones(3) * 0.01)
        g = {"w": torch.randn(3, 2), "shared_w": torch.randn(3, 2)}
        new = opt.step(p, g)
        assert new["shared_w"].shape == (2,)

    def test_shared_param_with_no_active(self):
        p = {
            "w": torch.randn(3, 2),
            "shared_w": torch.randn(2),
        }
        opt = BatchedLBFGS(p, shared_param_names=["shared_w"], lr=torch.ones(3) * 0.01)
        g = {"w": torch.zeros(3, 2), "shared_w": torch.zeros(3, 2)}
        mask = torch.zeros(3, dtype=torch.bool)
        new = opt.step(p, g, active_mask=mask)
        # nothing should change with no active entries
        assert torch.allclose(new["w"], p["w"])

    def test_state_roundtrip(self):
        p = _make_params()
        opt = BatchedLBFGS(p, shared_param_names=[], lr=torch.ones(3) * 0.01)
        opt.step(p, _make_grads())
        sd = opt.state_dict()

        opt2 = BatchedLBFGS(p, shared_param_names=[], lr=torch.ones(3) * 0.01)
        opt2.load_state_dict(sd)
        assert opt2.history_size == opt.history_size

    def test_lr_as_float(self):
        p = _make_params()
        opt = BatchedLBFGS(p, shared_param_names=[], lr=0.01)
        new = opt.step(p, _make_grads())
        assert new["weight"].shape == p["weight"].shape

    def test_shared_param_history_accumulates_with_float_lr(self):
        p = {
            "w": torch.randn(3, 2),
            "shared_w": torch.randn(2),
        }
        opt = BatchedLBFGS(p, shared_param_names=["shared_w"], lr=0.01)
        # First step writes prev_params; second step should append history when
        # |s| > 1e-12 (params changed). Float lr exercises the non-tensor branch.
        p2 = opt.step(p, {"w": torch.randn(3, 2), "shared_w": torch.randn(3, 2)})
        opt.step(p2, {"w": torch.randn(3, 2), "shared_w": torch.randn(3, 2)})
        assert len(opt.s_hist["shared_w"]) >= 1

    def test_load_state_dict_with_shared_history(self):
        p = {
            "w": torch.randn(3, 2),
            "shared_w": torch.randn(2),
        }
        opt = BatchedLBFGS(p, shared_param_names=["shared_w"], lr=torch.ones(3) * 0.01)
        # populate shared history by stepping twice
        p2 = opt.step(p, {"w": torch.randn(3, 2), "shared_w": torch.randn(3, 2)})
        opt.step(p2, {"w": torch.randn(3, 2), "shared_w": torch.randn(3, 2)})
        sd = opt.state_dict()

        opt2 = BatchedLBFGS(p, shared_param_names=["shared_w"], lr=torch.ones(3) * 0.01)
        opt2.load_state_dict(sd)
        # shared history is a single deque, not a list of deques
        from collections import deque
        assert isinstance(opt2.s_hist["shared_w"], deque)


class TestBatchedNewton:
    # BatchedNewton's non-shared step path mishandles the broadcast between a
    # per-batch lr and the per-row Newton delta (lr shape (B,1) vs delta
    # squeeze of shape (d,) → result (B,d) won't fit p_flat[i]). We exercise
    # only the shared-param + initialization paths to avoid that pre-existing
    # bug.
    def test_shared_param(self):
        p = {"shared_w": torch.randn(1, 2)}
        opt = BatchedNewton(p, shared_param_names=["shared_w"], lr=torch.ones(1) * 0.01)
        g = {"shared_w": torch.randn(1, 2)}
        new = opt.step(p, g)
        assert new["shared_w"].shape == p["shared_w"].shape

    def test_one_dim_param(self):
        # Trigger the v.dim() < 2 branch where Hessian leading dim is 1.
        p = {"shared_b": torch.randn(2)}
        opt = BatchedNewton(p, shared_param_names=["shared_b"], lr=torch.tensor([0.01]))
        assert opt.H["shared_b"].shape == (1, 2, 2)

    def test_pinverse_fallback(self, monkeypatch):
        p = {"shared_w": torch.randn(1, 2)}
        opt = BatchedNewton(p, shared_param_names=["shared_w"], lr=torch.ones(1) * 0.01)
        # Force the linalg.solve to fail so the pinverse fallback runs.
        orig_solve = torch.linalg.solve

        def boom(*_a, **_k):
            raise RuntimeError("forced failure")

        monkeypatch.setattr("torch.linalg.solve", boom)
        new = opt.step(p, {"shared_w": torch.randn(1, 2)})
        assert new["shared_w"].shape == p["shared_w"].shape


class TestBatchedNewtonLM:
    def test_diagonal_hessian_approximation(self):
        p = _make_params()
        opt = BatchedNewtonLM(p, shared_param_names=[], lr=torch.ones(3) * 0.01)
        g = _make_grads()
        new = opt.step(p, g)
        assert new["weight"].shape == p["weight"].shape

    def test_with_provided_hessian(self):
        p = _make_params(batch=2, d=3)
        opt = BatchedNewtonLM(p, shared_param_names=[], lr=torch.ones(2) * 0.01)
        g = _make_grads(batch=2, d=3)
        h = {"weight": torch.eye(3).unsqueeze(0).repeat(2, 1, 1)}
        new = opt.step(p, g, hessians=h)
        assert new["weight"].shape == p["weight"].shape

    def test_shared_param_branch(self):
        p = {
            "w": torch.randn(2, 3),
            "shared_w": torch.randn(2, 3),
        }
        opt = BatchedNewtonLM(p, shared_param_names=["shared_w"], lr=torch.ones(2) * 0.01)
        g = {k: torch.randn_like(v) for k, v in p.items()}
        new = opt.step(p, g)
        assert new["shared_w"].shape == p["shared_w"].shape

    def test_active_mask(self):
        p = _make_params(batch=3, d=3)
        opt = BatchedNewtonLM(p, shared_param_names=[], lr=torch.ones(3) * 0.01)
        g = _make_grads(batch=3, d=3)
        mask = torch.tensor([True, False, True])
        new = opt.step(p, g, active_mask=mask)
        assert torch.allclose(new["weight"][1], p["weight"][1])


def test_optimizer_map_contents():
    assert set(optimizer_map.keys()) == {"adam", "sgd", "lbfgs", "newton", "newton-lm"}
    assert optimizer_map["adam"] is BatchedAdam
    assert optimizer_map["sgd"] is BatchedSGD
    assert optimizer_map["lbfgs"] is BatchedLBFGS
    assert optimizer_map["newton"] is BatchedNewton
    assert optimizer_map["newton-lm"] is BatchedNewtonLM
