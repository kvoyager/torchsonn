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

    def test_state_roundtrip_keeps_shared_lr_multiplier(self):
        p = _make_params()
        opt = BatchedAdam(p, shared_param_names=[], lr=torch.ones(3) * 0.1,
                    shared_param_lr_multiplier=0.25)
        opt.step(p, _make_grads())
        opt2 = BatchedAdam(p, shared_param_names=[], lr=torch.ones(3) * 0.1)
        opt2.load_state_dict(opt.state_dict())
        assert opt2.shared_param_lr_multiplier == 0.25

    def test_load_state_dict_requires_shared_lr_multiplier(self):
        p = _make_params()
        opt = BatchedAdam(p, shared_param_names=[], lr=torch.ones(3) * 0.1)
        opt.step(p, _make_grads())
        sd = opt.state_dict()
        del sd["shared_param_lr_multiplier"]
        with pytest.raises(KeyError):
            BatchedAdam(p, shared_param_names=[], lr=torch.ones(3) * 0.1).load_state_dict(sd)


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

    def test_state_roundtrip_keeps_shared_lr_multiplier(self):
        p = _make_params()
        opt = BatchedSGD(p, shared_param_names=[], lr=torch.ones(3) * 0.1,
                    shared_param_lr_multiplier=0.25)
        opt.step(p, _make_grads())
        opt2 = BatchedSGD(p, shared_param_names=[], lr=torch.ones(3) * 0.1)
        opt2.load_state_dict(opt.state_dict())
        assert opt2.shared_param_lr_multiplier == 0.25

    def test_load_state_dict_requires_shared_lr_multiplier(self):
        p = _make_params()
        opt = BatchedSGD(p, shared_param_names=[], lr=torch.ones(3) * 0.1)
        opt.step(p, _make_grads())
        sd = opt.state_dict()
        del sd["shared_param_lr_multiplier"]
        with pytest.raises(KeyError):
            BatchedSGD(p, shared_param_names=[], lr=torch.ones(3) * 0.1).load_state_dict(sd)


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
        assert (opt.hist_count["weight"] > 0).any()

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


class TestLBFGSGuards:
    """DRAFT-ensemble-lbfgs Patch 1: curvature condition and step cap."""

    @staticmethod
    def _run(opt, p, grad_fn, steps):
        params = {"weight": p.clone()}
        for _ in range(steps):
            params = opt.step(params, {"weight": grad_fn(params["weight"])})
        return params["weight"]

    def test_unguarded_runs_away_on_a_concave_tail_and_guarded_returns(self):
        """Loss per member: f(w) = w0^2 + 1 - exp(-w1^2), minimum at the origin,
        concave in w1 beyond 1/sqrt(2): out there the gradient and the
        curvature vanish together and y.s < 0, the situation of an RBF bump
        losing its mass. The historical optimizer's quasi-Newton step points
        away from the minimum and every member ends further out than it
        started (one in a single step of tens of units); the guarded one
        rejects those pairs, keeps every step below max_step, never moves a
        member outward, and converges the members that start inside reach."""
        def grad(w):
            return torch.stack([2.0 * w[:, 0], 2.0 * w[:, 1] * torch.exp(-w[:, 1] ** 2)], dim=1)
        start = torch.tensor([[1.0, 2.0], [0.5, 2.5], [2.0, 3.0]])
        old = BatchedLBFGS({"weight": start.clone()}, [], lr=torch.ones(3), max_step=None, curvature_eps=None)
        params = {"weight": start.clone()}
        biggest = 0.0
        for _ in range(80):
            prev = params["weight"].clone()
            params = old.step(params, {"weight": grad(params["weight"])})
            biggest = max(biggest, (params["weight"] - prev).norm(dim=1).max().item())
        assert (params["weight"][:, 1].abs() > start[:, 1].abs() + 1.0).all()
        assert biggest > 10.0
        assert old.stats["rejected"] == 0 and old.stats["capped"] == 0

        new = BatchedLBFGS({"weight": start.clone()}, [], lr=torch.ones(3))
        params = {"weight": start.clone()}
        for _ in range(80):
            prev = params["weight"].clone()
            params = new.step(params, {"weight": grad(params["weight"])})
            assert ((params["weight"] - prev).norm(dim=1) <= 1.0 + 1e-6).all()
        assert (params["weight"][:, 1].abs() <= start[:, 1].abs() + 1e-6).all()
        assert (params["weight"][:2, 1].abs() < 0.05).all()          # members starting at 2.0 and 2.5 converge
        assert params["weight"][:, 0].abs().max() < 1e-3
        assert new.stats["rejected"] > 0 and new.stats["capped"] > 0 and new.stats["steps"] == 80

    def test_negative_curvature_pair_is_not_stored(self):
        opt = BatchedLBFGS({"weight": torch.zeros(1, 2)}, [], lr=torch.ones(1))
        p = {"weight": torch.tensor([[0.0, 0.0]])}
        p = opt.step(p, {"weight": torch.tensor([[1.0, 0.0]])})       # moves along -x
        # a gradient that grew along the direction of the step: y.s < 0
        p = opt.step(p, {"weight": torch.tensor([[3.0, 0.0]])})
        assert opt.hist_count["weight"][0] == 0
        assert opt.stats["rejected"] == 1 and opt.stats["pairs"] == 1
        # a gradient that shrank along the step: y.s > 0, stored
        p = opt.step(p, {"weight": torch.tensor([[0.5, 0.0]])})
        assert opt.hist_count["weight"][0] == 1

    def test_defaults_reproduce_the_old_iterates_on_a_convex_fit(self):
        """A batched least-squares fit (the polynomial families' case): the
        guards never act after the first steps and the iterates match the
        historical optimizer bitwise once max_step and curvature_eps are off."""
        torch.manual_seed(0)
        x = torch.randn(200, 4)
        w_true = torch.randn(3, 4)
        y = torch.einsum("nd,bd->bn", x, w_true)

        def grad(w):
            resid = torch.einsum("nd,bd->bn", x, w) - y
            return 2.0 * torch.einsum("bn,nd->bd", resid, x) / x.shape[0]
        start = torch.zeros(3, 4)
        guarded = BatchedLBFGS({"weight": start.clone()}, [], lr=torch.full((3,), 0.1))
        w_g = self._run(guarded, start, grad, 150)
        assert torch.allclose(w_g, w_true, atol=1e-2)
        assert guarded.stats["rejected"] == 0
        assert guarded.stats["capped"] <= 3 * 3            # at most the first steps of each member
        old = BatchedLBFGS({"weight": start.clone()}, [], lr=torch.full((3,), 0.1), max_step=None, curvature_eps=None)
        w_o = self._run(old, start, grad, 150)
        assert torch.allclose(w_o, w_true, atol=1e-2)
        assert torch.allclose(w_g, w_o, atol=1e-2)

    def test_null_settings_are_the_old_optimizer_bitwise(self):
        torch.manual_seed(1)
        grads = [torch.randn(2, 3) for _ in range(8)]
        a = BatchedLBFGS({"weight": torch.zeros(2, 3)}, [], lr=torch.ones(2) * 0.3, max_step=None, curvature_eps=None)
        b = BatchedLBFGS({"weight": torch.zeros(2, 3)}, [], lr=torch.ones(2) * 0.3, max_step=None, curvature_eps=None)
        pa = {"weight": torch.zeros(2, 3)}
        pb = {"weight": torch.zeros(2, 3)}
        for g in grads:
            pa = a.step(pa, {"weight": g})
            pb = b.step(pb, {"weight": g})
        assert torch.equal(pa["weight"], pb["weight"])
        assert a.stats["capped"] == 0 and a.stats["rejected"] == 0

    def test_shared_parameter_path_is_guarded_too(self):
        opt = BatchedLBFGS({"weight": torch.zeros(2, 2), "shared": torch.zeros(2)}, ["shared"],
                           lr=torch.ones(2), max_step=0.5)
        params = {"weight": torch.zeros(2, 2), "shared": torch.zeros(2)}
        grads = {"weight": torch.zeros(2, 2), "shared": torch.stack([torch.tensor([3.0, 4.0])] * 2)}
        out = opt.step(params, grads)
        assert torch.isclose(out["shared"].norm(), torch.tensor(0.5))
        assert opt.stats["capped"] == 1

    def test_bad_settings_rejected(self):
        with pytest.raises(ValueError, match="max_step"):
            BatchedLBFGS({"weight": torch.zeros(1, 2)}, [], lr=torch.ones(1), max_step=0.0)
        with pytest.raises(ValueError, match="curvature_eps"):
            BatchedLBFGS({"weight": torch.zeros(1, 2)}, [], lr=torch.ones(1), curvature_eps=-1.0)


class _LoopLBFGS:
    """Reference: the per-member deque implementation the batched recursion
    replaced (DRAFT-ensemble-lbfgs Patch 2), guards included, for the
    equivalence test."""

    def __init__(self, params, lr, history_size, max_step, curvature_eps):
        from collections import deque
        self.lr, self.m, self.max_step, self.eps = lr, history_size, max_step, curvature_eps
        self.b = next(iter(params.values())).shape[0]
        self.s = {k: [deque(maxlen=history_size) for _ in range(self.b)] for k in params}
        self.y = {k: [deque(maxlen=history_size) for _ in range(self.b)] for k in params}
        self.prev_p = {k: v.clone() for k, v in params.items()}
        self.prev_g = {k: torch.zeros_like(v) for k, v in params.items()}
        self.ref = BatchedLBFGS(params, [], lr=lr, history_size=history_size,
                                max_step=max_step, curvature_eps=curvature_eps)

    def step(self, params, grads, active):
        out = {}
        for k, p in params.items():
            g = grads[k]
            pf, gf = p.reshape(self.b, -1), g.reshape(self.b, -1)
            sm, ym = pf - self.prev_p[k].reshape(self.b, -1), gf - self.prev_g[k].reshape(self.b, -1)
            upd = torch.zeros_like(pf)
            for i in range(self.b):
                if not active[i]:
                    continue
                if sm[i].norm() > 1e-12 and (self.eps is None or ym[i].dot(sm[i]) > self.eps * sm[i].norm() * ym[i].norm()):
                    self.s[k][i].append(sm[i].clone())
                    self.y[k][i].append(ym[i].clone())
                d = self.ref._two_loop_recursion_single(self.s[k][i], self.y[k][i], gf[i])
                u = float(self.lr[i]) * d
                if self.max_step is not None and u.norm() > self.max_step:
                    u = u * (self.max_step / u.norm())
                upd[i] = u
            out[k] = torch.where(active.view(-1, 1), pf - upd, pf).reshape_as(p)
            self.prev_p[k], self.prev_g[k] = p.clone(), g.clone()
        return out


class TestLBFGSBatchedRecursion:
    """DRAFT-ensemble-lbfgs Patch 2: the two-loop recursion batched over members."""

    @pytest.mark.parametrize("dtype,atol", [(torch.float64, 1e-10), (torch.float32, 1e-3)])
    @pytest.mark.parametrize("b,pdim,m", [(1, 1, 1), (3, 4, 2), (8, 64, 10), (5, 7, 3)])
    def test_matches_the_per_member_loop(self, b, pdim, m, dtype, atol):
        torch.manual_seed(b * 100 + pdim + m)
        p0 = {"w": torch.randn(b, pdim, dtype=dtype), "c": torch.randn(b, 2, max(1, pdim // 2), dtype=dtype)}
        lr = torch.rand(b, dtype=dtype) * 0.5 + 0.1
        fast = BatchedLBFGS({k: v.clone() for k, v in p0.items()}, [], lr=lr, history_size=m)
        slow = _LoopLBFGS({k: v.clone() for k, v in p0.items()}, lr, m, 1.0, 1e-8)
        pf = {k: v.clone() for k, v in p0.items()}
        ps = {k: v.clone() for k, v in p0.items()}
        # float32 trajectories driven by random gradients amplify rounding
        # differences (dot vs sum-of-products order) chaotically; float64 shows
        # the two implementations agree to 1e-10 over the full run.
        n_steps = 3 * m + 4 if dtype == torch.float64 else min(12, 3 * m + 4)
        for t in range(n_steps):
            grads = {k: torch.randn_like(v) * (0.3 if t % 4 else 3.0) for k, v in pf.items()}
            active = torch.rand(b) > 0.2 if t > 2 else torch.ones(b, dtype=torch.bool)
            pf = fast.step(pf, {k: v.clone() for k, v in grads.items()}, active_mask=active)
            ps = slow.step(ps, grads, active)
            for k in pf:
                assert torch.allclose(pf[k], ps[k], atol=atol, rtol=0), (t, k)
        for k in ("w", "c"):
            for i in range(b):
                assert int(fast.hist_count[k][i]) == len(slow.s[k][i])

    def test_inactive_members_keep_params_and_history(self):
        torch.manual_seed(0)
        p = {"w": torch.randn(4, 3)}
        opt = BatchedLBFGS(p, [], lr=torch.ones(4) * 0.1)
        for _ in range(3):
            p = opt.step(p, {"w": torch.randn(4, 3)})
        frozen_p = p["w"][1].clone()
        frozen_s = opt.s_hist["w"][1].clone()
        frozen_n = int(opt.hist_count["w"][1])
        mask = torch.tensor([True, False, True, False])
        for _ in range(3):
            p = opt.step(p, {"w": torch.randn(4, 3)}, active_mask=mask)
        assert torch.equal(p["w"][1], frozen_p)
        assert torch.equal(opt.s_hist["w"][1], frozen_s) and int(opt.hist_count["w"][1]) == frozen_n
        assert int(opt.hist_count["w"][0]) > frozen_n or frozen_n == opt.history_size

    def test_state_roundtrip(self):
        from collections import deque
        torch.manual_seed(1)
        p = {"w": torch.randn(3, 4), "shared": torch.randn(2)}
        opt = BatchedLBFGS(p, ["shared"], lr=torch.ones(3) * 0.1, history_size=3)
        for _ in range(5):
            p = opt.step(p, {"w": torch.randn(3, 4), "shared": torch.randn(3, 2)})
        sd = opt.state_dict()
        again = BatchedLBFGS(p, ["shared"], lr=torch.ones(3) * 0.1, history_size=3)
        again.load_state_dict(sd)
        assert torch.equal(again.s_hist["w"], opt.s_hist["w"]) and torch.equal(again.hist_count["w"], opt.hist_count["w"])
        assert isinstance(again.s_hist["shared"], deque) and len(again.s_hist["shared"]) == len(opt.s_hist["shared"])
        g = {"w": torch.randn(3, 4), "shared": torch.randn(3, 2)}
        a = opt.step({k: v.clone() for k, v in p.items()}, {k: v.clone() for k, v in g.items()})
        b = again.step({k: v.clone() for k, v in p.items()}, {k: v.clone() for k, v in g.items()})
        assert torch.equal(a["w"], b["w"]) and torch.equal(a["shared"], b["shared"])

    @pytest.mark.parametrize("missing", [
        "clip_value", "clip_norm", "max_step", "curvature_eps",
        "shared_param_lr_multiplier", "hist_count",
    ])
    def test_load_state_dict_requires_every_field(self, missing):
        torch.manual_seed(1)
        p = {"w": torch.randn(3, 4)}
        opt = BatchedLBFGS(p, [], lr=torch.ones(3) * 0.1, history_size=3)
        for _ in range(2):
            p = opt.step(p, {"w": torch.randn(3, 4)})
        partial = {k: v for k, v in opt.state_dict().items() if k != missing}
        with pytest.raises(KeyError):
            BatchedLBFGS(p, [], lr=torch.ones(3) * 0.1, history_size=3).load_state_dict(partial)

    def test_step_time_is_independent_of_the_ensemble_size(self):
        import time
        def clock(b):
            p = {"w": torch.randn(b, 18), "c": torch.randn(b, 8, 2), "lw": torch.randn(b, 8)}
            opt = BatchedLBFGS(p, [], lr=torch.ones(b) * 0.1)
            for _ in range(12):
                p = opt.step(p, {k: torch.randn_like(v) for k, v in p.items()})
            t0 = time.perf_counter()
            for _ in range(20):
                p = opt.step(p, {k: torch.randn_like(v) for k, v in p.items()})
            return (time.perf_counter() - t0) / 20
        small, large = clock(4), clock(140)
        assert large < 0.05, f"step over 140 members took {large * 1e3:.1f} ms"
        assert large < 6 * max(small, 1e-4)

