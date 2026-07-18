import pytest
import torch

from torchsonn.optimizers import BatchedAdam
from torchsonn.schedulers import WarmupFlatScheduler
from torchsonn.schedulers.base import BaseScheduler


def _make_opt(lr: torch.Tensor | None = None) -> BatchedAdam:
    p = {"weight": torch.zeros(3, 2)}
    if lr is None:
        lr = torch.ones(3) * 0.1
    return BatchedAdam(p, shared_param_names=[], lr=lr)


class TestWarmupFlatScheduler:
    def test_lr_ramps_up(self):
        opt = _make_opt(torch.ones(3) * 1.0)
        sched = WarmupFlatScheduler(opt, warmup_steps=4)
        sched.step()
        # step 1 of 4 → 1/4 scale
        assert torch.allclose(opt.lr, torch.ones(3) * 0.25)
        sched.step()
        assert torch.allclose(opt.lr, torch.ones(3) * 0.5)
        sched.step()
        assert torch.allclose(opt.lr, torch.ones(3) * 0.75)

    def test_lr_flat_after_warmup(self):
        opt = _make_opt(torch.ones(3) * 2.0)
        sched = WarmupFlatScheduler(opt, warmup_steps=2)
        sched.step()
        sched.step()
        sched.step()  # past warmup → scale = 1.0
        assert torch.allclose(opt.lr, torch.ones(3) * 2.0)

    def test_scalar_lr(self):
        # base_lr branch where optimizer.lr is a scalar float
        class FakeOpt:
            lr = 0.5

        sched = WarmupFlatScheduler(FakeOpt(), warmup_steps=2)
        sched.step()
        assert sched.opt.lr == pytest.approx(0.25)

    def test_state_roundtrip(self):
        opt = _make_opt()
        sched = WarmupFlatScheduler(opt, warmup_steps=3)
        sched.step()
        sd = sched.state_dict()

        opt2 = _make_opt()
        sched2 = WarmupFlatScheduler(opt2, warmup_steps=100)  # overwritten by load
        sched2.load_state_dict(sd)
        assert sched2.warmup_steps == 3
        assert sched2.step_num == sched.step_num


class TestBaseScheduler:
    def test_base_methods_raise(self):
        class _Min(BaseScheduler):
            pass

        sched = _Min()
        with pytest.raises(NotImplementedError):
            sched.state_dict()
        with pytest.raises(NotImplementedError):
            sched.load_state_dict({})


def test_scheduler_map_contents():
    from torchsonn.schedulers import scheduler_map

    assert "warmup_flat" in scheduler_map
    assert scheduler_map["warmup_flat"] is WarmupFlatScheduler
