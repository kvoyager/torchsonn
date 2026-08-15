"""Tests for the Trainer class.

Coverage scope is intentionally limited to the unit-testable surface:
- checkpoint plumbing (parse, cleanup, save/load helpers),
- distributed-helpers via mocked `dist.is_initialized`,
- dataloader split utility,
- `LayerAccumulator` / `StepCheckpoint` dataclasses.

End-to-end `train()` runs live in `test_trainer_smoke.py`.

# region uncovered
The following Trainer branches are deliberately uncovered:

1. `torch.distributed` paths (init_distributed, _gather_ensemble_params /
   _allreduce_shared_grads with world_size > 1, sharded ensemble training).
   These require `torchrun --nproc_per_node=N` and cannot be exercised
   inside a single-process pytest without monkey-patching every `dist.*`
   call — which tests the patch, not the production code.

2. Mid-step checkpoint-resume paths inside `train_model_ensemble` (resume
   when `neuron_model_completed=False`). The trainer never writes a
   mid-step checkpoint in tests because `save_interval > steps`.

3. Opt-in features that default to off: `log_layer_diagnostics`,
   `divergence_threshold`-triggered NaN masking, the LBFGS strong-Wolfe
   branch inside `_train_out_proj_lbfgs`.
# endregion
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from torchsonn.config import SONNConfig
from torchsonn.data.dataset import SONNDataset
from torchsonn.loss import regularity_error
from torchsonn.model import SONN
from torchsonn.trainer import LayerAccumulator, StepCheckpoint, Trainer


def _make_cfg(**overrides) -> OmegaConf:
    base = OmegaConf.structured(SONNConfig)
    merged = OmegaConf.merge(base, OmegaConf.create(overrides))
    return merged


def _make_simple_model(tmp_path: Path) -> SONN:
    cfg = _make_cfg(
        model={
            "type": "regressor",
            "num_classes": 1,
            "nbest_neurons": 3,
            "soft_binner": False,
            "ref_functions": ["linear_cov"],
        },
        train={"checkpoint_dir": str(tmp_path)},
    )
    return SONN(cfg, d_model=4)


class TestParseCheckpointStep:
    def test_happy_match(self):
        a, b, c = Trainer.parse_checkpoint_step("model_layer_3_neuron_7_step_42.ckpt")
        assert (a, b, c) == (3, 7, 42)

    def test_happy_with_suffix(self):
        a, b, c = Trainer.parse_checkpoint_step("model_layer_0_neuron_0_step_1_last.ckpt")
        assert (a, b, c) == (0, 0, 1)

    def test_sentinel_on_no_match(self):
        a, b, c = Trainer.parse_checkpoint_step("garbage.txt")
        assert (a, b, c) == (-1, -1, -1)


class TestGetCheckpointDir:
    def test_custom_dir(self, tmp_path):
        cfg = _make_cfg(
            model={
                "type": "regressor",
                "num_classes": 1,
                "nbest_neurons": 3,
                "soft_binner": False,
                "ref_functions": ["linear_cov"],
            },
            train={"checkpoint_dir": str(tmp_path / "ckpts")},
        )
        model = SONN(cfg, d_model=4)
        out = Trainer.get_checkpoint_dir(model)
        assert out == tmp_path / "ckpts"
        assert out.exists()

    def test_empty_dir_defaults_to_repo_path(self, tmp_path):
        cfg = _make_cfg(
            model={
                "type": "regressor",
                "num_classes": 1,
                "nbest_neurons": 3,
                "soft_binner": False,
                "ref_functions": ["linear_cov"],
            },
            train={"checkpoint_dir": ""},
        )
        model = SONN(cfg, d_model=4)
        out = Trainer.get_checkpoint_dir(model)
        # the path resolves to <repo>/checkpoints; just check it ends with "checkpoints"
        assert out.name == "checkpoints"


def _norm_cfg(tmp_path: Path, normalization: str) -> OmegaConf:
    return _make_cfg(
        model={
            "type": "regressor",
            "num_classes": 1,
            "nbest_neurons": 3,
            "soft_binner": False,
            "ref_functions": ["linear_cov"],
        },
        train={"checkpoint_dir": str(tmp_path), "error_normalization": normalization},
    )


# Target whose mean dominates its spread — the regime where the `variance` and
# `energy` denominators diverge, and where float32 accumulation of
# `Σy² - (Σy)²/N` would lose most of its digits.
def _shifted_targets(n: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    y = 100.0 + torch.randn(n)
    preds = torch.stack([y + 0.3 * torch.randn(n), y + 1.5 * torch.randn(n)])
    return y, preds


class TestRegularityErrStreaming:
    """`Trainer.regularity_err` re-implements `loss.regularity_error` for the
    streaming case (the eval set is only ever seen one batch at a time), so the
    two need pinning to each other."""

    @staticmethod
    def _run(normalization: str, batch_size: int, tmp_path: Path) -> torch.Tensor:
        y, preds = _shifted_targets()
        # candidates are encoded as feature columns, so the fake pred_fn below
        # can hand them back per batch without a real neuron ensemble
        x = preds.T.contiguous()

        cfg = _norm_cfg(tmp_path, normalization)
        model = SONN(cfg, d_model=2)
        trainer = Trainer(cfg)
        dl = DataLoader(SONNDataset(x, y), batch_size=batch_size)

        def pred_fn(params, buffers, xb, yb):
            return xb.T                                  # (ensemble=2, batch)

        return trainer.regularity_err(
            model, pred_fn, {}, {}, dl, "cpu", skip_model_fwd=True,
        )

    @pytest.mark.parametrize("normalization", ["variance", "energy"])
    def test_matches_reference_helper(self, normalization, tmp_path):
        y, preds = _shifted_targets()
        out = self._run(normalization, batch_size=5, tmp_path=tmp_path)
        expected = regularity_error(preds, y, centered=normalization == "variance")
        assert torch.allclose(out, expected, rtol=1e-5)

    def test_batching_does_not_change_result(self, tmp_path):
        whole = self._run("variance", batch_size=12, tmp_path=tmp_path)
        chunked = self._run("variance", batch_size=5, tmp_path=tmp_path)
        assert torch.allclose(whole, chunked, rtol=1e-6)

    def test_variance_and_energy_differ_on_shifted_targets(self, tmp_path):
        # Guards against the two branches silently collapsing into one.
        variance = self._run("variance", batch_size=5, tmp_path=tmp_path)
        energy = self._run("energy", batch_size=5, tmp_path=tmp_path)
        assert (energy < variance / 100).all()
        # ...but the candidate ranking is identical either way
        assert torch.equal(variance.argsort(), energy.argsort())


class TestFitTargetScale:
    def test_sets_training_variance(self, tmp_path):
        y, _ = _shifted_targets()
        x = torch.randn(y.numel(), 2)
        model = SONN(_norm_cfg(tmp_path, "variance"), d_model=2)
        trainer = Trainer(_norm_cfg(tmp_path, "variance"))

        trainer._fit_target_scale(model, DataLoader(SONNDataset(x, y), batch_size=5))
        assert model.loss_fn.scale == pytest.approx(y.var(unbiased=False).item(), rel=1e-5)

    def test_energy_uses_mean_square(self, tmp_path):
        y, _ = _shifted_targets()
        x = torch.randn(y.numel(), 2)
        model = SONN(_norm_cfg(tmp_path, "energy"), d_model=2)
        trainer = Trainer(_norm_cfg(tmp_path, "energy"))

        trainer._fit_target_scale(model, DataLoader(SONNDataset(x, y), batch_size=5))
        assert model.loss_fn.scale == pytest.approx((y * y).mean().item(), rel=1e-5)

    def test_constant_target_leaves_scale_unset(self, tmp_path):
        y = torch.full((8,), 7.0)
        x = torch.randn(8, 2)
        model = SONN(_norm_cfg(tmp_path, "variance"), d_model=2)
        trainer = Trainer(_norm_cfg(tmp_path, "variance"))

        trainer._fit_target_scale(model, DataLoader(SONNDataset(x, y), batch_size=4))
        # falls back to the per-call denominator rather than dividing by ~0
        assert model.loss_fn.scale is None

    def test_noop_for_classifier(self, tmp_path):
        cfg = _make_cfg(
            model={"type": "multi-class", "num_classes": 3, "nbest_neurons": 3,
                   "ref_functions": ["linear_cov"]},
            train={"checkpoint_dir": str(tmp_path)},
        )
        model = SONN(cfg, d_model=2)
        trainer = Trainer(cfg)
        x, y = torch.randn(8, 2), torch.randint(0, 3, (8,))
        # NLLLoss has no `scale`; the helper must not touch it
        trainer._fit_target_scale(model, DataLoader(SONNDataset(x, y), batch_size=4))
        assert not hasattr(model.loss_fn, "scale")


class TestCleanupCheckpoints:
    def test_keeps_last_n(self, tmp_path):
        # create five fake checkpoints with increasing step counters
        for step in range(5):
            (tmp_path / f"model_layer_0_neuron_0_step_{step}.ckpt").write_text("x")
        # an unrelated _last file should be preserved
        (tmp_path / "model_last.ckpt").write_text("x")

        trainer = Trainer(config=None)
        trainer.cleanup_checkpoints(tmp_path, keep_last_n=2)

        remaining = sorted(p.name for p in tmp_path.iterdir())
        assert "model_last.ckpt" in remaining
        # only the two highest-step files survive
        assert "model_layer_0_neuron_0_step_3.ckpt" in remaining
        assert "model_layer_0_neuron_0_step_4.ckpt" in remaining
        assert "model_layer_0_neuron_0_step_0.ckpt" not in remaining

    def test_cleanup_layer(self, tmp_path):
        cfg = _make_cfg(
            model={
                "type": "regressor",
                "num_classes": 1,
                "nbest_neurons": 3,
                "soft_binner": False,
                "ref_functions": ["linear_cov"],
            },
            train={"checkpoint_dir": str(tmp_path)},
        )
        model = SONN(cfg, d_model=4)
        # spread checkpoints over two layers
        (tmp_path / "model_layer_0_neuron_0_step_0.ckpt").write_text("x")
        (tmp_path / "model_layer_0_neuron_1_step_3.ckpt").write_text("x")
        (tmp_path / "model_layer_1_neuron_0_step_0.ckpt").write_text("x")

        trainer = Trainer(config=None)
        trainer.cleanup_layer_checkpoints(model, layer_idx=0)

        names = sorted(p.name for p in tmp_path.iterdir())
        assert "model_layer_1_neuron_0_step_0.ckpt" in names
        # layer 0 files should be gone
        assert all("layer_0" not in n for n in names)


class TestFromCheckpoint:
    def test_no_checkpoints_returns_none(self, tmp_path):
        cfg = _make_cfg(
            model={
                "type": "regressor",
                "num_classes": 1,
                "nbest_neurons": 3,
                "soft_binner": False,
                "ref_functions": ["linear_cov"],
            },
            train={"checkpoint_dir": str(tmp_path)},
        )
        model = SONN(cfg, d_model=4)
        trainer = Trainer(config=None)
        assert trainer.from_checkpoint(model) is None

    def test_picks_latest(self, tmp_path):
        cfg = _make_cfg(
            model={
                "type": "regressor",
                "num_classes": 1,
                "nbest_neurons": 3,
                "soft_binner": False,
                "ref_functions": ["linear_cov"],
            },
            train={"checkpoint_dir": str(tmp_path)},
        )
        model = SONN(cfg, d_model=4)
        torch.save({"marker": 1}, tmp_path / "model_layer_0_neuron_0_step_5.ckpt")
        torch.save({"marker": 99}, tmp_path / "model_layer_0_neuron_0_step_99.ckpt")
        # spurious _last file should be ignored
        torch.save({"marker": -1}, tmp_path / "model_last.ckpt")

        trainer = Trainer(config=None)
        data = trainer.from_checkpoint(model)
        assert data["marker"] == 99


class TestSaveLoadModelCheckpoint:
    def test_roundtrip(self, tmp_path):
        cfg = _make_cfg(
            model={
                "type": "regressor",
                "num_classes": 1,
                "nbest_neurons": 3,
                "soft_binner": False,
                "ref_functions": ["linear_cov"],
            },
            train={"checkpoint_dir": str(tmp_path)},
        )
        model = SONN(cfg, d_model=4)
        model.layers.append(model.create_layer(0))

        trainer = Trainer(config=cfg)
        trainer.save_model_checkpoint(model)

        # Build a fresh model and reload state
        model2 = SONN(cfg, d_model=4)
        trainer.load_model_checkpoint(model2)
        assert len(model2.layers) == 1


class TestSplitLoader:
    def test_independent_split_state(self):
        x = np.arange(20).reshape(10, 2).astype("float32")
        ds = SONNDataset(x, np.arange(10).astype("float32"))
        dl = DataLoader(ds, batch_size=2)

        a = Trainer._split_loader(dl, 0)
        b = Trainer._split_loader(dl, 1)
        # different SONNDataset instances → not aliased
        assert a.dataset is not b.dataset
        # split flag faithfully recorded
        assert a.dataset.split == 0
        assert b.dataset.split == 1


class TestDistributedHelpers:
    def test_is_dist_false_when_not_initialized(self, monkeypatch):
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
        assert Trainer._is_dist() is False

    def test_ensemble_slice_none_when_not_initialized(self, monkeypatch):
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
        assert Trainer._ensemble_slice(16) is None

    def test_ensemble_slice_world_size_1(self, monkeypatch):
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
        monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)
        assert Trainer._ensemble_slice(16) is None

    def test_ensemble_slice_non_divisible(self, monkeypatch):
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
        monkeypatch.setattr("torch.distributed.get_world_size", lambda: 4)
        monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
        assert Trainer._ensemble_slice(10) is None

    def test_ensemble_slice_divisible(self, monkeypatch):
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
        monkeypatch.setattr("torch.distributed.get_world_size", lambda: 4)
        monkeypatch.setattr("torch.distributed.get_rank", lambda: 1)
        s = Trainer._ensemble_slice(16)
        assert s == (4, 8)

    def test_gather_passthrough_when_world_size_1(self, monkeypatch):
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
        p = {"w": torch.zeros(3, 2)}
        assert Trainer._gather_ensemble_params(p, []) is p

    def test_allreduce_noop_when_not_distributed(self, monkeypatch):
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
        g = {"w": torch.zeros(3)}
        Trainer._allreduce_shared_grads(g, ["w"])  # should not raise


class TestSetSeed:
    def test_runs_without_error(self):
        Trainer.set_seed(123)
        x = torch.rand(3)
        # determinism: same seed → same sample
        Trainer.set_seed(123)
        y = torch.rand(3)
        assert torch.allclose(x, y)


class TestStepCheckpoint:
    def test_to_dict_handles_state_dict_fields(self, tmp_path):
        cfg = _make_cfg(
            model={
                "type": "regressor",
                "num_classes": 1,
                "nbest_neurons": 3,
                "soft_binner": False,
                "ref_functions": ["linear_cov"],
            },
            train={"checkpoint_dir": str(tmp_path)},
        )
        model = SONN(cfg, d_model=4)

        class FakeOpt:
            def state_dict(self):
                return {"v": 1}

        ckpt = StepCheckpoint(
            model=model,
            opt=FakeOpt(),
            layer_idx=0,
            neuron_model_idx=0,
            epoch=0,
            step=0,
            global_step=0,
            best_val_losses=torch.zeros(2),
            smoothed_val_losses=torch.zeros(2),
            last_steps_improve=torch.zeros(2, dtype=torch.int),
            early_stop_flags=torch.zeros(2, dtype=torch.bool),
            lr=torch.zeros(2),
            neuron_model_completed=False,
            layer_completed=False,
            err=[],
            module_idxs=[],
            scheduler=None,
        )
        d = ckpt.to_dict()
        # FakeOpt's state_dict was unwrapped via the safe_value branch
        assert d["opt"] == {"v": 1}
        # `scheduler=None` is filtered out
        assert "scheduler" not in d


class TestLayerAccumulator:
    def test_defaults(self):
        a = LayerAccumulator()
        assert a.err == []
        assert a.module_idxs == []
        assert a.layer_completed is False
