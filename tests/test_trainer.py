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
        out = Trainer(cfg).checkpoint_root
        assert out == tmp_path / "ckpts"

    def test_default_is_checkpoints_in_the_working_directory(self, tmp_path, monkeypatch):
        cfg = _make_cfg(
            model={
                "type": "regressor",
                "num_classes": 1,
                "nbest_neurons": 3,
                "soft_binner": False,
                "ref_functions": ["linear_cov"],
            },
        )
        assert cfg.train.checkpoint_dir == "checkpoints"
        trainer = Trainer(cfg)
        assert trainer.checkpoint_root == Path("checkpoints")
        monkeypatch.chdir(tmp_path)
        trainer.save_model_checkpoint(SONN(cfg, d_model=4))
        assert (tmp_path / "checkpoints" / trainer.run_dir.name / "model_last.ckpt").is_file()

    def test_empty_dir_raises(self):
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
        with pytest.raises(ValueError, match="train.checkpoint_dir is empty"):
            Trainer(cfg)


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

    def test_leaves_everything_else_alone(self, tmp_path):
        for step in range(12):
            (tmp_path / f"model_layer_0_neuron_0_step_{step}.ckpt").write_text("x")
        (tmp_path / "model_layer_0_neuron_0_step_0_last.ckpt").write_text("x")
        (tmp_path / "model_last.ckpt").write_text("x")
        (tmp_path / "best_layer_0_finetune.ckpt").write_text("x")
        (tmp_path / "train.log").write_text("log")
        (tmp_path / "notes.txt").write_text("mine")
        (tmp_path / "2026-08-19-12-57-00").mkdir()

        Trainer(config=None).cleanup_checkpoints(tmp_path, keep_last_n=10)

        remaining = {p.name for p in tmp_path.iterdir()}
        assert {"train.log", "notes.txt", "2026-08-19-12-57-00", "model_last.ckpt",
                "best_layer_0_finetune.ckpt",
                "model_layer_0_neuron_0_step_0_last.ckpt"} <= remaining
        steps = sorted(Trainer.parse_checkpoint_step(n)[2] for n in remaining
                       if n.startswith("model_layer") and not n.endswith("_last.ckpt"))
        assert steps == list(range(2, 12))

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
        trainer = Trainer(config=cfg)
        trainer.save_model_checkpoint(model)  # opens a run folder
        run = trainer.run_dir
        # spread checkpoints over two layers
        (run / "model_layer_0_neuron_0_step_0.ckpt").write_text("x")
        (run / "model_layer_0_neuron_1_step_3_last.ckpt").write_text("x")
        (run / "model_layer_1_neuron_0_step_0.ckpt").write_text("x")
        (run / "train.log").write_text("log")

        trainer.cleanup_layer_checkpoints(layer_idx=0)

        names = sorted(p.name for p in run.iterdir())
        assert "model_layer_1_neuron_0_step_0.ckpt" in names
        assert "train.log" in names and "model_last.ckpt" in names
        # layer 0 files should be gone
        assert all("layer_0" not in n for n in names)


class TestFromCheckpoint:
    def test_no_checkpoints_returns_none(self, tmp_path):
        trainer = Trainer(config=None)
        assert trainer.from_checkpoint() is None  # no run folder yet
        assert trainer.from_checkpoint(tmp_path) is None

    def test_ignores_other_files(self, tmp_path):
        (tmp_path / "train.log").write_text("log")
        (tmp_path / "2026-08-19-12-57-00").mkdir()
        torch.save({"marker": 1}, tmp_path / "best_finetune.ckpt")
        trainer = Trainer(config=None)
        assert trainer.from_checkpoint(tmp_path) is None
        torch.save({"marker": 7}, tmp_path / "model_layer_0_neuron_0_step_7.ckpt")
        assert trainer.from_checkpoint(tmp_path)["marker"] == 7

    def test_picks_latest(self, tmp_path):
        torch.save({"marker": 1}, tmp_path / "model_layer_0_neuron_0_step_5.ckpt")
        torch.save({"marker": 99}, tmp_path / "model_layer_0_neuron_0_step_99.ckpt")
        # spurious _last file should be ignored
        torch.save({"marker": -1}, tmp_path / "model_last.ckpt")

        trainer = Trainer(config=None)
        data = trainer.from_checkpoint(tmp_path)
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
        run = trainer.run_dir
        assert run.parent == tmp_path and (run / "model_last.ckpt").exists()

        # Build a fresh model and reload state
        model2 = SONN(cfg, d_model=4)
        trainer.load_model_checkpoint(model2)
        assert len(model2.layers) == 1

        # A new trainer finds the newest run with a saved model, or a named one
        later = Trainer(config=cfg)
        model3 = SONN(cfg, d_model=4)
        later.load_model_checkpoint(model3)
        assert later.run_dir == run and len(model3.layers) == 1
        model4 = SONN(cfg, d_model=4)
        Trainer(config=cfg).load_model_checkpoint(model4, run=run.name)
        assert len(model4.layers) == 1

    def test_load_without_any_run_raises(self, tmp_path):
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
        with pytest.raises(FileNotFoundError, match="model_last.ckpt"):
            Trainer(config=cfg).load_model_checkpoint(SONN(cfg, d_model=4))


class TestRunFolders:
    def test_new_run_dir_name_and_clash_suffix(self, tmp_path, monkeypatch):
        import datetime as _dt

        class _Fixed(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, 4, 10, 15, 30)

        monkeypatch.setattr("torchsonn.trainer.datetime.datetime", _Fixed)
        first = Trainer._new_run_dir(tmp_path)
        second = Trainer._new_run_dir(tmp_path)
        assert first.name == "2026-10-04-10-15-30"
        assert second.name == "2026-10-04-10-15-30-2"

    def test_run_folders_order_and_filter(self, tmp_path):
        for name in ["2026-10-04-10-15-30-10", "2026-10-04-10-15-30", "2026-10-04-10-15-30-9",
                     "2026-10-03-23-59-59", "2026-08-19-12-57", "fold_01", "notes"]:
            (tmp_path / name).mkdir()
        (tmp_path / "2026-10-05-00-00-00").write_text("a file, not a folder")
        names = [p.name for p in Trainer.run_folders(tmp_path)]
        assert names == ["2026-10-03-23-59-59", "2026-10-04-10-15-30",
                         "2026-10-04-10-15-30-9", "2026-10-04-10-15-30-10"]
        assert Trainer.run_folders(tmp_path / "missing") == []


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
        )
        d = ckpt.to_dict()
        # FakeOpt's state_dict was unwrapped via the safe_value branch
        assert d["opt"] == {"v": 1}


class TestLayerAccumulator:
    def test_defaults(self):
        a = LayerAccumulator()
        assert a.err == []
        assert a.module_idxs == []
        assert a.layer_completed is False


# --- Layer-growth stop rule (GrowthCriterion) --------------------------------

from torchsonn.trainer import GrowthCriterion


def _run_rule(errors, **kw):
    g = GrowthCriterion(**kw)
    for i, e in enumerate(errors):
        if g.update(i, e):
            return i, g
    return None, g


class TestGrowthCriterion:
    # The California Legendre curve to three decimals; the two variants differ
    # at layer 8 by less than the printed precision (loop vs batched LBFGS).
    CURVE = [0.194, 0.182, 0.173, 0.172, 0.170, 0.170, 0.168, 0.169]

    def test_margin_makes_the_depth_insensitive_to_rounding(self):
        for tail in ([0.16799, 0.167, 0.167, 0.167], [0.1675, 0.167, 0.167, 0.167], [0.1681, 0.167, 0.167, 0.167]):
            stop, g = _run_rule(self.CURVE + tail, width=3, epsilon=1e-3, min_delta=0.002)
            assert stop == 9, (tail, stop)
            assert g.best_index == 9

    def test_small_steps_add_up_against_the_last_accepted_improvement(self):
        # 0.001 per layer never clears a 0.002 margin on its own; measured
        # against the last accepted improvement every second layer does.
        errs = [0.200] + [0.200 - 0.001 * k for k in range(1, 12)]
        stop, g = _run_rule(errs, width=3, epsilon=0.0, min_delta=0.002)
        assert stop is None
        assert g.last_improved_index == 10 and g.best_index == 11

    def test_legacy_settings_match_the_old_rule_except_the_documented_case(self):
        # Old rule, width 2, relative 1e-3: stop when two layers pass the best
        # without a new best, or when a new best falls short of the margin.
        stop, g = _run_rule([0.20, 0.19, 0.195, 0.196], width=2, epsilon=1e-3, min_delta=0.0)
        assert stop == 3 and g.best_index == 1
        # A new best short of the margin used to stop on the spot; now it
        # counts toward the window and the run continues.
        stop, g = _run_rule([0.20, 0.19, 0.18999, 0.17, 0.16], width=2, epsilon=1e-3, min_delta=0.0)
        assert stop is None and g.best_index == 4
        # ... and stops once the window is full of such layers.
        stop, g = _run_rule([0.20, 0.19, 0.18999, 0.18998], width=2, epsilon=1e-3, min_delta=0.0)
        assert stop == 3 and g.best_index == 3
        # Width 1 with a vanishing margin (the smoke-test setting): stop at
        # the first layer that does not lower the error.
        stop, g = _run_rule([0.20, 0.19, 0.19], width=1, epsilon=1e-9, min_delta=0.0)
        assert stop == 2 and g.best_index == 1

    def test_kept_depth_is_the_best_error_even_below_the_margin(self):
        stop, g = _run_rule([0.20, 0.19, 0.1899, 0.1898, 0.1897], width=3, epsilon=0.0, min_delta=0.01)
        assert stop == 4
        assert g.best_index == 4 and g.last_improved_index == 1

    def test_describe_mentions_gain_margin_and_window(self):
        g = GrowthCriterion(width=3, epsilon=1e-3, min_delta=0.002)
        g.update(0, 0.2)
        assert "first layer" in g.describe(0, 0.2)
        g.update(1, 0.19)
        line = g.describe(1, 0.19)
        assert "improved by +0.0100" in line and "margin 0.0020" in line and "0 of 3" in line
