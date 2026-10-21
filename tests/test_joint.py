"""The joint selection mode (model.selection_mode: joint, Trainer.train_joint)."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from torchsonn.config import SONNConfig
from torchsonn.data.dataset import SONNDataset
from torchsonn.model import SONN
from torchsonn.trainer import Trainer
from torchsonn.training import evaluation, normalization


def _dl(n: int, seed: int, d: int = 5, classes: int = 0) -> DataLoader:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype("float32")
    y = x[:, 0] + 0.5 * x[:, 1] * x[:, 2] - 0.3 * x[:, 3] ** 2 + 0.1 * rng.standard_normal(n)
    if classes:
        y = np.digitize(y, np.quantile(y, np.linspace(0, 1, classes + 1)[1:-1])).astype("int64")
        return DataLoader(SONNDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=128)
    return DataLoader(SONNDataset(torch.from_numpy(x), torch.from_numpy(y.astype("float32"))), batch_size=128)


def _cfg(tmp_path, model=None, joint=None, **train):
    over = {
        "model": {"type": "regressor", "num_classes": 1, "nbest_neurons": 3, "soft_binner": False,
                  "max_neuron_models": 100, "ref_functions": [{"legendre": {"degree": 2}}],
                  "use_output_projection": True, "selection_mode": "joint", "shortcut": False,
                  **(model or {})},
        "train": {"checkpoint_dir": str(tmp_path), "device": "cpu", "batch_size": 128, "steps": 20,
                  "eval_step_interval": 5, "verbose": False, "keep_last_n": 2, "max_layer_count": 1,
                  "optimizer": {"name": "adam", "verbose": False,
                                "optimizer_params": {"lr": 1e-2, "min_lr": 1e-4, "gamma": 0.5}},
                  "out_proj_train": {"optimizer": "lbfgs", "max_steps": 10, "eval_interval": 5},
                  "joint_sparsity": {"lam": 0.1, "gate_lr": 0.05, "max_steps": 60, "eval_interval": 20,
                                     "prefilter": "none", **(joint or {})},
                  **train},
    }
    return OmegaConf.merge(OmegaConf.structured(SONNConfig), over)


def _train(cfg, d=5, classes=0):
    torch.manual_seed(0)
    model = SONN(cfg, d_model=d)
    trainer = Trainer(config=cfg)
    trainer.train(model, _dl(512, 0, d, classes), _dl(128, 1, d, classes), _dl(64, 2, d, classes), verbose=False)
    return model, trainer


@pytest.mark.parametrize("kind", ["gates", "group_lasso"])
def test_joint_builds_one_layer_with_the_configured_head(tmp_path, kind):
    cfg = _cfg(tmp_path, joint={"kind": kind, "lam": 0.1 if kind == "gates" else 5.0})
    model, trainer = _train(cfg)
    assert len(model.layers) == 1
    assert 1 <= len(model.layers[0]) <= 3
    assert model.out_proj.in_features == 3
    assert model.layer_err == [model.layers[0].err]
    x = torch.randn(7, 5)
    pred = model.infer(x)
    assert pred.shape == (7,) and torch.isfinite(pred).all()
    # model_last.ckpt holds the finished model: a fresh model restores it.
    fresh = SONN(cfg, d_model=5)
    trainer.load_model_checkpoint(fresh)
    assert torch.equal(fresh.infer(x), pred)


def test_joint_repeats_with_a_seed(tmp_path):
    x = torch.randn(9, 5)
    a, _ = _train(_cfg(tmp_path / "a"))
    b, _ = _train(_cfg(tmp_path / "b"))
    assert torch.equal(a.infer(x), b.infer(x))


def test_gates_close_under_the_penalty(tmp_path, caplog):
    caplog.set_level("INFO", logger="torchsonn.trainer")
    cfg = _cfg(tmp_path, model={"nbest_neurons": 10},
               joint={"lam": 1.0, "gate_lr": 0.2, "max_steps": 80, "lam_control": "fixed"})
    model, _ = _train(cfg)
    # C(5, 2) = 10 candidates, a head of 10: only the penalty removes neurons.
    assert len(model.layers[0]) < 10
    steps = [r.getMessage() for r in caplog.records if r.getMessage().startswith("joint step")]
    assert steps and "active" in steps[-1] and "lam*penalty" in steps[-1]


@pytest.mark.parametrize("kind", ["gates", "group_lasso"])
def test_the_diversity_term_enters_the_pass(tmp_path, caplog, kind):
    caplog.set_level("INFO", logger="torchsonn.trainer")
    cfg = _cfg(tmp_path, joint={"kind": kind, "lam": 0.1 if kind == "gates" else 5.0,
                                "diversity_lam": 1.0, "diversity_tau": 0.1})
    model, _ = _train(cfg)
    assert 1 <= len(model.layers[0]) <= 3
    steps = [r.getMessage() for r in caplog.records if r.getMessage().startswith("joint step")]
    assert steps and all("diversity" in s for s in steps)


def test_group_lasso_zeroes_whole_neurons(tmp_path):
    cfg = _cfg(tmp_path, model={"nbest_neurons": 10},
               joint={"kind": "group_lasso", "lam": 1.0, "lr": 2e-2, "max_steps": 150, "lam_control": "fixed"})
    model, _ = _train(cfg)
    assert len(model.layers[0]) < 10


def test_prefilter_sizes_the_pool(tmp_path, monkeypatch):
    sizes = []
    original = Trainer._sparsify_joint_pool

    def spy(self, model, pool, *a, **kw):
        sizes.append(pool.size)
        return original(self, model, pool, *a, **kw)

    monkeypatch.setattr(Trainer, "_sparsify_joint_pool", spy)
    _train(_cfg(tmp_path / "n", model={"nbest_neurons": 2}, joint={"prefilter": "none"}))
    _train(_cfg(tmp_path / "p", model={"nbest_neurons": 2}, joint={"prefilter": "plain", "prefilter_mult": 2.5}))
    # 10 candidate pairs; 'plain' keeps ceil(2.5 x 2) = 5 of them.
    assert sizes == [10, 5]


def test_a_head_as_wide_as_the_pool_keeps_every_neuron_without_a_penalty(tmp_path):
    model, _ = _train(_cfg(tmp_path, model={"nbest_neurons": 12}, joint={"lam": 0.0, "lam_control": "fixed"}))
    assert len(model.layers[0]) == 10


def test_a_steered_pass_is_skipped_when_the_pool_fits_the_head(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(Trainer, "_sparsify_joint_pool", lambda *a, **kw: called.append(1))
    model, _ = _train(_cfg(tmp_path, model={"nbest_neurons": 12}))
    assert not called and len(model.layers[0]) == 10


@pytest.mark.parametrize("kind, lam", [("gates", 0.05), ("group_lasso", 0.05)])
def test_adaptive_control_lands_on_the_target(tmp_path, caplog, kind, lam):
    caplog.set_level("INFO", logger="torchsonn.trainer")
    cfg = _cfg(tmp_path, model={"nbest_neurons": 4},
               joint={"kind": kind, "lam": lam, "gate_lr": 0.05, "lr": 1e-2, "max_steps": 1500,
                      "eval_interval": 10, "count_patience": 2, "lam_step": 1.3})
    model, _ = _train(cfg)
    assert len(model.layers[0]) == 4
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("joint pass:") and ("stopping at step" in m or "rolling back" in m) for m in msgs)


def test_an_overshoot_rolls_back_to_the_closest_state_from_above(tmp_path, monkeypatch, caplog):
    """Counts 10, 10, 6, 2, 2 against a target of 3: the pass rolls back to the
    state of the third evaluation (6 active) and the cut keeps 3 of those 6."""
    from torchsonn import trainer as trainer_mod
    caplog.set_level("INFO", logger="torchsonn.trainer")
    script = iter([10, 10, 6, 2, 2])
    masks = {}
    loads = []
    original_load = trainer_mod._JointPool.load
    original_observe = trainer_mod._CountControl.observe

    def observe(self, step, active, loss, adjust, penalty=None):
        count = next(script)
        imp = torch.zeros(10)
        imp[:count] = 1.0
        masks["imp"] = imp
        masks["at"] = step
        return original_observe(self, step, count, loss, adjust, penalty)

    def load(self, state):
        loads.append(masks["at"])
        masks["imp"] = torch.zeros(10)
        masks["imp"][:6] = 1.0
        return original_load(self, state)

    monkeypatch.setattr(trainer_mod._CountControl, "observe", observe)
    monkeypatch.setattr(trainer_mod._JointPool, "load", load)
    monkeypatch.setattr(trainer_mod._JointPool, "importance",
                        lambda self: masks.get("imp", torch.ones(10)))
    cfg = _cfg(tmp_path, joint={"max_steps": 100, "eval_interval": 10, "count_patience": 2,
                                "lam_warmup_frac": 0.0})
    model, _ = _train(cfg)
    assert loads == [50]
    assert any("rolling back to step 30 (6 active)" in r.getMessage() for r in caplog.records)
    assert any("keeping the 3 with the largest contribution" in r.getMessage() for r in caplog.records)
    assert len(model.layers[0]) == 3


def _pool_model(tmp_path):
    """A one-layer model with 10 fitted candidates and a head fitted over them."""
    cfg = _cfg(tmp_path, model={"nbest_neurons": 10})
    torch.manual_seed(0)
    model = SONN(cfg, d_model=5)
    trainer = Trainer(config=cfg)
    train_dl, dev_dl = _dl(512, 0), _dl(128, 1)
    normalization.fit_target_scale(model, train_dl, "cpu", batch_callback=None)
    layer = model.create_layer(0)
    model.layers.append(layer)
    normalization.fit_layer_inputs(model, layer, train_dl, batch_callback=None)
    trainer.train_layer(model, layer, train_dl, dev_dl, None, nbest=10, selection_method="plain")
    model.out_proj = torch.nn.Linear(10, 1)
    trainer.train_out_proj(model, train_dl, dev_dl)
    return cfg, model, trainer, train_dl


def test_the_pool_reproduces_the_headed_layer_from_its_input(tmp_path):
    """Moving the fitted head into column order, standardizing the columns
    and reading the layer's precomputed input changes no prediction."""
    cfg, model, trainer, train_dl = _pool_model(tmp_path)
    x = torch.randn(16, 5)
    with torch.no_grad():
        headed = model.infer(x)
        x_in = model(x, skip_last_layer=True)
    pool = trainer._joint_pool(model, evaluation.precompute_dl(model, train_dl, model.device, batch_callback=None),
                               cfg.train.joint_sparsity)
    pool.eval()
    with torch.no_grad():
        assert torch.allclose(pool.predict(x_in), headed, atol=1e-5)
        # a closed gate removes the neuron's deviation from its mean, nothing more
        pool.gate.log_alpha.fill_(-9.0)
        assert torch.allclose(pool.predict(x_in, pool.eval_gates()), pool.head.bias.expand(16), atol=1e-6)


def _deep_cfg(tmp_path, **train):
    return _cfg(tmp_path, model={"shortcut": True},
                **{"max_layer_count": 3, "criterion_minimum_width": 3, "stop_train_epsilon_condition": 1e-9,
                   **train})


def test_joint_grows_layers_on_frozen_earlier_layers(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="torchsonn.trainer")
    snapshots = {}
    original = Trainer._train_joint_layer

    def spy(self, model, layer, *a, **kw):
        out = original(self, model, layer, *a, **kw)
        snapshots[layer.layer_index] = [p.detach().clone() for p in model.layers[0].parameters()]
        return out

    monkeypatch.setattr(Trainer, "_train_joint_layer", spy)
    cfg = _deep_cfg(tmp_path)
    model, trainer = _train(cfg)
    assert sorted(snapshots) == [0, 1, 2]          # three layers trained
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Joint layer #2:") for m in msgs)
    # layer 0 did not move while layers 1 and 2 trained, nor afterwards
    for later in (1, 2):
        assert all(torch.equal(a, b) for a, b in zip(snapshots[0], snapshots[later]))
    assert all(torch.equal(a, p) for a, p in zip(snapshots[0], model.layers[0].parameters()))
    assert 1 <= len(model.layers) <= 3
    assert all(len(layer) == 3 for layer in model.layers)
    assert model.layer_err[:len(model.layers)] == [layer.err for layer in model.layers]
    assert model.out_proj.in_features == 3
    x = torch.randn(7, 5)
    pred = model.infer(x)
    fresh = SONN(cfg, d_model=5)
    trainer.load_model_checkpoint(fresh)
    assert torch.equal(fresh.infer(x), pred)
    trainer.prune(model)
    assert torch.allclose(model.infer(x), pred, atol=1e-6)


def test_survivors_carry_their_own_dev_error_after_the_pass(tmp_path):
    cfg = _cfg(tmp_path)
    model, _ = _train(cfg)
    layer = model.layers[0]
    xs, ys = zip(*_dl(128, 1))
    x, y = torch.cat(xs), torch.cat(ys)
    with torch.no_grad():
        out = model(x)                                   # (N, 3), no head
    own = ((out - y[:, None]) ** 2).sum(0) / ((y - y.mean()) ** 2).sum()
    assert torch.allclose(layer.err_values, own, rtol=1e-4)
    assert layer.err == pytest.approx(float(own.min()), rel=1e-4)


def test_readout_layer_error_is_the_dev_loss_of_a_head_over_the_survivors(tmp_path, monkeypatch):
    readouts = []
    original = Trainer._joint_readout

    def spy(self, *a, **kw):
        out = original(self, *a, **kw)
        readouts.append(out[0])
        return out

    monkeypatch.setattr(Trainer, "_joint_readout", spy)
    model, _ = _train(_deep_cfg(tmp_path, max_layer_count=2, layer_err_source="readout"))
    assert len(readouts) == 2
    assert model.layer_err == pytest.approx(readouts)


def test_a_joint_pool_skips_the_per_layer_readout_head(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(Trainer, "_train_layer_finetune", lambda *a, **kw: called.append(1) or 0.0)
    _train(_cfg(tmp_path, layer_err_source="readout"))
    assert not called


def test_layer_finetune_runs_on_the_survivors_after_the_cut(tmp_path, monkeypatch):
    widths = []
    original = Trainer._train_layer_finetune

    def spy(self, model, layer, *a, **kw):
        widths.append(len(layer))
        return original(self, model, layer, *a, **kw)

    monkeypatch.setattr(Trainer, "_train_layer_finetune", spy)
    model, _ = _train(_deep_cfg(tmp_path, max_layer_count=2, layer_finetune=True))
    # Once per layer grown, on the cut layer rather than the pool.
    assert widths and all(w == 3 for w in widths)
    assert len(widths) >= len(model.layers)


def test_joint_layer_errors_are_taken_before_the_fine_tune(tmp_path, monkeypatch):
    seen = []
    original = Trainer._train_layer_finetune

    def spy(self, model, layer, *a, **kw):
        seen.append(layer.err_values.clone())
        return original(self, model, layer, *a, **kw)

    monkeypatch.setattr(Trainer, "_train_layer_finetune", spy)
    model, _ = _train(_cfg(tmp_path, layer_finetune=True))
    assert len(seen) == 1
    assert torch.equal(model.layers[0].err_values.cpu(), seen[0].cpu())


def test_restore_weights_keeps_the_candidates_own_fits(tmp_path, monkeypatch):
    fits = {}
    original = Trainer._sparsify_joint_pool

    def spy(self, model, pool, *a, **kw):
        fits["before"] = [p.detach().clone() for p in pool.layer.parameters()]
        out = original(self, model, pool, *a, **kw)
        fits["after"] = [p.detach().clone() for p in pool.layer.parameters()]
        return out

    monkeypatch.setattr(Trainer, "_sparsify_joint_pool", spy)
    cfg = _cfg(tmp_path, model={"nbest_neurons": 4}, joint={"restore_weights": True, "lam": 1.0, "gate_lr": 0.2})
    model, _ = _train(cfg)
    layer = model.layers[0]
    # The pass moved the coefficients; the kept ones are back at their own fits.
    assert any(not torch.equal(b, a) for b, a in zip(fits["before"], fits["after"]))
    kept = [p.detach() for p in layer.parameters()]
    pool_rows = torch.cat([b.reshape(b.shape[0], -1) for b in fits["before"]], dim=1)
    kept_rows = torch.cat([k.reshape(k.shape[0], -1) for k in kept], dim=1)
    for row in kept_rows:
        assert any(torch.equal(row, r) for r in pool_rows)


def test_joint_last_selects_only_the_last_layer_jointly(tmp_path, monkeypatch):
    calls = []
    original = Trainer._train_joint_layer

    def spy(self, model, layer, *a, **kw):
        calls.append(layer.layer_index)
        return original(self, model, layer, *a, **kw)

    monkeypatch.setattr(Trainer, "_train_joint_layer", spy)
    cfg = _cfg(tmp_path, model={"shortcut": True, "selection_mode": "joint_last"},
               max_layer_count=3, criterion_minimum_width=4, stop_train_epsilon_condition=1e-9)
    model, trainer = _train(cfg)
    assert calls == [len(model.layers) - 1]
    assert len(model.layers[-1]) == 3
    assert model.out_proj.in_features == 3
    trainer.train_out_proj(model, _dl(512, 0), _dl(128, 1))
    pred = model.infer(torch.randn(6, 5))
    assert pred.shape == (6,) and torch.isfinite(pred).all()


def test_joint_multiclass(tmp_path):
    cfg = _cfg(tmp_path, model={"type": "multi-class", "num_classes": 3, "soft_binner": True})
    model, _ = _train(cfg, classes=3)
    out = model.infer(torch.randn(4, 5))
    assert out.shape == (4, 3)
    assert torch.allclose(out.exp().sum(dim=1), torch.ones(4), atol=1e-5)


def test_train_layer_selection_overrides(tmp_path):
    cfg = _cfg(tmp_path, model={"selection_mode": "greedy", "nbest_neurons": 2})
    model = SONN(cfg, d_model=5)
    trainer = Trainer(config=cfg)
    layer = model.create_layer(0)
    model.layers.append(layer)
    normalization.fit_layer_inputs(model, layer, _dl(256, 0), batch_callback=None)
    trainer.train_layer(model, layer, _dl(256, 0), _dl(64, 1), None, nbest=6, selection_method="max_volume")
    assert len(layer) == 6


@pytest.mark.parametrize("model_over, match", [
    ({"use_output_projection": False}, "use_output_projection"),
    ({"type": "binary", "num_classes": 2}, "use_output_projection"),
    ({"use_layer_norm": True}, "use_layer_norm"),
    ({"selection_mode": "joint_greedy"}, "selection_mode"),
])
def test_joint_rejects_configs_it_cannot_honour(tmp_path, model_over, match):
    with pytest.raises(ValueError, match=match):
        Trainer(config=_cfg(tmp_path, model=model_over))


@pytest.mark.parametrize("joint", [{"prefilter": "top"}, {"prefilter_mult": 0.5}, {"lam": -1.0},
                                   {"lam_warmup_frac": 2.0}, {"max_steps": 0}, {"gate_lr": 0.0},
                                   {"diversity_lam": -1.0}, {"diversity_tau": 1.0}])
def test_joint_rejects_bad_pass_settings(tmp_path, joint):
    with pytest.raises(ValueError, match="joint_sparsity"):
        Trainer(config=_cfg(tmp_path, joint=joint))


def test_joint_does_not_resume(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="resume"):
        Trainer(config=cfg).train(SONN(cfg, d_model=5), _dl(64, 0), _dl(32, 1), _dl(32, 2), resume=True)


class TestCountControl:
    @staticmethod
    def _control(mode="adaptive", target=10, margin=0.2, patience=2):
        from torchsonn.trainer import _CountControl
        return _CountControl(mode, target, margin, patience, lam_step=2.0, lam=1.0)

    def test_band_from_the_margin(self):
        assert (self._control().target, self._control().hi) == (10, 12)
        assert self._control(target=3, margin=0.1).hi == 3

    def test_lam_grows_above_the_band_only_while_the_count_stalls(self):
        c = self._control()
        c.observe(1, 30, 0.5, adjust=True)      # first look: not falling -> x2
        assert c.lam == 2.0
        c.observe(2, 25, 0.5, adjust=True)      # falling -> hold
        assert c.lam == 2.0
        c.observe(3, 25, 0.5, adjust=True)      # stalled -> x2
        assert c.lam == 4.0
        c.observe(4, 25, 0.5, adjust=False)     # warm-up -> never moves
        assert c.lam == 4.0

    def test_lam_holds_while_the_penalty_falls(self):
        c = self._control()
        c.observe(1, 30, 0.5, adjust=True, penalty=10.0)     # x2
        c.observe(2, 30, 0.5, adjust=True, penalty=9.5)      # penalty -5% -> hold
        assert c.lam == 2.0
        c.observe(3, 30, 0.5, adjust=True, penalty=9.45)     # -0.5% -> stalled, x2
        assert c.lam == 4.0

    def test_adaptive_lowers_lam_below_the_target_and_anneal_does_not(self):
        a, n = self._control("adaptive", patience=5), self._control("anneal", patience=5)
        for c in (a, n):
            c.observe(1, 8, 0.5, adjust=True)
        assert a.lam == 0.5 and n.lam == 1.0

    def test_holds_in_the_band_and_stops_after_patience(self):
        c = self._control()
        assert c.observe(1, 12, 0.5, adjust=True) == (True, None)
        assert c.lam == 1.0
        assert c.observe(2, 11, 0.4, adjust=True) == (True, "band")

    def test_stops_below_after_patience(self):
        c = self._control()
        c.observe(1, 20, 0.5, adjust=True)
        assert c.observe(2, 9, 0.5, adjust=True) == (False, None)
        assert c.observe(3, 9, 0.5, adjust=True) == (False, "below")
        assert (c.best_count, c.best_step) == (20, 1)

    def test_copy_is_the_closest_count_from_above_and_the_lower_loss_on_a_tie(self):
        c = self._control(patience=99)
        assert c.observe(1, 20, 0.5, adjust=True)[0]
        assert c.observe(2, 15, 0.6, adjust=True)[0]
        assert not c.observe(3, 15, 0.7, adjust=True)[0]
        assert c.observe(4, 15, 0.4, adjust=True)[0]
        assert not c.observe(5, 9, 0.1, adjust=True)[0]
        assert (c.best_count, c.best_loss, c.best_step) == (15, 0.4, 4)

    def test_fixed_never_stops_or_moves(self):
        c = self._control("fixed")
        for step, count in enumerate([30, 11, 11, 11, 3, 3, 3]):
            assert c.observe(step, count, 0.5, adjust=True)[1] is None
        assert c.lam == 1.0
