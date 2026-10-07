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
                  "eval_step_interval": 5, "verbose": False, "keep_last_n": 2,
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
    cfg = _cfg(tmp_path, model={"nbest_neurons": 10}, joint={"lam": 1.0, "gate_lr": 0.2, "max_steps": 80})
    model, _ = _train(cfg)
    # C(5, 2) = 10 candidates, a head of 10: only the penalty removes neurons.
    assert len(model.layers[0]) < 10
    steps = [r.getMessage() for r in caplog.records if r.getMessage().startswith("joint step")]
    assert steps and "active" in steps[-1] and "lam*penalty" in steps[-1]


def test_group_lasso_zeroes_whole_neurons(tmp_path):
    cfg = _cfg(tmp_path, model={"nbest_neurons": 10},
               joint={"kind": "group_lasso", "lam": 200.0, "lr": 1e-2, "max_steps": 80})
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
    model, _ = _train(_cfg(tmp_path, model={"nbest_neurons": 12}, joint={"lam": 0.0}))
    assert len(model.layers[0]) == 10


def test_fold_keeps_the_prediction_of_the_gated_pool(tmp_path):
    """Folding the scale and the deterministic gates into the head, with every
    neuron kept, reproduces the pool's prediction exactly (up to rounding)."""
    cfg = _cfg(tmp_path, model={"nbest_neurons": 10})
    torch.manual_seed(0)
    model = SONN(cfg, d_model=5)
    trainer = Trainer(config=cfg)
    train_dl, dev_dl = _dl(512, 0), _dl(128, 1)
    trainer._fit_target_scale(model, train_dl)
    layer = model.create_layer(0)
    model.layers.append(layer)
    trainer.fit_layer_inputs(model, layer, train_dl)
    trainer.train_layer(model, layer, train_dl, dev_dl, None, nbest=10, selection_method="plain")
    model.out_proj = torch.nn.Linear(10, 1)
    trainer.train_out_proj(model, train_dl, dev_dl)
    x = torch.randn(16, 5)
    with torch.no_grad():
        headed = model.infer(x)
    pool = trainer._joint_pool(model, train_dl, cfg.train.joint_sparsity)
    pool.eval()
    with torch.no_grad():
        # moving the head into column order and scaling it changes nothing
        assert torch.allclose(pool.predict(x), headed, atol=1e-5)
        pool.gate.log_alpha.copy_(torch.linspace(-1.0, 3.0, 10))
        gated = pool.predict(x, pool.eval_gates())
    trainer._fold_joint_pool(model, pool, torch.arange(10), 10)
    with torch.no_grad():
        assert torch.allclose(model.infer(x), gated, atol=1e-5)


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
    trainer.fit_layer_inputs(model, layer, _dl(256, 0))
    trainer.train_layer(model, layer, _dl(256, 0), _dl(64, 1), None, nbest=6, selection_method="omp")
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
                                   {"lam_warmup_frac": 2.0}, {"max_steps": 0}, {"gate_lr": 0.0}])
def test_joint_rejects_bad_pass_settings(tmp_path, joint):
    with pytest.raises(ValueError, match="joint_sparsity"):
        Trainer(config=_cfg(tmp_path, joint=joint))


def test_joint_does_not_resume(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="resume"):
        Trainer(config=cfg).train(SONN(cfg, d_model=5), _dl(64, 0), _dl(32, 1), _dl(32, 2), resume=True)
