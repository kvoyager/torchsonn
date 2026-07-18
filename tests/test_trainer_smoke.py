"""End-to-end smoke test for the Trainer.

Spins up a tiny regressor on synthetic 2-D input, runs two layers, and
checks that the resulting model can predict + serialize. The goal is
coverage of `train_layer → train_model_ensemble → neuron_selection →
save_model_checkpoint`, not numerical accuracy.
"""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from torchsonn.config import SONNConfig
from torchsonn.data.dataset import SONNDataset
from torchsonn.model import SONN
from torchsonn.trainer import Trainer


def _cfg(tmp_path, **train_overrides) -> OmegaConf:
    base = OmegaConf.structured(SONNConfig)
    overrides = OmegaConf.create(
        {
            "model": {
                "type": "regressor",
                "num_classes": 1,
                "nbest_neurons": 2,
                "soft_binner": False,
                "max_neuron_models": 3,
                "ref_functions": ["linear_cov"],
                "shortcut": False,
                "use_layer_norm": False,
            },
            "train": {
                "checkpoint_dir": str(tmp_path),
                "device": "cpu",
                "dtype": "float32",
                "batch_size": 8,
                "steps": 20,
                "eval_step_interval": 5,
                "criterion_type": "validate",
                "max_layer_count": 2,
                "criterion_minimum_width": 1,
                "stop_train_epsilon_condition": 1e-9,
                "early_stop_tolerance_steps": 4,
                "keep_last_n": 2,
                "verbose": False,
                "optimizer": {
                    "name": "adam",
                    "verbose": False,
                    "optimizer_params": {
                        "lr": 1.0e-2,
                        "min_lr": 1.0e-4,
                        "gamma": 0.5,
                        "clip_value": 1.0,
                        "clip_norm": 5.0,
                    },
                },
                "scheduler": {"name": None, "scheduler_params": None},
                "save_interval": 1000,
            },
        }
    )
    overrides = OmegaConf.merge(overrides, OmegaConf.create({"train": train_overrides}))
    return OmegaConf.merge(base, overrides)


def _make_dl(n: int, d: int = 4) -> DataLoader:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, d)).astype("float32")
    y = (x[:, 0] + 0.5 * x[:, 1] * x[:, 2]).astype("float32")
    ds = SONNDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=8)


def test_train_end_to_end_runs(tmp_path):
    cfg = _cfg(tmp_path)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)

    train_dl = _make_dl(48)
    dev_dl = _make_dl(16)
    test_dl = _make_dl(8)

    trained = trainer.train(model, train_dl, dev_dl, test_dl, verbose=False)
    # at least one layer survived
    assert len(trained.layers) >= 1
    # predict
    x = torch.randn(4, 4)
    out = trained.infer(x)
    assert out.shape[0] == 4


def test_train_multiclass_softbinner(tmp_path):
    """Drive the multi-class soft-binner branches end-to-end."""
    base = OmegaConf.structured(SONNConfig)
    cfg = OmegaConf.merge(
        base,
        OmegaConf.create(
            {
                "model": {
                    "type": "multi-class",
                    "num_classes": 3,
                    "nbest_neurons": 2,
                    "soft_binner": True,
                    "max_neuron_models": 3,
                    "ref_functions": ["linear_cov"],
                    "shortcut": False,
                    "use_layer_norm": False,
                },
                "train": {
                    "checkpoint_dir": str(tmp_path),
                    "device": "cpu",
                    "dtype": "float32",
                    "batch_size": 16,
                    "steps": 20,
                    "eval_step_interval": 5,
                    "criterion_type": "validate",
                    "max_layer_count": 1,
                    "criterion_minimum_width": 1,
                    "stop_train_epsilon_condition": 1e-9,
                    "early_stop_tolerance_steps": 4,
                    "keep_last_n": 2,
                    "verbose": False,
                    "optimizer": {
                        "name": "adam",
                        "verbose": False,
                        "optimizer_params": {
                            "lr": 1.0e-2,
                            "min_lr": 1.0e-4,
                            "gamma": 0.5,
                            "clip_value": 1.0,
                            "clip_norm": 5.0,
                        },
                    },
                    "scheduler": {"name": None, "scheduler_params": None},
                    "save_interval": 1000,
                },
            }
        ),
    )
    model = SONN(cfg, d_model=4)

    rng = np.random.default_rng(0)
    n = 48
    x = rng.standard_normal((n, 4)).astype("float32")
    y = (x[:, 0] > 0).astype("int64") + (x[:, 1] > 0).astype("int64")  # values in {0,1,2}
    ds = SONNDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=16)

    trainer = Trainer(config=cfg)
    trained = trainer.train(model, dl, dl, dl, verbose=False)
    assert len(trained.layers) >= 1
    pred = trained.infer(torch.from_numpy(x[:4]))
    assert pred.shape == (4, 3)


def test_train_with_bias_criterion(tmp_path):
    """Drive the bias-criterion path (deepcopy + split loaders + bias_err)."""
    cfg = _cfg(tmp_path, criterion_type="bias", max_layer_count=1)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    train_dl = _make_dl(48)
    dev_dl = _make_dl(16)
    test_dl = _make_dl(8)
    trained = trainer.train(model, train_dl, dev_dl, test_dl, verbose=False)
    assert len(trained.layers) >= 1


def test_train_with_validate_bias_criterion(tmp_path):
    """Combined criterion exercises both regularity and bias branches."""
    cfg = _cfg(tmp_path, criterion_type="validate_bias", max_layer_count=1)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    train_dl = _make_dl(48)
    dev_dl = _make_dl(16)
    test_dl = _make_dl(8)
    trained = trainer.train(model, train_dl, dev_dl, test_dl, verbose=False)
    assert len(trained.layers) >= 1


def test_train_with_lbfgs_out_proj(tmp_path):
    """Drive the train_out_proj LBFGS branch."""
    cfg = _cfg(tmp_path, max_layer_count=1)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create({
            "model": {
                "use_output_projection": True,
                "num_out_neurons": 2,
            },
            "train": {
                "out_proj_train": {
                    "max_steps": 3,
                    "optimizer": "lbfgs",
                    "lbfgs_max_iter": 2,
                    "eval_interval": 1,
                    "early_stop_patience": 5,
                }
            },
        }),
    )
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    train_dl = _make_dl(48)
    dev_dl = _make_dl(16)
    trainer.train(model, train_dl, dev_dl, dev_dl, verbose=False)
    trainer.train_out_proj(model, train_dl, dev_dl)


def test_train_with_precompute_and_shortcut(tmp_path):
    cfg = _cfg(tmp_path, max_layer_count=2, precompute_features=True)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create({"model": {"shortcut": True, "use_layer_norm": True}}),
    )
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    train_dl = _make_dl(48)
    dev_dl = _make_dl(16)
    test_dl = _make_dl(8)
    trained = trainer.train(model, train_dl, dev_dl, test_dl, verbose=False)
    assert len(trained.layers) >= 1


def test_train_with_omp_selection(tmp_path):
    cfg = _cfg(tmp_path, max_layer_count=1, neuron_selection_method="omp")
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    train_dl = _make_dl(48)
    dev_dl = _make_dl(16)
    test_dl = _make_dl(8)
    trainer.train(model, train_dl, dev_dl, test_dl, verbose=False)


def test_train_finetune_and_layer_finetune_multiclass(tmp_path):
    """Drive layer_finetune (inside train_layer) and train_finetune externally."""
    base = OmegaConf.structured(SONNConfig)
    cfg = OmegaConf.merge(
        base,
        OmegaConf.create({
            "model": {
                "type": "multi-class",
                "num_classes": 3,
                "nbest_neurons": 2,
                "soft_binner": True,
                "max_neuron_models": 3,
                "ref_functions": ["linear_cov"],
                "shortcut": False,
                "use_layer_norm": False,
            },
            "train": {
                "checkpoint_dir": str(tmp_path),
                "device": "cpu",
                "dtype": "float32",
                "batch_size": 16,
                "steps": 8,
                "eval_step_interval": 4,
                "criterion_type": "validate",
                "max_layer_count": 2,
                "criterion_minimum_width": 1,
                "stop_train_epsilon_condition": 1e-9,
                "early_stop_tolerance_steps": 4,
                "keep_last_n": 2,
                "verbose": False,
                "layer_finetune": True,
                "optimizer": {
                    "name": "adam",
                    "verbose": False,
                    "optimizer_params": {"lr": 1.0e-2, "min_lr": 1.0e-4, "gamma": 0.5,
                                          "clip_value": 1.0, "clip_norm": 5.0},
                },
                "scheduler": {"name": None, "scheduler_params": None},
                "out_proj_train": {
                    "max_steps": 4,
                    "optimizer": "adam",
                    "eval_interval": 2,
                    "early_stop_patience": 5,
                    "lr": 1.0e-2,
                    "lr_factor": 0.5,
                    "lr_patience": 5,
                    "lr_min": 1.0e-5,
                    "early_stop_min_delta": 1.0e-4,
                    "weight_decay": 0.0,
                },
            },
        }),
    )
    model = SONN(cfg, d_model=4)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((48, 4)).astype("float32")
    y = ((x[:, 0] > 0).astype("int64") + (x[:, 1] > 0).astype("int64")).clip(max=2)
    ds = SONNDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=16)

    trainer = Trainer(config=cfg)
    trained = trainer.train(model, dl, dl, dl, verbose=False)
    # Also run train_finetune explicitly (loss_fn=NLL is needed → multi-class works)
    trainer.train_finetune(trained, dl, dl)


def test_train_with_omp_mixed_selection(tmp_path):
    cfg = _cfg(tmp_path, max_layer_count=1, neuron_selection_method="omp_mixed",
               neuron_selection_orth_threshold=0.1)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    train_dl = _make_dl(48)
    dev_dl = _make_dl(16)
    test_dl = _make_dl(8)
    trainer.train(model, train_dl, dev_dl, test_dl, verbose=False)


def test_trainer_infer_and_prune(tmp_path):
    cfg = _cfg(tmp_path, max_layer_count=2)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    train_dl = _make_dl(48)
    dev_dl = _make_dl(16)
    test_dl = _make_dl(8)
    trained = trainer.train(model, train_dl, dev_dl, test_dl, verbose=False)

    preds, targets = trainer.infer(trained, test_dl, verbose=False)
    assert preds.shape[0] == targets.shape[0]
    # Trainer.prune collapses to best-neuron-only across all layers
    trainer.prune(trained)
    assert len(trained.layers[-1].neuron_models) == 1


def test_train_with_out_proj(tmp_path):
    """Exercise train_out_proj on a regressor with out_proj enabled."""
    cfg = _cfg(tmp_path, max_layer_count=1)
    # Patch model to enable output projection
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create({
            "model": {
                "use_output_projection": True,
                "num_out_neurons": 2,
            },
            "train": {
                "out_proj_train": {
                    "max_steps": 10,
                    "optimizer": "adam",
                    "eval_interval": 5,
                    "early_stop_patience": 5,
                }
            },
        }),
    )
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    train_dl = _make_dl(48)
    dev_dl = _make_dl(16)
    trainer.train(model, train_dl, dev_dl, dev_dl, verbose=False)
    trainer.train_out_proj(model, train_dl, dev_dl)


def test_train_out_proj_raises_without_head(tmp_path):
    cfg = _cfg(tmp_path)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    with pytest.raises(ValueError, match="use_output_projection"):
        trainer.train_out_proj(model, _make_dl(8), _make_dl(8))


def test_train_resume_from_checkpoint(tmp_path):
    """`resume=True` with no existing checkpoint must still kick off a fresh run."""
    cfg = _cfg(tmp_path, max_layer_count=1)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    train_dl = _make_dl(48)
    dev_dl = _make_dl(16)
    test_dl = _make_dl(8)
    # First call writes checkpoints, second resumes from them.
    trainer.train(model, train_dl, dev_dl, test_dl, verbose=False)

    model2 = SONN(cfg, d_model=4)
    trained = trainer.train(model2, train_dl, dev_dl, test_dl, verbose=False, resume=True)
    assert len(trained.layers) >= 1


def test_train_checkpoint_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)

    train_dl = _make_dl(48)
    dev_dl = _make_dl(16)
    test_dl = _make_dl(8)

    trained = trainer.train(model, train_dl, dev_dl, test_dl, verbose=False)

    # Fresh model: reload via model_last.ckpt
    model2 = SONN(cfg, d_model=4)
    trainer.load_model_checkpoint(model2)
    assert len(model2.layers) == len(trained.layers)
