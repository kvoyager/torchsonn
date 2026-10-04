"""End-to-end smoke test for the Trainer.

Spins up a tiny regressor on synthetic 2-D input, runs two layers, and
checks that the resulting model can predict + serialize. The goal is
coverage of `train_layer → train_model_ensemble → neuron_selection →
save_model_checkpoint`, not numerical accuracy.
"""
import math
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


def _mc_cfg(tmp_path, **train_overrides) -> OmegaConf:
    base = OmegaConf.structured(SONNConfig)
    overrides = OmegaConf.create(
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
    )
    overrides = OmegaConf.merge(overrides, OmegaConf.create({"train": train_overrides}))
    return OmegaConf.merge(base, overrides)


def _make_mc_dl(n: int = 48, d: int = 4):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, d)).astype("float32")
    y = (x[:, 0] > 0).astype("int64") + (x[:, 1] > 0).astype("int64")  # values in {0,1,2}
    ds = SONNDataset(torch.from_numpy(x), torch.from_numpy(y))
    return x, DataLoader(ds, batch_size=16)


def test_train_multiclass_softbinner(tmp_path):
    """Drive the multi-class soft-binner branches end-to-end."""
    cfg = _mc_cfg(tmp_path)
    model = SONN(cfg, d_model=4)
    x, dl = _make_mc_dl()

    trainer = Trainer(config=cfg)
    trained = trainer.train(model, dl, dl, dl, verbose=False)
    assert len(trained.layers) >= 1
    pred = trained.infer(torch.from_numpy(x[:4]))
    assert pred.shape == (4, 3)


@pytest.mark.parametrize("criterion_type", ["bias", "validate_bias"])
@pytest.mark.parametrize("bias_ce_type", ["js", "l2"])
def test_train_multiclass_bias_criteria(tmp_path, criterion_type, bias_ce_type):
    """Both multi-class bias criteria must reduce to one error per candidate.

    The 'l2' variant used to return a per-sample (ensemble, N) tensor. That
    broke both bias-using criteria: 'validate_bias' broadcast it against the
    (ensemble,) regularity error in SONN.get_error, and 'bias' carried it into
    neuron selection and failed there on a mask shape.
    """
    cfg = _mc_cfg(tmp_path, criterion_type=criterion_type, bias_ce_type=bias_ce_type)
    model = SONN(cfg, d_model=4)
    x, dl = _make_mc_dl()

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


def _offset_dl(n: int, d: int = 4, seed: int = 0) -> DataLoader:
    """Far off-centre, wide-scale features — identity stats would be visibly
    wrong, so a calibrated squash has to actually move to pass."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n, d)) * 25.0 + 100.0).astype("float32")
    y = (x[:, 0] + 0.5 * x[:, 1] * x[:, 2]).astype("float32")
    ds = SONNDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=8)


def _legendre_cfg(tmp_path, **model_overrides):
    cfg = _cfg(tmp_path, max_layer_count=2)
    return OmegaConf.merge(cfg, OmegaConf.create({
        "model": {"ref_functions": ["legendre"], "shortcut": True,
                  **model_overrides},
    }))


@pytest.mark.parametrize("squash_method", ["sigma", "tanh"])
def test_train_legendre_end_to_end(tmp_path, squash_method):
    """The orthogonal-polynomial families carry per-neuron buffers that the
    trainer vmaps at in_dims=0 — a path no other smoke test covers."""
    cfg = _legendre_cfg(tmp_path, squash_method=squash_method)
    model = SONN(cfg, d_model=4)
    trained = Trainer(config=cfg).train(
        model, _offset_dl(64, seed=1), _offset_dl(24, seed=2),
        _offset_dl(16, seed=3), verbose=False,
    )
    out = trained.infer(torch.randn(5, 4) * 25.0 + 100.0)
    assert out.shape[0] == 5
    assert torch.isfinite(out).all()


def test_train_calibrates_sigma_squash_on_training_set(tmp_path):
    cfg = _legendre_cfg(tmp_path)
    model = SONN(cfg, d_model=4)
    trained = Trainer(config=cfg).train(
        model, _offset_dl(64, seed=1), _offset_dl(24, seed=2),
        _offset_dl(16, seed=3), verbose=False,
    )
    neuron = trained.layers[0].neuron_models[0]
    # Data is centered at 100 with std 25; identity stats (0, 1) would mean the
    # calibration pass never ran.
    assert neuron.squash_norm.mean.abs().min() > 50.0
    assert neuron.squash_norm.std.min() > 5.0
    # prune must have kept the stats aligned with the surviving neurons
    assert neuron.squash_norm.mean.shape == (neuron.num_neurons, neuron.dim)


def test_fit_layer_inputs_matches_actual_layer_inputs(tmp_path):
    """A deeper layer's inputs are the frozen prefix's outputs concatenated
    with the shortcut originals — a different width and scale from the raw
    input, which is what the per-layer (not global) calibration exists for."""
    cfg = _legendre_cfg(tmp_path)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    train_dl = _offset_dl(64, seed=1)

    layer0 = model.create_layer(0)
    model.layers.append(layer0)
    trainer.fit_layer_inputs(model, layer0, train_dl)
    trainer.train_layer(model, layer0, train_dl, _offset_dl(24, seed=2), None)

    layer1 = model.create_layer(1)
    model.layers.append(layer1)
    neuron = layer1.neuron_models[0]
    assert neuron.num_feat == layer0.d_model + model.d_model

    before = neuron.squash_norm.mean.clone()
    trainer.fit_layer_inputs(model, layer1, train_dl)

    with torch.no_grad():
        feats = torch.cat([model(b[0], skip_last_layer=True) for b in train_dl], 0)
    assert feats.shape[1] == neuron.num_feat
    idx = neuron.src_idxs
    assert torch.allclose(neuron.squash_norm.mean, feats.mean(0)[idx], atol=1e-3)
    assert torch.allclose(
        neuron.squash_norm.std, feats.std(0, unbiased=False)[idx], atol=1e-3
    )
    assert not torch.allclose(before, neuron.squash_norm.mean)

    # and the squash bounds those deep, wildly-scaled features
    x = torch.index_select(feats, 1, idx.view(-1)).view(feats.shape[0], -1, neuron.dim)
    assert neuron._squash(x).abs().max() <= 1.0


def test_fit_layer_inputs_skipped_without_sigma_neurons(tmp_path):
    """tanh needs no statistics, so the extra pass must not run at all."""
    cfg = _legendre_cfg(tmp_path, squash_method="tanh")
    model = SONN(cfg, d_model=4)
    layer = model.create_layer(0)
    model.layers.append(layer)

    def _explode(*args, **kwargs):
        raise AssertionError("calibration pass ran for a tanh-squashed layer")

    Trainer(config=cfg).fit_layer_inputs(model, layer, _explode)


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


def test_prune_keeps_every_last_neuron_of_a_headless_neuron_proj_model(tmp_path):
    """Without a head, a use_neuron_proj model predicts from every neuron of
    its last layer, so prune keeps them all and the predictions stay put;
    a second prune changes nothing."""
    cfg = OmegaConf.merge(
        _mc_cfg(tmp_path, max_layer_count=2),
        OmegaConf.create({"model": {"soft_binner": False, "use_neuron_proj": True,
                                    "nbest_neurons": 3, "max_neuron_models": 6}}),
    )
    model = SONN(cfg, d_model=4)
    x, dl = _make_mc_dl(64)
    trainer = Trainer(config=cfg)
    trained = trainer.train(model, dl, dl, dl, verbose=False)
    xs = torch.from_numpy(x)
    width = len(trained.layers[-1])
    assert width > 1 and trained._readout_width(trained.layers[-1]) == width
    with torch.no_grad():
        before = trained.infer(xs)
    trainer.prune(trained)
    assert len(trained.layers[-1]) == width
    with torch.no_grad():
        after = trained.infer(xs)
    assert torch.allclose(before, after, atol=1e-5)
    trainer.prune(trained)
    with torch.no_grad():
        assert torch.allclose(trained.infer(xs), after, atol=1e-5)


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
    first_run = trainer.run_dir

    model2 = SONN(cfg, d_model=4)
    trained = trainer.train(model2, train_dl, dev_dl, test_dl, verbose=False, resume=True)
    assert len(trained.layers) >= 1
    # Resumed in place: no new run folder, the log continues
    assert trainer.run_dir == first_run
    assert Trainer.run_folders(tmp_path) == [first_run]
    assert "Resuming the run in" in (first_run / "train.log").read_text()


def test_train_resume_by_run_name_from_a_new_trainer(tmp_path):
    cfg = _cfg(tmp_path, max_layer_count=1)
    train_dl, dev_dl, test_dl = _make_dl(48), _make_dl(16), _make_dl(8)
    first = Trainer(config=cfg)
    first.train(SONN(cfg, d_model=4), train_dl, dev_dl, test_dl, verbose=False)
    other = Trainer(config=cfg)
    other.train(SONN(cfg, d_model=4), train_dl, dev_dl, test_dl, verbose=False,
                resume=first.run_dir.name)
    assert other.run_dir == first.run_dir
    assert Trainer.run_folders(tmp_path) == [first.run_dir]


def test_train_without_checkpoints_to_resume_starts_new_run(tmp_path):
    cfg = _cfg(tmp_path, max_layer_count=1)
    trainer = Trainer(config=cfg)
    trainer.train(SONN(cfg, d_model=4), _make_dl(48), _make_dl(16), _make_dl(8),
                  verbose=False, resume=True)
    assert Trainer.run_folders(tmp_path) == [trainer.run_dir]
    assert "No run to resume" in (trainer.run_dir / "train.log").read_text()


def test_each_train_call_gets_its_own_run_folder_and_log(tmp_path):
    cfg = _cfg(tmp_path, max_layer_count=1)
    first, second = Trainer(config=cfg), Trainer(config=cfg)
    first.train(SONN(cfg, d_model=4), _make_dl(48), _make_dl(16), _make_dl(8), verbose=False)
    second.train(SONN(cfg, d_model=4), _make_dl(48), _make_dl(16), _make_dl(8), verbose=False)
    assert first.run_dir != second.run_dir
    assert Trainer.run_folders(tmp_path) == [first.run_dir, second.run_dir]
    for trainer in (first, second):
        files = {p.name for p in trainer.run_dir.iterdir()}
        assert {"train.log", "model_last.ckpt"} <= files
        log = (trainer.run_dir / "train.log").read_text()
        assert f"Run folder: {trainer.run_dir}" in log
    # One run log at a time: the first run's log stops when the second starts
    assert f"Run folder: {second.run_dir}" not in (first.run_dir / "train.log").read_text()


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
    assert (trainer.run_dir / "model_last.ckpt").exists()
    assert trainer.run_dir.parent == tmp_path


def test_layer_err_source_readout_rejects_incompatible_config(tmp_path):
    """'readout' needs a model head; a bogus value and multi-class are rejected."""
    with pytest.raises(ValueError, match="use_output_projection"):
        Trainer(config=_cfg(tmp_path, layer_err_source="readout"))
    with pytest.raises(ValueError, match="use_output_projection"):
        Trainer(config=_cfg(tmp_path, layer_err_source="readout", layer_finetune=True))
    with pytest.raises(ValueError, match="layer_err_source"):
        Trainer(config=_cfg(tmp_path, layer_err_source="bogus"))
    cfg = OmegaConf.merge(
        _cfg(tmp_path, layer_err_source="readout", layer_finetune=True),
        OmegaConf.create({"model": {"type": "multi-class", "num_classes": 3,
                                    "use_output_projection": True}}),
    )
    with pytest.raises(NotImplementedError):
        Trainer(config=cfg)


@pytest.mark.parametrize("layer_finetune", [True, False])
def test_layer_err_source_readout_scores_layers_by_head_dev_loss(tmp_path, layer_finetune):
    """Under 'readout' the layer error is a head's dev loss, not the best neuron's.

    With layer_finetune on it is the fine-tune head; off, a temporary head over
    the frozen survivors, whose weights must come out of the search unchanged
    by that measurement.
    """
    cfg = OmegaConf.merge(
        _cfg(tmp_path, layer_err_source="readout", layer_finetune=layer_finetune, max_layer_count=2),
        OmegaConf.create({
            "model": {"use_output_projection": True, "num_out_neurons": 2},
            "train": {"out_proj_train": {"optimizer": "lbfgs", "max_steps": 6,
                                         "eval_interval": 1, "early_stop_patience": 3}},
        }),
    )
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    dl = _make_dl(64)
    trained = trainer.train(model, dl, dl, dl, verbose=False)
    assert len(trained.layers) >= 1
    for layer in trained.layers:
        assert np.isfinite(layer.err)
        # The head over all survivors is a different quantity from the single
        # best neuron's regularity error.
        assert not np.isclose(layer.err, layer.err_values.min().item())
        # The measurement pass must hand the weights back trainable.
        for nm in layer.neuron_models:
            assert nm.weight.requires_grad
    # The rest of the pipeline is unaffected by the criterion switch.
    trainer.train_out_proj(trained, dl, dl)
    out = trained.infer(torch.randn(5, 4))
    assert out.shape == (5,)


# --- The per-layer input pass ------------------------------------------------

from torchsonn.neurons import LinearCovPolynomNeuron as _LinearCov


class _SamplingStub(_LinearCov):
    """A plain family that asks the input pass for a row sample and, when the
    mode resolves to stream, for the streaming pass; records every call."""

    def __init__(self, *a, streams=True, **kw):
        super().__init__(*a, **kw)
        self.streams = streams
        self.samples = []
        self.stream_flags = []
        self.batches = []
        self.finished = 0

    @property
    def needs_input_sample(self):
        return True

    def fit_input_sample(self, x_sample, stream=False, **kw):
        self.samples.append(x_sample.clone())
        self.stream_flags.append(stream)

    @property
    def needs_input_stream(self):
        return self.streams

    def stream_input_batch(self, x_batch):
        self.batches.append(x_batch.clone())

    def finish_input_stream(self):
        self.finished += 1


def _stub_layer(cfg, streams=True):
    model = SONN(cfg, d_model=4)
    layer = model.create_layer(0)
    model.layers.append(layer)
    stub = _SamplingStub(4, 4, None, 0, 0, max_neuron_models=3, streams=streams)
    layer.neuron_models[0] = stub
    return model, layer, stub


def _all_rows(model, dl):
    with torch.no_grad():
        return torch.cat([model(b[0], skip_last_layer=True) for b in dl], 0)


def test_input_pass_hands_the_whole_split_below_the_cap(tmp_path):
    cfg = _cfg(tmp_path, input_sample_rows=1000)
    model, layer, stub = _stub_layer(cfg, streams=False)
    dl = _make_dl(48)
    Trainer(config=cfg).fit_layer_inputs(model, layer, dl)
    assert len(stub.samples) == 1 and stub.stream_flags == [False]
    assert torch.equal(stub.samples[0], _all_rows(model, dl))
    assert stub.batches == [] and stub.finished == 0


def test_input_pass_reservoir_above_the_cap_is_seeded_and_from_the_split(tmp_path):
    cfg = _cfg(tmp_path, input_sample_rows=20, rbf_kmeans_mode="sample")
    dl = _make_dl(96)
    model, layer, stub = _stub_layer(cfg)
    Trainer(config=cfg).fit_layer_inputs(model, layer, dl)
    sample = stub.samples[0]
    rows = _all_rows(model, dl)
    assert sample.shape == (20, rows.shape[1])
    # every sampled row is a row of the split, and not just the first 20
    for r in sample:
        assert (rows == r).all(dim=1).any()
    assert not torch.equal(sample, rows[:20])
    # 'sample' mode forced: no streaming even above the cap
    assert stub.stream_flags == [False] and stub.batches == [] and stub.finished == 0
    # seeded: a fresh trainer with the same seed reproduces it, another seed does not
    model2, layer2, stub2 = _stub_layer(cfg)
    Trainer(config=cfg).fit_layer_inputs(model2, layer2, dl)
    assert torch.equal(stub2.samples[0], sample)
    cfg3 = _cfg(tmp_path, input_sample_rows=20, rbf_kmeans_mode="sample", seed=99)
    model3, layer3, stub3 = _stub_layer(cfg3)
    Trainer(config=cfg3).fit_layer_inputs(model3, layer3, dl)
    assert not torch.equal(stub3.samples[0], sample)


def test_input_pass_streams_above_the_cap_in_auto_mode(tmp_path):
    cfg = _cfg(tmp_path, input_sample_rows=20, rbf_kmeans_passes=2)
    dl = _make_dl(96)          # 12 batches of 8
    model, layer, stub = _stub_layer(cfg)
    Trainer(config=cfg).fit_layer_inputs(model, layer, dl)
    assert stub.stream_flags == [True]
    assert len(stub.batches) == 2 * 12 and stub.finished == 1
    assert torch.equal(torch.cat(stub.batches[:12], 0), _all_rows(model, dl))
    assert torch.equal(torch.cat(stub.batches[12:], 0), _all_rows(model, dl))
    # a module that samples but does not stream is left alone by the pass
    model2, layer2, stub2 = _stub_layer(cfg, streams=False)
    Trainer(config=cfg).fit_layer_inputs(model2, layer2, dl)
    assert stub2.stream_flags == [True] and stub2.batches == [] and stub2.finished == 0


def test_input_pass_stream_mode_can_be_forced_below_the_cap(tmp_path):
    cfg = _cfg(tmp_path, input_sample_rows=1000, rbf_kmeans_mode="stream")
    dl = _make_dl(48)
    model, layer, stub = _stub_layer(cfg)
    Trainer(config=cfg).fit_layer_inputs(model, layer, dl)
    assert stub.stream_flags == [True] and len(stub.batches) == 6 and stub.finished == 1


def test_input_pass_rejects_bad_settings(tmp_path):
    dl = _make_dl(16)
    for bad in ({"rbf_kmeans_mode": "lloyd"}, {"input_sample_rows": 0},
                {"rbf_kmeans_mode": "stream", "rbf_kmeans_passes": 0}):
        cfg = _cfg(tmp_path, **bad)
        model, layer, stub = _stub_layer(cfg)
        with pytest.raises(ValueError):
            Trainer(config=cfg).fit_layer_inputs(model, layer, dl)


def test_input_pass_serves_stats_and_sample_in_one_pass(tmp_path):
    """A Legendre family (moments) next to a sampling family: both get what
    they asked for from one pass, and the old entry point still works."""
    cfg = _legendre_cfg(tmp_path)
    model = SONN(cfg, d_model=4)
    layer = model.create_layer(0)
    model.layers.append(layer)
    stub = _SamplingStub(4, 4, None, 0, 0, max_neuron_models=3, streams=False)
    layer.neuron_models.append(stub)
    dl = _offset_dl(64)
    Trainer(config=cfg).fit_layer_inputs(model, layer, dl)
    leg = layer.neuron_models[0]
    rows = _all_rows(model, dl)
    assert torch.allclose(leg.squash_norm.mean, rows.mean(0)[leg.src_idxs], atol=1e-3)
    assert torch.equal(stub.samples[0], rows)


# --- The RBF family through the trainer --------------------------------------

from torchsonn.neurons import RBFNeuron as _RBFNeuron


@pytest.mark.parametrize("mode", ["sample", "stream"])
def test_rbf_family_trains_end_to_end(tmp_path, mode):
    """Input pass (k-means start, sample or streamed), joint candidate fit of
    weights + centers + widths, selection with the survivor report, prune,
    inference and the checkpoint round trip."""
    cfg = OmegaConf.merge(
        _cfg(tmp_path, max_layer_count=2, rbf_kmeans_mode=mode, rbf_kmeans_iters=5),
        OmegaConf.create({"model": {"ref_functions": [{"rbf": {"centers": 6}}, "linear_cov"],
                                    "shortcut": True}}),
    )
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    dl = _make_dl(64)
    trained = trainer.train(model, dl, dl, dl, verbose=False)
    assert len(trained.layers) >= 1
    rbf_seen = False
    for layer in trained.layers:
        for nm in layer.neuron_models:
            if isinstance(nm, _RBFNeuron):
                rbf_seen = True
                assert nm.centers().shape == (nm.num_neurons, 6, 2)
                assert torch.isfinite(nm.centers()).all() and torch.isfinite(nm.log_width).all()
                assert (nm.displacements().norm(dim=-1) < nm.radius).all()
                with torch.no_grad():
                    scale = nm.width_scales()
                assert (scale <= 4.0).all() and (scale >= 0.25).all()
                assert "center movement" in nm.fit_report()
    assert rbf_seen, "no RBF neuron survived selection in any layer"
    x = torch.randn(5, 4)
    with torch.inference_mode():
        pred = trained.infer(x)
    fresh = SONN(cfg, d_model=4)
    trainer.load_model_checkpoint(fresh, "cpu")
    with torch.inference_mode():
        assert torch.allclose(fresh.infer(x), pred, atol=1e-6)
    trainer.prune(trained)
    with torch.inference_mode():
        assert torch.allclose(trained.infer(x), pred, atol=1e-6)


def test_rbf_candidate_fit_moves_centers(tmp_path):
    """The vmapped candidate fit trains the centers and widths, not just the
    weights: after one layer the survivors' centers differ from their k-means
    start."""
    cfg = OmegaConf.merge(
        _cfg(tmp_path, max_layer_count=1, steps=40),
        OmegaConf.create({"model": {"ref_functions": [{"rbf": {"centers": 4}}], "shortcut": False}}),
    )
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    dl = _make_dl(96)
    trained = trainer.train(model, dl, dl, dl, verbose=False)
    nm = trained.layers[0].neuron_models[0]
    moved = nm.displacements().detach().norm(dim=-1)
    assert moved.max() > 1e-4
    assert isinstance(nm.center_shift, torch.nn.Parameter)


def test_rbf_layer_finetune_unfreezes_centers(tmp_path):
    """train.layer_finetune trains every neuron parameter, so an RBF
    survivor's centers change during the per-layer pass."""
    cfg = OmegaConf.merge(
        _cfg(tmp_path, max_layer_count=1, layer_finetune=True),
        OmegaConf.create({"model": {"ref_functions": [{"rbf": {"centers": 4}}], "shortcut": False,
                                    "use_output_projection": True, "num_out_neurons": 2}}),
    )
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    dl = _make_dl(64)
    layer = model.create_layer(0)
    model.layers.append(layer)
    trainer.fit_layer_inputs(model, layer, dl)
    trainer.train_layer(model, layer, dl, dl, None)
    nm = layer.neuron_models[0]
    before = nm.centers().detach().clone()
    trainer._train_layer_finetune(model, layer, dl, dl)
    assert not torch.equal(nm.centers().detach(), before)
    for p in model.parameters():
        assert p.requires_grad


# --- Candidate early stop on the training loss (train.early_stop_source) ----

@pytest.mark.parametrize("source", ["dev", "train"])
def test_early_stop_source_routes_the_candidate_stop(tmp_path, source, monkeypatch):
    """With 'train' the candidate fit never evaluates the dev split (ds_loss
    is only called after the fit, for selection); with 'dev' it does."""
    cfg = _cfg(tmp_path, max_layer_count=1, early_stop_source=source, steps=30)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    calls = {"during_fit": 0, "total": 0}
    real = Trainer.ds_loss
    state = {"fitting": False}

    def spy(self, *a, **kw):
        calls["total"] += 1
        if state["fitting"]:
            calls["during_fit"] += 1
        return real(self, *a, **kw)

    monkeypatch.setattr(Trainer, "ds_loss", spy)
    real_reg = Trainer.regularity_err
    reg_calls = {"n": 0}

    def reg_spy(self, *a, **kw):
        reg_calls["n"] += 1
        return real_reg(self, *a, **kw)

    monkeypatch.setattr(Trainer, "regularity_err", reg_spy)
    real_step = Trainer.train_model_ensemble

    def wrapped(self, *a, **kw):
        state["fitting"] = True
        try:
            return real_step(self, *a, **kw)
        finally:
            state["fitting"] = False

    monkeypatch.setattr(Trainer, "train_model_ensemble", wrapped)
    dl = _make_dl(64)
    trained = trainer.train(model, dl, dl, dl, verbose=False)
    assert len(trained.layers) == 1
    if source == "train":
        assert calls["total"] == 0                    # the dev split is not touched by the fit
    else:
        assert calls["during_fit"] > 0
    assert reg_calls["n"] > 0                         # selection still scores every candidate on dev


def test_early_stop_source_rejected_when_unknown(tmp_path):
    cfg = _cfg(tmp_path, max_layer_count=1, early_stop_source="test")
    model = SONN(cfg, d_model=4)
    dl = _make_dl(32)
    with pytest.raises(ValueError, match="early_stop_source"):
        Trainer(config=cfg).train(model, dl, dl, dl, verbose=False)


# --- Validation split: per-layer report and stop_source ---------------------

class _NoIter:
    """A loader stand-in that must not be iterated (a split the pass must not read)."""

    def __init__(self, inner):
        self.inner = inner

    def __iter__(self):
        raise AssertionError("this split must not be read here")

    def __len__(self):
        return len(self.inner)


def test_val_split_is_reported_per_layer_and_stored(tmp_path, caplog):
    cfg = _cfg(tmp_path, max_layer_count=2)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    dl, val = _make_dl(64), _make_dl(24)
    with caplog.at_level("INFO", logger="torchsonn.trainer"):
        trained = trainer.train(model, dl, dl, dl, verbose=False, val_dl=val)
    n = len(trained.layers)
    assert len(trained.layer_val_err) >= n and all(v == v for v in trained.layer_val_err)   # finite
    assert all(hasattr(l, "val_err") for l in trained.layers)
    assert any("| val" in r.message and "gap" in r.message for r in caplog.records)
    # without a validation loader nothing is reported or stored
    plain = Trainer(config=cfg).train(SONN(cfg, d_model=4), dl, dl, dl, verbose=False)
    assert plain.layer_val_err == []


def test_layer_val_err_is_the_best_neuron_loss_on_the_split(tmp_path):
    cfg = _cfg(tmp_path, max_layer_count=1)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    dl, val = _make_dl(64), _make_dl(24)
    trained = trainer.train(model, dl, dl, dl, verbose=False, val_dl=val)
    layer = trained.layers[0]
    x = torch.cat([b[0] for b in val]); y = torch.cat([b[1] for b in val])
    with torch.no_grad():
        out = trained(x)
        per_col = torch.stack([trained.loss_fn(out[:, j], y).mean() for j in range(out.shape[1])])
    assert abs(per_col.min().item() - layer.val_err) < 1e-5
    # the dev figure is the same statistic on the dev rows
    x = torch.cat([b[0] for b in dl]); y = torch.cat([b[1] for b in dl])
    with torch.no_grad():
        out = trained(x)
        dev_best = torch.stack([trained.loss_fn(out[:, j], y).mean() for j in range(out.shape[1])]).min().item()
    assert abs(dev_best - layer.err) < 1e-4


def test_stop_source_val_routes_the_growth_rule_and_the_end_to_end_stop(tmp_path, monkeypatch):
    from torchsonn.trainer import GrowthCriterion
    cfg = _cfg(tmp_path, max_layer_count=2, stop_source="val")
    seen = []
    real = GrowthCriterion.update

    def spy(self, layer_index, err):
        seen.append(float(err))
        return real(self, layer_index, err)

    monkeypatch.setattr(GrowthCriterion, "update", spy)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    dl, val = _make_dl(64), _make_dl(24)
    trained = trainer.train(model, dl, dl, dl, verbose=False, val_dl=val)
    assert seen == [float(v) for v in trained.layer_val_err]
    assert any(abs(l.val_err - l.err) > 1e-6 for l in trained.layers)   # the two splits do differ
    # the end-to-end pass reads val for its stop and never touches dev
    cfg2 = OmegaConf.merge(cfg, OmegaConf.create({"train": {"finetune_train": {"max_steps": 30, "eval_interval": 10}}}))
    trainer.config = cfg2
    trainer.train_finetune(trained, dl, _NoIter(dl), val_dl=val)


def test_stop_source_val_needs_a_loader_and_rejects_unknown(tmp_path):
    dl = _make_dl(32)
    cfg = _cfg(tmp_path, max_layer_count=1, stop_source="val")
    with pytest.raises(ValueError, match="val_dl"):
        Trainer(config=cfg).train(SONN(cfg, d_model=4), dl, dl, dl, verbose=False)
    cfg = _cfg(tmp_path, max_layer_count=1, stop_source="test")
    with pytest.raises(ValueError, match="stop_source"):
        Trainer(config=cfg).train(SONN(cfg, d_model=4), dl, dl, dl, verbose=False)


# --- head column index cached on the device ---------------------------------

def test_head_columns_are_cached_and_invalidated_by_prune(tmp_path):
    cfg = OmegaConf.merge(
        _cfg(tmp_path, max_layer_count=2),
        OmegaConf.create({"model": {"use_output_projection": True, "num_out_neurons": 2}}),
    )
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    dl = _make_dl(64)
    trained = trainer.train(model, dl, dl, dl, verbose=False)
    trainer.train_out_proj(trained, dl, dl)
    last = trained.layers[-1]
    k = trained.out_proj.in_features
    cols = trained._head_columns(last, k)
    assert cols.dtype == torch.long and cols.device == trained.device
    assert cols.tolist() == trained._best_neuron_columns(last, k)
    assert trained._head_columns(last, k) is cols                      # cache hit
    x = torch.randn(6, 4)
    with torch.inference_mode():
        before = trained.infer(x)
    trainer.prune(trained)
    after_cols = trained._head_columns(trained.layers[-1], k)
    assert after_cols is not cols                                       # invalidated
    assert after_cols.tolist() == trained._best_neuron_columns(trained.layers[-1], k)
    with torch.inference_mode():
        assert torch.allclose(trained.infer(x), before, atol=1e-6)



# --- End-to-end pass: static-batch CUDA graph, batch sources, lr sync -------

from torchsonn.trainer import _FinetuneStep, _finetune_batches


def _trained_head_model(tmp_path, n=64, bs=8, **over):
    cfg = OmegaConf.merge(
        _cfg(tmp_path, max_layer_count=1, batch_size=bs),
        OmegaConf.create({"model": {"use_output_projection": True, "num_out_neurons": 2}, **over}),
    )
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    dl = _make_dl(n)
    dl = DataLoader(dl.dataset, batch_size=bs, shuffle=True)
    trainer.train(model, dl, dl, dl, verbose=False)
    trainer.train_out_proj(model, dl, dl)
    return cfg, model, trainer, dl


def test_finetune_batches_resident_visits_every_row_once_per_epoch_with_a_tail():
    x = torch.arange(50, dtype=torch.float32).unsqueeze(1).repeat(1, 4)
    y = torch.arange(50, dtype=torch.float32)
    dl = DataLoader(SONNDataset(x, y), batch_size=8, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen, desc = _finetune_batches(dl, torch.device(device), "true" if device == "cuda" else "false", None)
    for epoch in range(2):
        seen, shapes = [], []
        while True:
            bx, by = next(gen)
            if bx is None:
                break
            assert torch.equal(bx[:, 0].cpu(), by.cpu())
            seen.append(by.cpu()); shapes.append(bx.shape[0])
        rows = torch.cat(seen)
        assert sorted(rows.tolist()) == list(range(50))            # every row once
        assert shapes == [8] * 6 + [2]                               # a tail batch of 2
        if epoch == 0:
            first = rows.clone()
        else:
            assert not torch.equal(first, rows)                      # reshuffled per epoch
    if device == "cuda":
        assert "resident" in desc


def test_finetune_batches_streamed_matches_the_loader():
    x = torch.randn(30, 4); y = torch.randn(30)
    dl = DataLoader(SONNDataset(x, y), batch_size=7, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen, desc = _finetune_batches(dl, torch.device(device), "false", None)
    out = []
    while True:
        bx, by = next(gen)
        if bx is None:
            break
        out.append(bx.cpu())
    assert torch.equal(torch.cat(out), x) and "streamed" in desc


def test_finetune_step_eager_path_trains_and_counts(tmp_path):
    cfg, model, trainer, dl = _trained_head_model(tmp_path)
    model.to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stepper = _FinetuneStep(model, opt, model.loss_fn, Trainer._finetune_prediction, use_graph=False)
    x = torch.cat([b[0] for b in dl]); y = torch.cat([b[1] for b in dl])
    first = None
    for _ in range(30):
        stepper.step(x, y)
        if first is None:
            first = stepper.drain()
    last = stepper.drain()
    assert stepper.eager_steps == 30 and stepper.graph_steps == 0 and stepper.n_steps == 0
    assert last < first


def test_finetune_lr_tensor_follows_the_plateau_scheduler(tmp_path):
    """Force a plateau and check the lr the optimizer group holds halves and
    stays a tensor (the object the captured step reads) when a tensor lr is
    in use; on CPU the pass uses a float lr and the scheduler as before."""
    cfg, model, trainer, dl = _trained_head_model(tmp_path)
    device = model.device
    use_graph = torch.cuda.is_available()
    if use_graph:
        model.to("cuda")
        for layer in model.layers:
            for nm in layer.neuron_models:
                nm.to("cuda")
    lr = torch.tensor(1e-3, device=model.device) if use_graph else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, capturable=use_graph)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=0, min_lr=1e-6)
    for _ in range(3):
        sched.step(1.0)                                   # no improvement -> reduce each time
        group = opt.param_groups[0]
        if use_graph and not isinstance(group["lr"], torch.Tensor):
            lr.fill_(float(group["lr"])); group["lr"] = lr
    got = float(opt.param_groups[0]["lr"]) if not use_graph else float(lr.item())
    assert abs(got - 1e-3 * 0.5 ** 2) < 1e-9 or abs(got - 1e-3 * 0.5 ** 3) < 1e-9
    if use_graph:
        assert opt.param_groups[0]["lr"] is lr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_finetune_graph_matches_eager_and_handles_the_tail(tmp_path):
    """Two identical models: one trained by replayed graphs over 8-row
    buffers (with a 2-row tail batch each epoch), one eagerly on the same
    batches. Same loss trajectory to 1e-4; lr change is picked up."""
    cfg, model, trainer, dl = _trained_head_model(tmp_path, n=50, bs=8)
    import copy
    twin = copy.deepcopy(model)
    for m in (model, twin):
        m.to("cuda")
        for layer in m.layers:
            for nm in layer.neuron_models:
                nm.to("cuda")
    torch.manual_seed(0)
    x = torch.randn(50, 4, device="cuda"); y = torch.randn(50, device="cuda")
    batches = [(x[i:i + 8], y[i:i + 8]) for i in range(0, 50, 8)]        # last one has 2 rows
    lr_g = torch.tensor(1e-3, device="cuda")
    opt_g = torch.optim.AdamW(model.parameters(), lr=lr_g, capturable=True)
    opt_e = torch.optim.AdamW(twin.parameters(), lr=1e-3)
    g = _FinetuneStep(model, opt_g, model.loss_fn, Trainer._finetune_prediction, use_graph=True)
    e = _FinetuneStep(twin, opt_e, twin.loss_fn, Trainer._finetune_prediction, use_graph=False)
    # the graph path runs 1 rehearsal + 3 warm-up steps on the first batch: do the same eagerly
    for _ in range(4):
        e.step(*batches[0])
    for epoch in range(4):
        for bx, by in batches:
            g.step(bx, by); e.step(bx, by)
        assert abs(g.drain() - e.drain()) < 1e-4, epoch
    assert g.graph is not None and g.capture_error is None
    assert g.graph_steps == 4 * 6 and g.eager_steps == 4 + 4 * 1          # rehearsal, warm-ups, tails
    for pg, pe in zip(model.parameters(), twin.parameters()):
        assert torch.allclose(pg, pe, atol=1e-4)
    # an lr change in place reaches the captured step
    before = [p.detach().clone() for p in model.parameters()]
    lr_g.fill_(0.0)
    g.step(*batches[0])
    for p, b in zip(model.parameters(), before):
        assert torch.allclose(p, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_train_finetune_uses_the_graph_on_cuda_and_falls_back_when_told(tmp_path, caplog):
    cfg, model, trainer, dl = _trained_head_model(tmp_path, n=64, bs=16)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"train": {"device": "cuda", "finetune_train": {
        "optimizer": "adamw", "max_steps": 40, "eval_interval": 10, "early_stop_patience": 100}}}))
    model.to("cuda")
    for layer in model.layers:
        for nm in layer.neuron_models:
            nm.to("cuda")
    trainer.config = cfg
    x = torch.randn(5, 4, device="cuda")
    with caplog.at_level("INFO", logger="torchsonn.trainer"):
        trainer.train_finetune(model, dl, dl, cfg=cfg.train.finetune_train)
    assert any("CUDA graph captured" in r.message for r in caplog.records)
    assert any("steps replayed" in r.message for r in caplog.records)
    with torch.inference_mode():
        assert torch.isfinite(model.infer(x)).all()
    caplog.clear()
    cfg_e = OmegaConf.merge(cfg, OmegaConf.create({"train": {"finetune_train": {"cuda_graph": False}}}))
    trainer.config = cfg_e
    with caplog.at_level("INFO", logger="torchsonn.trainer"):
        trainer.train_finetune(model, dl, dl, cfg=cfg_e.train.finetune_train)
    assert any("eager steps" in r.message for r in caplog.records)
    assert not any("CUDA graph captured" in r.message for r in caplog.records)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_finetune_graph_falls_back_when_the_capture_fails(tmp_path, caplog):
    """A prediction that syncs with the host cannot be captured; the
    rehearsal under the sync debug mode flags it before any capture is
    attempted (a failed capture would poison the CUDA context for the rest
    of the process), and the stepper keeps training eagerly."""
    cfg, model, trainer, dl = _trained_head_model(tmp_path, n=64, bs=16)
    model.to("cuda")
    for layer in model.layers:
        for nm in layer.neuron_models:
            nm.to("cuda")

    def syncing_pred(m, x):
        out = Trainer._finetune_prediction(m, x)
        float(out.sum().item())        # host sync: not permitted during capture
        return out

    opt = torch.optim.AdamW(model.parameters(), lr=torch.tensor(1e-3, device="cuda"), capturable=True)
    stepper = _FinetuneStep(model, opt, model.loss_fn, syncing_pred, use_graph=True)
    x = torch.randn(16, 4, device="cuda"); y = torch.randn(16, device="cuda")
    for _ in range(5):
        stepper.step(x, y)
    assert stepper.capture_error and "synchroniz" in stepper.capture_error.lower()
    assert stepper.graph is None and stepper.graph_steps == 0 and stepper.eager_steps == 5
    assert math.isfinite(stepper.drain())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_headless_model_captures(tmp_path):
    """The best-neuron readout (no head) used a host-side lookup per call;
    cached, it captures like the head path."""
    cfg = _cfg(tmp_path, max_layer_count=1, batch_size=16)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    dl = _make_dl(64)
    trainer.train(model, dl, dl, dl, verbose=False)
    assert model.out_proj is None
    model.to("cuda")
    for layer in model.layers:
        for nm in layer.neuron_models:
            nm.to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=torch.tensor(1e-3, device="cuda"), capturable=True)
    stepper = _FinetuneStep(model, opt, model.loss_fn, Trainer._finetune_prediction, use_graph=True)
    x = torch.randn(16, 4, device="cuda"); y = torch.randn(16, device="cuda")
    for _ in range(5):
        stepper.step(x, y)
    assert stepper.capture_error is None and stepper.graph is not None and stepper.graph_steps >= 2


# --- The best evaluated weights (best_weights_copy) --------------------------

import torchsonn.trainer as _trainer_mod
from torchsonn.trainer import _BestWeights


@pytest.mark.parametrize("where", ["device", "cpu", "disk"])
def test_best_weights_keeps_the_lowest_loss_and_restores_in_place(tmp_path, where):
    """Strictly the lowest loss wins (NaN never does); the restore writes into
    the same tensor, and the disk copy is gone afterwards."""
    p = torch.nn.Parameter(torch.zeros(3))
    best = _BestWeights([p], True, where, tmp_path / "best_x.ckpt", "x")
    for step, loss in [(1, 2.0), (2, 1.0), (3, float("nan")), (4, 1.0), (5, 1.5)]:
        with torch.no_grad():
            p.fill_(step)
        best.update(step, loss)
    ptr = p.data_ptr()
    best.restore()
    assert torch.equal(p.detach(), torch.full((3,), 2.0)) and p.data_ptr() == ptr
    assert (best.best_step, best.best_loss, best.last_step) == (2, 1.0, 5)
    assert best.final_loss == 1.0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("where", ["device", "cpu", "disk"])
def test_best_weights_off_copies_nothing_and_ends_on_the_last_step(tmp_path, where):
    p = torch.nn.Parameter(torch.zeros(3))
    best = _BestWeights([p], False, where, tmp_path / "best_x.ckpt", "x")
    for step, loss in [(1, 2.0), (2, 1.0), (3, 1.5)]:
        with torch.no_grad():
            p.fill_(step)
        best.update(step, loss)
        assert not list(tmp_path.iterdir())
    best.restore()
    assert torch.equal(p.detach(), torch.full((3,), 3.0))
    assert (best.best_loss, best.final_loss) == (1.0, 1.5)
    # With no parameters (a head over frozen survivors) the lowest loss is final.
    head_only = _BestWeights([], False, where, None, "x")
    for step, loss in [(1, 2.0), (2, 1.0), (3, 1.5)]:
        head_only.update(step, loss)
    assert head_only.final_loss == 1.0


def test_best_weights_rejects_an_unknown_place(tmp_path):
    with pytest.raises(ValueError, match="best_weights_copy"):
        _BestWeights([], True, "gpu", None, "x")
    for block in ("out_proj_train", "finetune_train"):
        cfg = OmegaConf.merge(_cfg(tmp_path), OmegaConf.create(
            {"train": {block: {"best_weights_copy": "gpu"}}}))
        with pytest.raises(ValueError, match=f"train.{block}.best_weights_copy"):
            Trainer(config=cfg)


def test_best_weights_disk_file_per_rank(tmp_path, monkeypatch):
    """In a distributed run every rank writes its own file."""
    cfg = OmegaConf.merge(_cfg(tmp_path), OmegaConf.create(
        {"train": {"finetune_train": {"best_weights_copy": "disk", "keep_best_weights": True}}}))
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    p = torch.nn.Parameter(torch.zeros(2))
    monkeypatch.setattr(Trainer, "_is_dist", staticmethod(lambda: True))
    monkeypatch.setattr(_trainer_mod.dist, "get_rank", lambda: 3)
    best = trainer._best_weights(model, [p], cfg.train.finetune_train, "finetune", "finetune")
    assert best.path.name == "best_finetune_rank3.ckpt"
    assert best.path.parent == trainer.run_dir


class _ScriptedBest(_BestWeights):
    """Records the parameters at every evaluation and reports a scripted loss
    in place of the real one: the second evaluation is the lowest and every
    later one higher, so a pass must end on the second evaluation's
    parameters whatever its real losses. The patience counter still reads
    the real losses, so the pass runs as configured."""
    made: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []
        _ScriptedBest.made.append(self)

    def update(self, step, loss):
        self.seen.append((step, [p.detach().cpu().clone() for p in self.params]))
        super().update(step, 0.0 if len(self.seen) == 2 else float(len(self.seen)))


@pytest.fixture
def scripted_best(monkeypatch):
    _ScriptedBest.made = []
    monkeypatch.setattr(_trainer_mod, "_BestWeights", _ScriptedBest)
    return _ScriptedBest.made


def _pass_settings(optimizer, max_steps=6, eval_interval=1, keep=True, **extra):
    s = {"optimizer": optimizer, "max_steps": max_steps, "eval_interval": eval_interval,
         "early_stop_patience": 100, "lr": 0.05}
    if keep is not None:          # None: leave keep_best_weights at its default
        s["keep_best_weights"] = keep
    if optimizer == "lbfgs":
        # One plain iteration per outer step, so the head keeps moving.
        s.update({"lbfgs_max_iter": 1, "lbfgs_line_search": "", "lr": 0.5})
    s.update(extra)
    return s


def _headed_model(tmp_path, out_proj_train=None, finetune_train=None):
    train = {}
    if out_proj_train:
        train["out_proj_train"] = out_proj_train
    if finetune_train:
        train["finetune_train"] = finetune_train
    cfg = OmegaConf.merge(
        _cfg(tmp_path, max_layer_count=1),
        OmegaConf.create({"model": {"use_output_projection": True, "num_out_neurons": 2},
                          "train": train}),
    )
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    dl = _make_dl(64)
    trainer.train(model, dl, dl, dl, verbose=False)
    return cfg, model, trainer, dl


def _assert_ends_on_evaluation(best, params, i):
    assert len(best.seen) >= 3
    kept = best.seen[i][1]
    assert any(not torch.equal(a, b) for a, b in zip(best.seen[1][1], best.seen[-1][1])), \
        "the parameters never moved after the second evaluation"
    for p, s in zip(params, kept):
        assert torch.equal(p.detach().cpu(), s)


@pytest.mark.parametrize("which", ["out_proj_adam", "out_proj_lbfgs", "layer_finetune", "finetune"])
def test_passes_end_on_their_last_step_by_default(tmp_path, scripted_best, which):
    """keep_best_weights is off by default: nothing is copied, the pass ends
    on its last evaluated step, and the readout error is that step's loss."""
    defaults = OmegaConf.structured(SONNConfig).train
    assert defaults.out_proj_train.keep_best_weights is False
    assert defaults.finetune_train.keep_best_weights is False
    settings = _pass_settings("lbfgs" if which == "out_proj_lbfgs" else "adam", keep=None,
                              best_weights_copy="disk")
    if which == "finetune":
        cfg, model, trainer, dl = _headed_model(tmp_path, finetune_train=settings)
    else:
        cfg, model, trainer, dl = _headed_model(tmp_path, out_proj_train=settings)
    scripted_best.clear()
    if which == "finetune":
        trainer.train_finetune(model, dl, dl)
        params = list(model.parameters())
    elif which == "layer_finetune":
        readout_err = trainer._train_layer_finetune(model, model.layers[-1], dl, dl)
        params = None
    else:
        trainer.train_out_proj(model, dl, dl)
        params = list(model.out_proj.parameters())
    (best,) = scripted_best
    assert not best.keep and best.path is None
    if params is None:
        params = best.params
        assert readout_err == float(len(best.seen))     # the last evaluation's scripted loss
    _assert_ends_on_evaluation(best, params, -1)
    assert not list(trainer.run_dir.glob("best_*"))


def _assert_ends_on_second_evaluation(best, params):
    assert len(best.seen) >= 3
    second = best.seen[1][1]
    assert any(not torch.equal(a, b) for a, b in zip(best.seen[-1][1], second)), \
        "the parameters never moved after the second evaluation"
    for p, s in zip(params, second):
        assert torch.equal(p.detach().cpu(), s)


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_head_fit_ends_on_its_best_evaluation(tmp_path, scripted_best, optimizer):
    cfg, model, trainer, dl = _headed_model(tmp_path, out_proj_train=_pass_settings(optimizer))
    scripted_best.clear()
    trainer.train_out_proj(model, dl, dl)
    (best,) = scripted_best
    _assert_ends_on_second_evaluation(best, list(model.out_proj.parameters()))
    saved = torch.load(trainer.run_dir / "model_last.ckpt", weights_only=False)["model"]
    assert torch.equal(saved["out_proj.weight"], model.out_proj.weight.detach().cpu())


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_layer_finetune_ends_on_its_best_evaluation(tmp_path, scripted_best, optimizer):
    """The survivors end on the best evaluation's weights, and the readout
    error the pass returns is that evaluation's loss."""
    cfg, model, trainer, dl = _headed_model(tmp_path, out_proj_train=_pass_settings(optimizer))
    layer = model.layers[-1]
    scripted_best.clear()
    readout_err = trainer._train_layer_finetune(model, layer, dl, dl)
    (best,) = scripted_best
    assert readout_err == 0.0
    survivors = [p for nm in layer.neuron_models for name, p in nm.named_parameters()
                 if name not in ("proj_weight", "proj_bias")]
    assert [id(p) for p in best.params] == [id(p) for p in survivors]
    _assert_ends_on_second_evaluation(best, survivors)
    for p in model.parameters():
        assert p.requires_grad


def test_layer_finetune_with_frozen_neurons_copies_nothing(tmp_path, scripted_best):
    cfg, model, trainer, dl = _headed_model(
        tmp_path, out_proj_train=_pass_settings("adam", best_weights_copy="disk"))
    layer = model.layers[-1]
    before = [p.detach().clone() for nm in layer.neuron_models for p in nm.parameters()]
    scripted_best.clear()
    readout_err = trainer._train_layer_finetune(model, layer, dl, dl, freeze_neurons=True)
    (best,) = scripted_best
    assert best.params == [] and readout_err == 0.0
    after = [p.detach() for nm in layer.neuron_models for p in nm.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(after, before))
    assert not list(trainer.run_dir.glob("best_*"))


@pytest.mark.parametrize("where", ["device", "cpu", "disk"])
def test_end_to_end_pass_ends_on_its_best_evaluation_and_saves(tmp_path, scripted_best, where):
    cfg, model, trainer, dl = _headed_model(
        tmp_path, finetune_train=_pass_settings("adamw", lr=1e-2, best_weights_copy=where))
    scripted_best.clear()
    trainer.train_finetune(model, dl, dl)
    (best,) = scripted_best
    _assert_ends_on_second_evaluation(best, list(model.parameters()))
    saved = torch.load(trainer.run_dir / "model_last.ckpt", weights_only=False)["model"]
    for k, v in model.state_dict().items():
        if torch.is_tensor(v):
            assert torch.equal(saved[k], v.cpu()), k
    assert not list(trainer.run_dir.glob("best_*"))


@pytest.mark.parametrize("which", ["out_proj_adam", "out_proj_lbfgs", "layer_finetune", "finetune"])
def test_a_last_step_between_evaluations_is_evaluated(tmp_path, scripted_best, which):
    settings = _pass_settings("lbfgs" if which == "out_proj_lbfgs" else "adam",
                              max_steps=7, eval_interval=5)
    if which == "finetune":
        cfg, model, trainer, dl = _headed_model(tmp_path, finetune_train=settings)
    else:
        cfg, model, trainer, dl = _headed_model(tmp_path, out_proj_train=settings)
    scripted_best.clear()
    if which == "finetune":
        trainer.train_finetune(model, dl, dl)
    elif which == "layer_finetune":
        trainer._train_layer_finetune(model, model.layers[-1], dl, dl)
    else:
        trainer.train_out_proj(model, dl, dl)
    (best,) = scripted_best
    assert [step for step, _ in best.seen] == [5, 7]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("where", ["device", "cpu", "disk"])
def test_end_to_end_graph_pass_restores_in_place(tmp_path, scripted_best, where, caplog):
    """Under a captured CUDA graph the restore copies into the tensors the
    graph holds: same storage, best values, and the graph still trains them."""
    cfg, model, trainer, dl = _headed_model(
        tmp_path, finetune_train=_pass_settings("adamw", max_steps=9, lr=1e-2,
                                                best_weights_copy=where))
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"train": {"device": "cuda"}}))
    model.to("cuda")
    for layer in model.layers:
        for nm in layer.neuron_models:
            nm.to("cuda")
    trainer.config = cfg
    ptrs = [p.data_ptr() for p in model.parameters()]
    scripted_best.clear()
    with caplog.at_level("INFO", logger="torchsonn.trainer"):
        trainer.train_finetune(model, dl, dl, cfg=cfg.train.finetune_train)
    assert any("CUDA graph captured" in r.message for r in caplog.records)
    assert any("kept the weights of step" in r.message for r in caplog.records)
    (best,) = scripted_best
    _assert_ends_on_second_evaluation(best, list(model.parameters()))
    assert [p.data_ptr() for p in model.parameters()] == ptrs
    assert not list(trainer.run_dir.glob("best_*"))
