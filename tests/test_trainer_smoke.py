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


def test_fit_layer_squash_matches_actual_layer_inputs(tmp_path):
    """A deeper layer's inputs are the frozen prefix's outputs concatenated
    with the shortcut originals — a different width and scale from the raw
    input, which is what the per-layer (not global) calibration exists for."""
    cfg = _legendre_cfg(tmp_path)
    model = SONN(cfg, d_model=4)
    trainer = Trainer(config=cfg)
    train_dl = _offset_dl(64, seed=1)

    layer0 = model.create_layer(0)
    model.layers.append(layer0)
    trainer.fit_layer_squash(model, layer0, train_dl)
    trainer.train_layer(model, layer0, train_dl, _offset_dl(24, seed=2), None)

    layer1 = model.create_layer(1)
    model.layers.append(layer1)
    neuron = layer1.neuron_models[0]
    assert neuron.num_feat == layer0.d_model + model.d_model

    before = neuron.squash_norm.mean.clone()
    trainer.fit_layer_squash(model, layer1, train_dl)

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


def test_fit_layer_squash_skipped_without_sigma_neurons(tmp_path):
    """tanh needs no statistics, so the extra pass must not run at all."""
    cfg = _legendre_cfg(tmp_path, squash_method="tanh")
    model = SONN(cfg, d_model=4)
    layer = model.create_layer(0)
    model.layers.append(layer)

    def _explode(*args, **kwargs):
        raise AssertionError("calibration pass ran for a tanh-squashed layer")

    Trainer(config=cfg).fit_layer_squash(model, layer, _explode)


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
    Trainer(config=cfg).fit_layer_squash(model, layer, dl)
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
