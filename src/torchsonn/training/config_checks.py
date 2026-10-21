"""Configuration checks run when a `Trainer` is built, before any data is
read or a run folder is made."""
import os
from pathlib import Path
from typing import Any

import torch

from torchsonn.optimizers import optimizer_map
from torchsonn.training import selection

# Where `best_weights_copy` keeps the best weights of the head and
# fine-tune passes.
BEST_WEIGHTS_COPIES = ("device", "cpu", "disk")


def validate_config(config: Any) -> None:
    """Run every check below; each one returns quietly when its section is missing."""
    _validate_selection_method(config)
    _validate_layer_err_source(config)
    _validate_best_weights_copy(config)
    _validate_optimizer_name(config)
    _validate_checkpoint_dir(config)
    _validate_joint_selection(config)
    _validate_layer_keep(config)


def checkpoint_root(config: Any) -> Path:
    """`train.checkpoint_dir` of `config`, taken from the working directory when relative.

    Raises `ValueError` when it is empty: `Path("")` is the working
    directory itself, so the run folders would land loose in it.
    """
    checkpoint_dir = str(config.train.checkpoint_dir)
    if checkpoint_dir == "":
        raise ValueError(
            "train.checkpoint_dir is empty; set the folder that holds the run folders "
            "(the default is 'checkpoints', in the working directory)."
        )
    return Path(checkpoint_dir)


def apply_deterministic(config: Any) -> None:
    """Switch PyTorch to deterministic kernels under `train.use_deterministic_algorithms`.

    Process-wide, like `set_seed`. cuBLAS gives repeatable results only
    with `CUBLAS_WORKSPACE_CONFIG` set, and raises without it once the
    flag is on, so it is set to `:4096:8` when absent; a value already set
    is kept. When the key is false nothing changes, so a flag the caller
    switched on stays on.
    """
    train_cfg = getattr(config, "train", None) if config is not None else None
    if not bool(getattr(train_cfg, "use_deterministic_algorithms", False)):
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def _validate_selection_method(config: Any) -> None:
    """Reject a `train.neuron_selection_method` that `selection.SELECTION_METHODS` does not hold.

    Checked when the Trainer is built, so a wrong name fails before any data
    is read, instead of when the first layer's candidates are selected.
    """
    train_cfg = getattr(config, "train", None) if config is not None else None
    method = getattr(train_cfg, "neuron_selection_method", None)
    if method is None or str(method) in selection.SELECTION_METHODS:
        return
    names = [repr(m) for m in selection.SELECTION_METHODS]
    raise ValueError(
        f"train.neuron_selection_method={str(method)!r}; expected {', '.join(names[:-1])} or {names[-1]}."
    )


def _validate_layer_err_source(config: Any) -> None:
    """Reject `train.layer_err_source` settings the rest of the config cannot honor.

    'readout' scores a layer by the dev loss of a head fitted over its
    survivors (the `layer_finetune` head when that pass is on, otherwise a
    temporary head over the frozen survivors), so it only makes sense when
    inference goes through a head (`use_output_projection`); otherwise the
    criterion would optimize depth for a readout the model never uses.
    Multi-class is left out until measured.
    """
    # Some callers (checkpoint-housekeeping tests, tooling) build a Trainer
    # with no config or a partial one; there is nothing to validate then.
    train_cfg = getattr(config, "train", None) if config is not None else None
    model_cfg = getattr(config, "model", None) if config is not None else None
    if train_cfg is None or model_cfg is None:
        return
    source = str(getattr(train_cfg, "layer_err_source", "neuron"))
    if source not in ("neuron", "readout"):
        raise ValueError(
            f"train.layer_err_source={source!r}; expected 'neuron' or 'readout'."
        )
    if source != "readout":
        return
    if not bool(model_cfg.use_output_projection):
        raise ValueError(
            "train.layer_err_source='readout' requires model.use_output_projection=true: "
            "without an output head, inference reads the single best neuron, so the "
            "growth criterion must score that neuron (layer_err_source='neuron')."
        )
    if str(model_cfg.type) == "multi-class":
        raise NotImplementedError(
            "train.layer_err_source='readout' is implemented for regressor / binary "
            "models only."
        )


def _validate_best_weights_copy(config: Any) -> None:
    """Reject an unknown `best_weights_copy` or a bad `dropout` in `train.out_proj_train` or
    `train.finetune_train`, and a negative `train.layer_finetune_weight_decay`."""
    train_cfg = getattr(config, "train", None) if config is not None else None
    if train_cfg is None:
        return
    for block in ("out_proj_train", "finetune_train"):
        where = str(getattr(getattr(train_cfg, block, None), "best_weights_copy", "device"))
        if where not in BEST_WEIGHTS_COPIES:
            raise ValueError(
                f"train.{block}.best_weights_copy={where!r}; expected 'device', 'cpu' or 'disk'."
            )
        block_cfg = getattr(train_cfg, block, None)
        p = float(getattr(block_cfg, "dropout", 0.0))
        if not 0.0 <= p < 1.0:
            raise ValueError(f"train.{block}.dropout={p}; expected a value in [0, 1).")
        if p > 0.0 and str(getattr(block_cfg, "optimizer", "adam")) == "lbfgs":
            raise ValueError(
                f"train.{block}.dropout={p} needs a minibatch optimizer: LBFGS needs a "
                "deterministic objective."
            )
    wd = getattr(train_cfg, "layer_finetune_weight_decay", None)
    if wd is not None and float(wd) < 0.0:
        raise ValueError(f"train.layer_finetune_weight_decay={wd}; expected a value of at least 0.")


def _validate_layer_keep(config: Any) -> None:
    """Reject a `train.layer_keep` other than 'best' or 'all'."""
    train_cfg = getattr(config, "train", None) if config is not None else None
    if train_cfg is None:
        return
    keep = str(getattr(train_cfg, "layer_keep", "best"))
    if keep not in ("best", "all"):
        raise ValueError(f"train.layer_keep={keep!r}; expected 'best' or 'all'.")


def _validate_optimizer_name(config: Any) -> None:
    """Reject a `train.optimizer.name` that `optimizer_map` does not hold.

    Checked when the Trainer is built, so a wrong name fails before any
    data is read or a run folder made, instead of as a `KeyError` at the
    first candidate fit.
    """
    train_cfg = getattr(config, "train", None) if config is not None else None
    name = getattr(getattr(train_cfg, "optimizer", None), "name", None)
    if name is None or str(name) in optimizer_map:
        return
    names = [repr(k) for k in optimizer_map]
    raise ValueError(
        f"train.optimizer.name={str(name)!r}; expected {', '.join(names[:-1])} or {names[-1]}."
    )


def _validate_checkpoint_dir(config: Any) -> None:
    """Reject an empty `train.checkpoint_dir` when the Trainer is built, before any run folder is made."""
    train_cfg = getattr(config, "train", None) if config is not None else None
    if getattr(train_cfg, "checkpoint_dir", None) is not None:
        checkpoint_root(config)


def _validate_joint_selection(config: Any) -> None:
    """Reject a `model.selection_mode` / `train.joint_sparsity` setting the joint mode cannot honor."""
    train_cfg = getattr(config, "train", None) if config is not None else None
    model_cfg = getattr(config, "model", None) if config is not None else None
    js = getattr(train_cfg, "joint_sparsity", None)
    if js is None:
        return
    kind = str(js.kind)
    if kind not in ("gates", "group_lasso"):
        raise ValueError(f"train.joint_sparsity.kind={kind!r}; expected 'gates' or 'group_lasso'.")
    if not (float(js.gamma) < 0.0 and float(js.zeta) > 1.0):
        raise ValueError(
            f"train.joint_sparsity needs gamma < 0 and zeta > 1, got gamma={js.gamma}, zeta={js.zeta}."
        )
    if not float(js.beta) > 0.0:
        raise ValueError(f"train.joint_sparsity.beta={js.beta}; expected a positive temperature.")
    if not 0.0 < float(js.init_open) < 1.0:
        raise ValueError(f"train.joint_sparsity.init_open={js.init_open}; expected a value in (0, 1).")
    control = str(js.lam_control)
    if control not in ("fixed", "anneal", "adaptive"):
        raise ValueError(
            f"train.joint_sparsity.lam_control={control!r}; expected 'fixed', 'anneal' or 'adaptive'."
        )
    prefilter = str(js.prefilter)
    choices = ("none", *selection.SELECTION_METHODS)
    if prefilter not in choices:
        names = [repr(c) for c in choices]
        raise ValueError(
            f"train.joint_sparsity.prefilter={prefilter!r}; expected {', '.join(names[:-1])} or {names[-1]}."
        )
    for key, ok, what in (
        ("lam", float(js.lam) >= 0.0, "a value >= 0"),
        ("lam_warmup_frac", 0.0 <= float(js.lam_warmup_frac) <= 1.0, "a value in [0, 1]"),
        ("diversity_lam", float(js.diversity_lam) >= 0.0, "a value >= 0"),
        ("diversity_tau", 0.0 <= float(js.diversity_tau) < 1.0, "a value in [0, 1)"),
        ("prune_threshold", float(js.prune_threshold) >= 0.0, "a value >= 0"),
        ("prefilter_mult", float(js.prefilter_mult) >= 1.0, "a value >= 1"),
        ("lr", float(js.lr) > 0.0, "a positive value"),
        ("gate_lr", float(js.gate_lr) > 0.0, "a positive value"),
        ("weight_decay", float(js.weight_decay) >= 0.0, "a value >= 0"),
        ("max_steps", int(js.max_steps) >= 1, "at least 1"),
        ("eval_interval", int(js.eval_interval) >= 1, "at least 1"),
        ("lam_step", float(js.lam_step) > 1.0, "a value above 1"),
        ("count_margin", float(js.count_margin) >= 0.0, "a value >= 0"),
        ("count_patience", int(js.count_patience) >= 1, "at least 1"),
    ):
        if not ok:
            raise ValueError(f"train.joint_sparsity.{key}={getattr(js, key)}; expected {what}.")
    if model_cfg is None:
        return
    mode = str(getattr(model_cfg, "selection_mode", "greedy"))
    if mode not in ("greedy", "joint", "joint_last"):
        raise ValueError(f"model.selection_mode={mode!r}; expected 'greedy', 'joint' or 'joint_last'.")
    if mode == "greedy":
        return
    if not bool(model_cfg.use_output_projection) or str(model_cfg.type) == "binary":
        raise ValueError(
            f"model.selection_mode={mode!r} trains the pool through the output head: it needs "
            "model.use_output_projection=true and a regressor or multi-class model."
        )
    if bool(model_cfg.use_layer_norm):
        raise ValueError(
            f"model.selection_mode={mode!r} removes neurons from a trained layer, which changes "
            "what a LayerNorm over the layer computes; it does not work with model.use_layer_norm."
        )
