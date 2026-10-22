import gc
import logging
import math
import random
import sys
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchsonn.layer import SONNLayer
from torchsonn.model import SONN, _linear_from_seed
from torchsonn.neurons import BasePolynomNeuron
from torchsonn.optimizers import optimizer_map
from torchsonn.training import (
    config_checks, distributed, ensemble, evaluation, finetune, head_fit, joint_selection, keep_best,
    normalization, pruning, runs, selection,
)
from torchsonn.training.state import GrowthCriterion, LayerAccumulator, StepCheckpoint
from torchsonn.utils import abbrev_floats, fmt_err, timed_block

logger = logging.getLogger(__name__)


class Trainer:
    """
    Grows and trains a SONN, layer by layer.

    `train` runs the structural search: it creates a layer of candidate
    neurons, calibrates them on the layer's inputs, fits all candidates in
    parallel and keeps the best ones (`train_layer`), and adds layers until
    the growth rule (`GrowthCriterion`) stops it. The layers after the best
    one are dropped and the model is saved. After
    that, `train_out_proj` fits the optional output head, `train_finetune`
    trains every parameter end to end, `prune` removes the neurons that do
    not reach the output and `infer` predicts over a loader.

    Every `train` call writes to a run folder of its own, `run_dir`:
    `<train.checkpoint_dir>/<YYYY-MM-DD-HH-MM-SS>/`. It holds the step
    checkpoints written as training goes, `model_last.ckpt` and the run's
    `train.log`. `load_model_checkpoint` restores a saved model.

    Parameters
    ----------
    config : SONNConfig-like
        The full configuration (`model` and `train` sections).
    batch_callback : callable, optional
        Maps a loader batch to an `(x, y)` pair. Default: batches already
        are `(x, y)`.
    feature_names : list of str, optional
        Feature names, used in logs and model descriptions.
    class_weights : tensor, optional
        Per-class loss weights, passed to the model at the start of
        `train`.
    """
    def __init__(
        self,
        config: Any,
        batch_callback: Callable[[Any], tuple[torch.Tensor, torch.Tensor]] | None = None,
        feature_names: list[str] | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.feature_names = feature_names
        self.batch_callback = batch_callback
        self.config = config
        self.class_weights = class_weights
        self._run_dir: Path | None = None
        config_checks.validate_config(config)
        config_checks.apply_deterministic(config)

    @classmethod
    def set_seed(cls, seed: int = 42) -> None:
        """Seed `random`, NumPy and PyTorch (CPU and every GPU), and make cuDNN deterministic."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # =======================================
    # region Checkpoints
    # =======================================
    @property
    def run_dir(self) -> Path | None:
        """The current run folder, `<checkpoint_root>/<YYYY-MM-DD-HH-MM-SS>`.

        `train` creates a new one at the start of every run, or reopens the
        run it resumes, and `load_model_checkpoint` sets it to the run it
        loads from. The run's step checkpoints, `model_last.ckpt` and
        `train.log` live there. None before the first run or load.
        """
        return self._run_dir

    @property
    def checkpoint_root(self) -> Path:
        """The folder that holds the run folders: `train.checkpoint_dir` of the trainer's configuration, `checkpoints` in the working directory by default."""
        return config_checks.checkpoint_root(self.config)

    @classmethod
    def run_folders(cls, root: str | Path) -> list[Path]:
        """
        List the run folders in `root`, oldest first.

        A run folder's name is its start time, `YYYY-MM-DD-HH-MM-SS`, with a
        `-<n>` suffix for runs started in the same second. They sort by the
        time, then by the suffix as a number. Other entries are left out.

        Parameters
        ----------
        root : str or Path
            The folder to look in, usually `checkpoint_root`.

        Returns
        -------
        list of Path
            The run folders; empty when `root` does not exist.
        """
        return runs.run_folders(root)

    def _ensure_run_dir(self, model: SONN) -> Path:
        """The current run folder, or a new one when the trainer has none yet."""
        if self._run_dir is None:
            self._run_dir = runs.new_run_dir(config_checks.checkpoint_root(model.param))
        return self._run_dir

    @classmethod
    def parse_checkpoint_step(cls, fname: str) -> tuple[int, int, int]:
        """
        Parse a step checkpoint file name.

        Parameters
        ----------
        fname : str
            File name of the form
            `model_layer_<L>_neuron_<N>_step_<S>[_last].ckpt`.

        Returns
        -------
        key : tuple of int
            `(layer, neuron_model, step)`, or `(-1, -1, -1)` when the name
            does not match, so that such files sort below every real
            checkpoint.
        """
        return runs.parse_checkpoint_step(fname)

    def from_checkpoint(self, run_dir: str | Path | None = None) -> dict[str, Any] | None:
        """
        Load the latest step checkpoint of a run, for resuming `train`.

        The latest one is the highest `(layer, neuron_model, step)` among the
        run folder's step checkpoints, `_last` copies included. Other files
        in the folder, such as `train.log` and `model_last.ckpt`, are
        ignored.

        Parameters
        ----------
        run_dir : str or Path, optional
            The run folder. Default: `run_dir`.

        Returns
        -------
        checkpoint : dict or None
            The checkpoint contents, or None if there is no run folder or it
            holds no step checkpoint.
        """
        folder = Path(run_dir) if run_dir is not None else self._run_dir
        if folder is None:
            return None
        return runs.latest_step_checkpoint(folder)

    def save_checkpoint(self, ckpt: StepCheckpoint, suffix: str = "") -> None:
        """
        Save a step checkpoint in the run folder and delete the older ones.

        Writes `model_layer_<L>_neuron_<N>_step_<S>[_<suffix>].ckpt` with the
        snapshot and the model config, then keeps the `train.keep_last_n`
        most recent step checkpoints of the run folder (see
        `cleanup_checkpoints`).

        Parameters
        ----------
        ckpt : StepCheckpoint
            The training state to save.
        suffix : str
            Optional suffix for the file name, such as "last". Default "".
        """
        runs.save_step_checkpoint(self._ensure_run_dir(ckpt.model), ckpt, suffix)

    def load_model_checkpoint(self, model: SONN, device: torch.device | str = "cpu",
                              run: str | Path | None = None) -> None:
        """
        Restore a model saved as `model_last.ckpt`.

        Reads it from a run folder, rebuilds the layer stack, loads the
        weights and moves the model to `device`. The run folder becomes
        `run_dir`, so a later `save_model_checkpoint` writes to the same run.

        Parameters
        ----------
        model : SONN
            The model to restore into, built with the same config.
        device : torch.device or str
            Device to move the restored model to. Default "cpu".
        run : str or Path, optional
            The run folder, by its name in `checkpoint_root` or by its path.
            Default: the trainer's current run, or, for a trainer that has
            none, the newest run folder in `checkpoint_root` that holds a
            `model_last.ckpt`.

        Raises
        ------
        FileNotFoundError
            If there is no such run folder, or no run folder holds a
            `model_last.ckpt`.
        """
        run_dir = runs.model_run_dir(config_checks.checkpoint_root(model.param), run, self._run_dir)
        runs.load_model(model, run_dir, device)
        self._run_dir = run_dir

    def save_model_checkpoint(self, model: SONN) -> None:
        """Save the model's state dict as `model_last.ckpt` in the run folder (a new one if the trainer has none yet)."""
        runs.save_model(model, self._ensure_run_dir(model))

    def cleanup_checkpoints(self, checkpoint_dir: str | Path, keep_last_n: int = 10) -> None:
        """
        Delete a run folder's oldest step checkpoints, keeping the `keep_last_n` most recent.

        Only the files named `model_layer_<L>_neuron_<N>_step_<S>.ckpt`
        count, ordered by `parse_checkpoint_step`. Everything else in the
        folder (`model_last.ckpt`, the `_last` copies, `train.log`, other
        files and subfolders) is left alone.

        Parameters
        ----------
        checkpoint_dir : str or Path
            The run folder.
        keep_last_n : int
            Number of most recent step checkpoints to keep. Default 10.
        """
        runs.cleanup_checkpoints(Path(checkpoint_dir), keep_last_n)

    def cleanup_layer_checkpoints(self, layer_idx: int) -> None:
        """
        Delete every step checkpoint of one layer from the run folder, `_last` copies included.

        Parameters
        ----------
        layer_idx : int
            The layer whose checkpoints are deleted.
        """
        if self._run_dir is None:
            return
        runs.cleanup_layer_checkpoints(self._run_dir, layer_idx)
    # endregion

    # =======================================
    # region Distributed helpers
    # =======================================
    @classmethod
    def init_distributed(cls, backend: str = "nccl") -> None:
        """Initialize torch.distributed for multi-GPU / multi-node training.

        Call once before Trainer.train(), then launch with torchrun:
            torchrun --nproc_per_node=<N> --nnodes=<M> script.py

        Sets LOCAL_RANK as the active CUDA device so each rank uses a
        distinct GPU within the node.
        """
        distributed.init_distributed(backend)
    # endregion

    # =======================================
    # region Train
    # =======================================
    def train(
        self,
        model: SONN,
        train_dl: DataLoader,
        dev_dl: DataLoader,
        test_dl: DataLoader,
        verbose: bool = True,
        resume: bool | str = False,
        val_dl: DataLoader | None = None,
    ) -> SONN:
        """
        Grow and train the network, one layer at a time.

        Each call is a run with a folder of its own, `run_dir`: a new
        `<train.checkpoint_dir>/<YYYY-MM-DD-HH-MM-SS>/`, or the folder of the
        run it resumes. The run's step checkpoints, `model_last.ckpt` and
        `train.log` (every record of the root logger from the start of the
        run until the next run starts) are written there.

        Each iteration creates a layer of candidate neurons, calibrates them
        on the layer's inputs, trains them and keeps the best ones
        (`train_layer`). Growth stops when `GrowthCriterion` sees no
        improvement for `train.criterion_minimum_width` layers, or at
        `train.max_layer_count`. The layers after the one with the smallest
        error are then dropped, and the model is saved as `model_last.ckpt`
        in the run folder. The output head and the end-to-end fine-tune are
        separate steps (`train_out_proj`, `train_finetune`).

        With `model.selection_mode: joint` each layer's survivors are
        selected jointly instead, by `train_joint`.

        Parameters
        ----------
        model : SONN
            The model to train. Its existing layers are discarded.
        train_dl : DataLoader
            Training split: fits the neuron coefficients.
        dev_dl : DataLoader
            Dev split: scores and selects the neurons, and drives the growth
            rule unless `train.stop_source` is 'val'.
        test_dl : DataLoader
            Not used during training.
        verbose : bool
            Log the time each layer takes. Default True.
        resume : bool or str
            False starts a new run. True continues the newest run folder in
            `checkpoint_root` that holds a step checkpoint, from its latest
            one, and starts a new run when there is none. A run folder's
            name continues that run. Default False.
        val_dl : DataLoader, optional
            Validation split that selects nothing. Its error is logged per
            layer, and it drives the growth rule when `train.stop_source` is
            'val'.

        Returns
        -------
        model : SONN
            The trained model (the same object).
        """
        if self.class_weights is not None:
            model.set_class_weights(self.class_weights)

        mode = str(model.param.model.selection_mode)
        joint = mode == "joint"
        if mode != "greedy" and resume:
            raise ValueError(f"model.selection_mode={mode!r} does not resume a run; start a new one (resume=False).")

        growth = GrowthCriterion(
            width=int(self.config.train.criterion_minimum_width),
            epsilon=float(self.config.train.stop_train_epsilon_condition),
            min_delta=float(getattr(self.config.train, "stop_train_min_delta", 0.0)),
        )
        stop_source = evaluation.stop_source(self.config, val_dl)
        self._run_dir, resumed = runs.start_run(config_checks.checkpoint_root(model.param), resume)
        model.layers = nn.ModuleList()
        model.layer_val_err = []
        checkpoint_data = None
        # Per-layer snapshots of projection weights (multi-class only).
        # Each train_layer call overwrites model.shared_proj / model.neuron_proj
        # with the best neuron's state; we snapshot after each layer so we can
        # restore whichever layer achieved the minimum error before final save.
        shared_proj_states: dict[int, dict] = {}

        # Fix the NormMSE denominator to the training-set target spread before
        # any layer trains. Re-measured on resume too — it describes the data,
        # not the checkpoint.
        normalization.fit_target_scale(model, train_dl, self.config.train.device, batch_callback=self.batch_callback)

        if joint:
            return self.train_joint(model, train_dl, dev_dl, verbose=verbose, val_dl=val_dl)

        if resumed:
            # load checkpoint
            checkpoint_data = self.from_checkpoint()
            if checkpoint_data is not None:
                # restore model architecture
                model.restore_from_checkpoint_metadata(checkpoint_data["model"])
                # load weights
                model.load_state_dict(checkpoint_data["model"], strict=False)
                model.to(device=self.config.train.device)

        while True:
            # create layer, calculate all possible neurons and then select the best ones
            # using specified criterion
            if checkpoint_data is not None and not checkpoint_data["layer_completed"]:
                layer_index = len(model.layers) - 1
                layer = model.layers[-1]
            else:
                # create new layer with neurons
                layer_index = len(model.layers)
                layer = model.create_layer(layer_index)
                model.layers.append(layer)
                checkpoint_data = None
                # Calibrate the neurons on the layer's inputs before it trains
                # (squash statistics, RBF centers). Only on the freshly-created
                # path: a layer resumed mid-training keeps the state its
                # checkpoint carries, which was fit on this same training set
                # before it started.
                normalization.fit_layer_inputs(model, layer, train_dl, batch_callback=self.batch_callback)

            with timed_block(f"train layer #{layer_index}", verbose=verbose):
                self.train_layer(model, layer, train_dl, dev_dl, checkpoint_data)
                checkpoint_data = None

            # Safety net for the CUDA caching allocator — train_layer's own
            # cleanup is the primary release point, this flushes anything
            # remaining (e.g. allocator fragments) before the next layer
            # starts allocating ensemble params and precomputed features.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if model.shared_proj is not None:
                # Snapshot to CPU — keeping clones on GPU would grow residency
                # by one shared_proj per layer and contribute to OOM as depth
                # increases. Restored back to device at end of train() below.
                shared_proj_states[layer_index] = {
                    k: v.detach().cpu().clone() for k, v in model.shared_proj.state_dict().items()
                }

            # The validation split, when given, selects nothing: it is reported
            # next to the dev error (the gap is the selection bias, per layer)
            # and, under stop_source 'val', is what the growth rule reads.
            stop_err = layer.err
            if val_dl is not None:
                val_err = evaluation.layer_val_err(model, layer, val_dl, batch_callback=self.batch_callback)
                layer.val_err = val_err
                model.layer_val_err.append(val_err)
                logger.info(f"Layer #{layer.layer_index}: dev {fmt_err(layer.err)} | val {fmt_err(val_err)} "
                            f"(best neuron on each split; gap {val_err - layer.err:+.4f})")
                if stop_source == "val":
                    stop_err = val_err
            # proceed until the stop rule fires (see GrowthCriterion) or the
            # number of layers reaches the limit
            error_stopped_decrease = growth.update(layer.layer_index, stop_err)
            logger.info(growth.describe(layer.layer_index, stop_err))
            if error_stopped_decrease or not (layer.layer_index < self.config.train.max_layer_count - 1):
                break
        kept_index = growth.kept_index(getattr(self.config.train, "layer_keep", "best"), len(model.layers))

        model.layer_err.clear()
        for i in range(0, len(model.layers)):
            model.layer_err.append(model.layers[i].err)

        # delete the layers after kept_index
        for layer_idx in range(kept_index + 1, len(model.layers)):
            self.cleanup_layer_checkpoints(layer_idx)
        del model.layers[kept_index + 1:]

        for layer in model.layers:
            layer.set_neuron_module()

        # Restore the projection that was co-trained with the last kept layer.
        # Each train_layer call overwrites model.shared_proj in the shared-proj path;
        # without this restore the checkpoint contains the projection from the
        # last trained layer (a different input space than layers[kept_index]).
        if model.shared_proj is not None and kept_index in shared_proj_states:
            model.shared_proj.load_state_dict(shared_proj_states[kept_index])

        if mode == "joint_last":
            self._joint_last_layer(model, train_dl, dev_dl, val_dl, verbose)

        self.save_model_checkpoint(model)

        return model

    def train_layer(
            self,
            model: SONN,
            layer: SONNLayer,
            train_dl: DataLoader,
            dev_dl: DataLoader,
            checkpoint_data: dict[str, Any] | None,
            nbest: int | None = None,
            selection_method: str | None = None,
            joint_pool: bool = False,
    ) -> None:
        """Calculate neuron weights

        `nbest` and `selection_method` override `model.nbest_neurons` and
        `train.neuron_selection_method` for this layer's selection; the
        joint selection mode uses them to keep its pool. `joint_pool` skips
        the per-layer fine-tune and the readout head (`layer_finetune`,
        `layer_err_source: readout`): the joint pass takes their place.
        """

        accumulator = LayerAccumulator()
        last_ckpt: StepCheckpoint | None = None
        device = layer.neuron_models[0].device
        logger.info(f"Training layer #{layer.layer_index}")
        # Maps original neuron_model_idx → trained params_batch so we can restore
        # the best neuron's projection (shared_proj or neuron_proj) after
        # neuron selection.  Keyed by the index used during the training loop.
        trained_params_per_model: dict[int, dict] = {}

        # Optionally precompute frozen-layer features once so the inner training
        # loop does not re-run O(num_layers) previous layers on every gradient
        # step. Controlled by train.precompute_features (default False = keep
        # the historical per-step forward). Precompute is faster but caches
        # (N_samples, d_layer) floats per split and can spike CPU RAM at
        # deeper layers.
        precompute = bool(model.param.train.precompute_features)
        if precompute:
            train_feat_dl = evaluation.precompute_dl(model, train_dl, device, batch_callback=self.batch_callback)
            dev_feat_dl   = evaluation.precompute_dl(model, dev_dl,   device, batch_callback=self.batch_callback)
        else:
            train_feat_dl = train_dl
            dev_feat_dl   = dev_dl

        # A neuron model whose fit finished before the stop but whose error
        # was not recorded yet: the end-of-fit checkpoint is written before
        # the error. The resume skips its fit and computes the rest.
        fitted_neuron_model_idx = None
        if checkpoint_data is not None:
            accumulator.err = checkpoint_data["err"]
            accumulator.module_idxs = checkpoint_data["module_idxs"]
            loaded_neuron_model_idx = checkpoint_data["neuron_model_idx"]
            # Every neuron model before the loaded one finished with its
            # error recorded; the loaded one either stopped partway through
            # its fit, which resumes from the checkpoint, or finished it.
            start_neuron_model_idx = loaded_neuron_model_idx
            if checkpoint_data["neuron_model_completed"]:
                fitted_neuron_model_idx = loaded_neuron_model_idx

            # The checkpoint restores `err` and `module_idxs` but not the dev
            # responses `z_val`, which are too large to save. Recompute them
            # for the neuron models the resume skips, so selection sees one
            # response per restored error. Their weights came back with the
            # model state, so the responses are the same as before the stop.
            for done_idx in range(start_neuron_model_idx):
                z_val, y_val = selection.compute_z_val(
                    layer.neuron_models[done_idx],
                    evaluation.iter_eval_batches(model, dev_feat_dl, device, precompute, batch_callback=self.batch_callback))
                accumulator.z_val.append(z_val)
                if accumulator.y_val is None:
                    accumulator.y_val = y_val
            if start_neuron_model_idx:
                logger.info("resume: recomputed the dev responses of %d trained neuron model(s)",
                            start_neuron_model_idx)
        else:
            start_neuron_model_idx = 0

        for neuron_model_idx, neuron_model in enumerate(layer.neuron_models[start_neuron_model_idx:]):
            neuron_model_idx += start_neuron_model_idx

            regularity_err = None
            bias_err = None

            # Every criterion fits the layer's own modules on the whole
            # training split, with the early stop on dev: they are the
            # survivors selection keeps. The bias criterion's two half-fits
            # below only score the candidates.
            loss_fn_vmapped, eval_loss_fn_vmapped, pred_fn_vmapped, params_batch, buffers_batch, shared_param_names = (
                ensemble.create_loss_functions(model, neuron_model))
            if neuron_model_idx == fitted_neuron_model_idx:
                # Fitted before the stop: params_batch holds the trained
                # weights. The loaded checkpoint stands in for the one the fit
                # would have returned, so the layer's end is still saved.
                trained_params = params_batch
                last_ckpt = StepCheckpoint(model=model, **{
                    f.name: checkpoint_data.get(f.name) for f in fields(StepCheckpoint) if f.name != "model"})
            else:
                accumulator, trained_params, last_ckpt = self.train_model_ensemble(
                    model,
                    neuron_model,
                    neuron_model_idx,
                    layer.layer_index,
                    train_feat_dl,
                    dev_feat_dl,
                    checkpoint_data,
                    accumulator,
                    loss_fn_vmapped,
                    eval_loss_fn_vmapped,
                    params_batch,
                    buffers_batch,
                    shared_param_names,
                    features_precomputed=precompute)
            # The checkpoint resumed this fit; every later fit, the bias
            # criterion's half fits included, starts fresh.
            checkpoint_data = None
            trained_params_per_model[neuron_model_idx] = trained_params
            if model.need_regularity_err:
                regularity_err = evaluation.regularity_err(model, pred_fn_vmapped, trained_params, buffers_batch, dev_feat_dl, device, skip_model_fwd=precompute, batch_callback=self.batch_callback)
            # Cache the per-candidate raw scalar outputs on the full dev
            # split for neuron_selection's OMP decorrelation.
            z_val, y_val = selection.compute_z_val(
                neuron_model, evaluation.iter_eval_batches(model, dev_feat_dl, device, precompute, batch_callback=self.batch_callback))
            accumulator.z_val.append(z_val)
            if accumulator.y_val is None:
                accumulator.y_val = y_val

            if model.need_bias_err:
                neuron_module_a = deepcopy(neuron_model)
                neuron_module_b = deepcopy(neuron_model)

                train_dl_a = evaluation.split_loader(train_feat_dl, 0)
                train_dl_b = evaluation.split_loader(train_feat_dl, 1)
                dev_dl_a   = evaluation.split_loader(dev_feat_dl,   0)
                dev_dl_b   = evaluation.split_loader(dev_feat_dl,   1)

                loss_fn_vmapped_a, eval_loss_fn_vmapped_a, pred_fn_vmapped_a, params_batch_a, buffers_batch_a, shared_param_names_a = (
                    ensemble.create_loss_functions(model, neuron_model))
                accumulator, trained_params_a, last_ckpt = self.train_model_ensemble(
                    model,
                    neuron_module_a,
                    neuron_model_idx,
                    layer.layer_index,
                    train_dl_a,
                    dev_dl_a,
                    checkpoint_data,
                    accumulator,
                    loss_fn_vmapped_a,
                    eval_loss_fn_vmapped_a,
                    params_batch_a,
                    buffers_batch_a,
                    shared_param_names_a,
                    features_precomputed=precompute)

                loss_fn_vmapped_b, eval_loss_fn_vmapped_b, pred_fn_vmapped_b, params_batch_b, buffers_batch_b, shared_param_names_b = (
                    ensemble.create_loss_functions(model, neuron_model))
                accumulator, trained_params_b, last_ckpt = self.train_model_ensemble(
                    model,
                    neuron_module_b,
                    neuron_model_idx,
                    layer.layer_index,
                    train_dl_b,
                    dev_dl_b,
                    checkpoint_data,
                    accumulator,
                    loss_fn_vmapped_b,
                    eval_loss_fn_vmapped_b,
                    params_batch_b,
                    buffers_batch_b,
                    shared_param_names_b,
                    features_precomputed=precompute)
                bias_err = evaluation.bias_err(
                    model,
                    pred_fn_vmapped_a,
                    pred_fn_vmapped_b,
                    trained_params_a,
                    trained_params_b,
                    buffers_batch_a,
                    buffers_batch_b,
                    train_dl_a,
                    train_dl_b,
                    device,
                    bias_method=model.param.train.bias_ce_type,
                    skip_model_fwd=precompute,
                    batch_callback=self.batch_callback)

            module_err = model.get_error(model.criterion_type, regularity_err, bias_err)
            accumulator.module_idxs.append(
                torch.stack([
                   torch.empty(module_err.shape[0], dtype=torch.long, device=device).fill_(neuron_model_idx),
                   torch.arange(module_err.shape[0], device=device)],
                dim=1)
            )
            accumulator.err.append(module_err)

        err_values, module_idxs = selection.neuron_selection(
            model, layer, accumulator, trained_params_per_model, device,
            nbest=nbest, method=selection_method,
        )

        # Streaming dev-set diagnostics on the survivor pool. O(D²) memory —
        # never materializes the full (N_val, D) Z matrix, so it scales from
        # N_val=10k to 10M+ unchanged. Gated by train.log_layer_diagnostics
        # because it still adds one full dev forward per layer.
        if model.param.train.log_layer_diagnostics:
            model.eval()
            N_val = 0
            sum_z: torch.Tensor | None = None    # (D,)
            sum_zz: torch.Tensor | None = None   # (D, D)
            with torch.inference_mode():
                for batch in dev_dl:
                    x_inp, _ = self.batch_callback(batch) if self.batch_callback else batch
                    # float64 accumulation — D is small so the cost is negligible
                    # and it keeps the second moment numerically stable at N_val≫1e5.
                    z = model(x_inp.to(device=device)).double()
                    if sum_z is None:
                        sum_z  = z.sum(dim=0)
                        sum_zz = z.T @ z
                    else:
                        sum_z  = sum_z  + z.sum(dim=0)
                        sum_zz = sum_zz + z.T @ z
                    N_val += z.size(0)

            assert sum_z is not None and sum_zz is not None  # for type checker
            mean = sum_z / N_val
            cov  = sum_zz / N_val - torch.outer(mean, mean)
            var  = torch.diagonal(cov).clamp_min(1e-12)
            corr = cov / torch.sqrt(torch.outer(var, var))

            off_mask = ~torch.eye(corr.size(0), dtype=torch.bool, device=corr.device)
            off = corr.abs()[off_mask]
            layer_idx = layer.layer_index
            logger.info(
                f"[layer {layer_idx}] correlation off-diagonal — "
                f"median |ρ|: {off.median().item():.3f}, max: {off.max().item():.3f}"
            )

            # Singular values of (Z - mean): symmetric PSD second-moment matrix
            # → eigvalsh; flip for descending order to match np.linalg.svd.
            M = sum_zz - N_val * torch.outer(mean, mean)
            eigvals = torch.linalg.eigvalsh(M).clamp_min(0)
            S = torch.sqrt(eigvals).flip(0)
            top10 = S[:10].cpu().numpy()
            eff_rank = int((S > 0.01 * S[0]).sum().item())
            logger.info(f"[layer {layer_idx}] top 10 singular values: {top10}")
            logger.info(f"[layer {layer_idx}] effective rank (>1% of top): {eff_rank}")

        layer.err_values = err_values
        layer.module_idxs = module_idxs
        if model.param.train.layer_err_criterion == 'top':
            # err_values is sorted by module_idx (not by error), so take the
            # minimum explicitly instead of trusting position 0.
            neuron_err = err_values.min().item()
        elif model.param.train.layer_err_criterion == 'avg':
            neuron_err = err_values.mean().item()
        else:
            raise NotImplementedError
        layer.err = neuron_err

        # Optional per-layer joint fine-tune (surviving neurons' polynomial
        # weights + temporary (D, K) head trained end-to-end on the model's
        # loss). Runs BEFORE the layer-completed checkpoint save so the
        # persisted state captures the fine-tuned weights. Returns the dev
        # loss of the weights it leaves (with frozen neurons, the temporary
        # head's lowest), which under `layer_err_source: readout` becomes the
        # layer error the growth criterion in train() compares: it measures
        # the readout a headed model is scored on, whereas the best-neuron
        # error above describes survivors the fine-tune has just turned into a
        # basis (individually worse, jointly better).
        readout_err: float | None = None
        if joint_pool:
            pass
        elif model.param.train.layer_finetune:
            readout_err = finetune.train_layer_finetune(
                model, layer, train_dl, dev_dl, batch_callback=self.batch_callback,
                run_dir=lambda: self._ensure_run_dir(model))
        elif model.param.train.layer_err_source == "readout":
            # No fine-tune: fit the temporary head over the frozen survivors,
            # purely to measure what a head reads out of this layer. The
            # neurons keep their calibrated per-neuron fits, so the search
            # grows exactly as under 'neuron'; only the stop decision changes.
            readout_err = finetune.train_layer_finetune(
                model, layer, train_dl, dev_dl, freeze_neurons=True, batch_callback=self.batch_callback,
                run_dir=lambda: self._ensure_run_dir(model))
        if readout_err is not None:
            if model.param.train.layer_err_source == "readout":
                if not (readout_err < float("inf")):
                    raise RuntimeError(
                        f"layer {layer.layer_index}: the per-layer head has no finite dev loss "
                        "(out_proj_train.max_steps is 0, or no evaluation gave a finite loss), "
                        "so there is no readout error."
                    )
                layer.err = readout_err
                logger.info(f"Layer readout error (per-layer head on dev): {fmt_err(readout_err)}; "
                            f"best-neuron error: {fmt_err(neuron_err)}")
        logger.info(f"Current layer error: {fmt_err(layer.err)}")
        logger.info(f"Layer errors: {abbrev_floats([l.err for l in model.layers], fmt=fmt_err)}")

        if last_ckpt is not None:
            layer_end_ckpt = replace(last_ckpt, layer_completed=True,
                err=list(accumulator.err), module_idxs=list(accumulator.module_idxs))
            self.save_checkpoint(layer_end_ckpt)
            del layer_end_ckpt

        # Release per-layer GPU state before the next layer's precompute_dl
        # runs — otherwise trained_params_per_model (full ensemble params for
        # every candidate neuron model) and the optimizer state inside
        # last_ckpt stay resident in the CUDA caching allocator and stack up
        # layer over layer until OOM.
        del trained_params_per_model, accumulator, last_ckpt
        del train_feat_dl, dev_feat_dl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_model_ensemble(
            self,
            model: SONN,
            neuron_model: BasePolynomNeuron,
            neuron_model_idx: int,
            layer_idx: int,
            train_dl: DataLoader,
            dev_dl: DataLoader,
            checkpoint_data: dict[str, Any] | None,
            accumulator: LayerAccumulator,
            loss_fn_vmapped: Callable,
            eval_loss_fn_vmapped: Callable,
            params_batch: dict[str, torch.Tensor],
            buffers_batch: dict[str, torch.Tensor],
            shared_param_names: list[str],
            features_precomputed: bool = False,
    ) -> tuple[LayerAccumulator, dict[str, torch.Tensor], "StepCheckpoint | None"]:
        """
        Train one neuron module's candidate ensemble in parallel.

        Runs the configured batched optimizer over the stacked parameters of
        all candidates, with per-candidate early stopping and learning-rate
        drops (on the dev loss, or on the training loss when
        `train.early_stop_source` is 'train'). Step checkpoints are saved
        every `train.save_interval` steps and at the end, and the trained
        parameters are written back to the module. Under distributed
        training each rank trains its own slice of the ensemble.

        Parameters
        ----------
        model : SONN
            The model being grown.
        neuron_model : BasePolynomNeuron
            The neuron module whose candidates are trained.
        neuron_model_idx : int
            Index of that module in the layer.
        layer_idx : int
            Index of the layer.
        train_dl, dev_dl : DataLoader
            Training and dev splits.
        checkpoint_data : dict or None
            Step checkpoint to resume from, or None to start fresh.
        accumulator : LayerAccumulator
            The layer's running state, stored in the checkpoints.
        loss_fn_vmapped, eval_loss_fn_vmapped : callable
            Gradient and evaluation functions from `create_loss_functions`.
        params_batch, buffers_batch : dict of str to tensor
            Stacked candidate parameters and buffers.
        shared_param_names : list of str
            Parameters shared by all candidates (the shared projection).
        features_precomputed : bool
            The loaders already yield the layer's input features, so the
            model forward is skipped. Default False.

        Returns
        -------
        accumulator : LayerAccumulator
            The accumulator passed in.
        params_batch : dict of str to tensor
            The trained candidate parameters.
        last_checkpoint : StepCheckpoint
            The snapshot of the finished ensemble, as saved.
        """

        device = neuron_model.device
        ensemble_size = neuron_model.ensemble_size

        lr = torch.empty(ensemble_size, dtype=torch.float, device=device).fill_(model.param.train.optimizer.optimizer_params.lr)
        min_lr = torch.empty(1, dtype=torch.float, device=device).fill_(model.param.train.optimizer.optimizer_params.min_lr)
        optimizer_params = {k: v for k, v in model.param.train.optimizer.optimizer_params.items()}
        # min_lr / gamma are consumed by the trainer's LR-drop logic, not the
        # optimizer constructor, so strip them before forwarding kwargs.
        optimizer_params.pop("min_lr", None)
        optimizer_params.pop("gamma", None)
        optimizer_params["lr"] = lr
        shared_param_lr_multiplier = model.param.train.shared_proj_lr_multiplier
        optimizer_cls = optimizer_map[model.param.train.optimizer.name]

        # Ensemble-parallel sharding: each rank trains a disjoint slice of the
        # ensemble batch dimension so all N GPUs work in parallel.
        # Shared params (e.g. shared_proj) are NOT sliced — their gradients are
        # averaged across ranks via all_reduce after every forward-backward pass.
        dist_slice = distributed.ensemble_slice(ensemble_size)
        if dist_slice is not None:
            _local_start, _local_end = dist_slice
            _local_size = _local_end - _local_start
            _shared = set(shared_param_names)
            params_batch = {
                k: (v[_local_start:_local_end].clone() if k not in _shared else v)
                for k, v in params_batch.items()
            }
            buffers_batch = {
                k: (v[_local_start:_local_end].clone() if k not in _shared else v)
                for k, v in buffers_batch.items()
            }
            lr = lr[_local_start:_local_end]
            optimizer_params["lr"] = lr
        else:
            _local_start, _local_size = 0, ensemble_size

        early_stop_source = str(model.param.train.early_stop_source).lower()
        if early_stop_source not in ("dev", "train"):
            raise ValueError(
                f"train.early_stop_source must be 'dev' or 'train', got {model.param.train.early_stop_source!r}")
        opt = optimizer_cls(params_batch, shared_param_names, shared_param_lr_multiplier=shared_param_lr_multiplier, **optimizer_params)

        last_ckpt: StepCheckpoint | None = None

        start_step = 0
        cmp_tolerance = 1e-8
        global_step = 0
        eval_step = 0
        stop = False
        eval_step_interval = model.param.train.eval_step_interval

        start_from_scratch = (
                checkpoint_data is None or
                (checkpoint_data is not None and
                 checkpoint_data["neuron_model_completed"])
        )
        if start_from_scratch:
            # A finished fit's checkpoint has no step to resume from; kept, it
            # would make the loop below skip this fit's first batch.
            checkpoint_data = None
            epoch = 0
            best_val_losses = torch.full((_local_size,), float('inf'), device=device)
            last_steps_improve = torch.zeros(_local_size, dtype=torch.int, device=device)
            early_stop_flags = torch.zeros(_local_size, dtype=torch.bool, device=device)
            smoothed_val_losses = torch.zeros(_local_size, device=device)
        else:
            # restore module params; shard per-ensemble tensors for this rank
            epoch = checkpoint_data["epoch"]
            start_step = checkpoint_data["step"]
            global_step = checkpoint_data["global_step"]
            _s, _e = _local_start, _local_start + _local_size
            best_val_losses    = checkpoint_data["best_val_losses"][_s:_e]    if dist_slice else checkpoint_data["best_val_losses"]
            last_steps_improve = checkpoint_data["last_steps_improve"][_s:_e] if dist_slice else checkpoint_data["last_steps_improve"]
            early_stop_flags   = checkpoint_data["early_stop_flags"][_s:_e]   if dist_slice else checkpoint_data["early_stop_flags"]
            smoothed_val_losses = checkpoint_data["smoothed_val_losses"][_s:_e] if dist_slice else checkpoint_data["smoothed_val_losses"]
            lr = checkpoint_data["lr"][_s:_e] if dist_slice else checkpoint_data["lr"]
            opt.load_state_dict(checkpoint_data["opt"])

        train_dl_len = len(train_dl)
        step = 0
        # One persistent bar tracking the global step budget. The previous
        # design wrapped the inner DataLoader in tqdm, so with batch_size that
        # collapses an epoch to 1 batch (LBFGS / full-batch GD) every epoch
        # blasted the bar to 100% and printed a fresh line. A single
        # outer-level bar advanced by 1 per optimizer step gives a stable,
        # Lightning-style single-line readout regardless of batches per epoch.
        _rank0 = not distributed.is_dist() or dist.get_rank() == 0
        tbar = tqdm(
            total=int(model.param.train.steps),
            initial=int(global_step),
            file=sys.stdout,
            ncols=140,
            desc=f"Training {type(neuron_model).__name__}",
            dynamic_ncols=True,
            leave=True,
            disable=not model.param.train.optimizer.verbose or not _rank0,
        )
        while not stop:
            for step, batch in enumerate(train_dl):
                # Resume logic: the saved `start_step` is the index of the step
                # that was fully executed before checkpointing, so the next step
                # to run is start_step + 1. If start_step was already the final
                # step of the epoch, the whole epoch is done — break out and let
                # the outer while-loop advance to the next epoch (instead of the
                # previous behavior of silently replaying the whole epoch).
                if checkpoint_data is not None:
                    if start_step >= train_dl_len - 1:
                        checkpoint_data = None
                        break
                    if step <= start_step:
                        continue
                    checkpoint_data = None

                x_inp, targets = self.batch_callback(batch) if self.batch_callback else batch

                x_inp = x_inp.to(device=neuron_model.device)
                targets = targets.to(device=neuron_model.device)
                if features_precomputed:
                    x = x_inp
                else:
                    model.eval()
                    with torch.inference_mode():
                        x = model(x_inp, skip_last_layer=True)

                model.train()
                grads = loss_fn_vmapped(params_batch, buffers_batch, x, targets)
                grads = {k: g.detach() for k, g in grads.items()}
                # Synchronize shared-param gradients so every rank applies the same update.
                distributed.allreduce_shared_grads(grads, shared_param_names)
                params_batch = opt.step(params_batch, grads, active_mask=~early_stop_flags)

                if global_step > 0 and ((eval_step_interval != -1 and global_step % eval_step_interval == 0) or (eval_step_interval == -1 and step == train_dl_len - 1)):

                    model.eval()
                    with torch.inference_mode():
                        if early_stop_source == "train":
                            # The training loss of the current batch, per
                            # member; the dev split is left to selection.
                            val_losses = eval_loss_fn_vmapped(params_batch, buffers_batch, x, targets).detach()
                        else:
                            val_losses = evaluation.ds_loss(model, eval_loss_fn_vmapped, params_batch, buffers_batch, dev_dl, device, features_precomputed, batch_callback=self.batch_callback)

                    # Exclude models that have diverged. NaN is unrecoverable.
                    # The absolute val-loss threshold is configurable via
                    # train.divergence_threshold; default +inf disables it,
                    # which is what gmdhpy does — for normalized-but-not-
                    # clipped features the initial val_loss before the first
                    # optimizer step can be O(1e4) on heavy-tailed datasets
                    # and would otherwise wipe the whole ensemble in one shot.
                    divergence_threshold = float(model.param.train.divergence_threshold)
                    early_stop_flags = (
                        early_stop_flags
                        | (val_losses > divergence_threshold).bool()
                        | torch.isnan(val_losses)
                    )

                    # Exponential moving average smoothing
                    smoothing_factor = model.param.train.eval_smoothing_factor
                    if global_step > 1:
                        smoothed_val_losses = (
                            smoothing_factor * val_losses + (1 - smoothing_factor) * smoothed_val_losses
                        )
                    else:
                        # don't smooth at step 0 to avoid grop of values
                        smoothed_val_losses = val_losses

                    improved = smoothed_val_losses < (best_val_losses - model.param.train.early_stop_patience)
                    best_val_losses = torch.where(improved, smoothed_val_losses, best_val_losses)
                    last_steps_improve = torch.where(improved, eval_step, last_steps_improve)
                    steps_since_improve = eval_step - last_steps_improve

                    current_early_stop_flags = (steps_since_improve >= model.param.train.early_stop_tolerance_steps) & (lr <= min_lr + cmp_tolerance)
                    need_drop_lr = (steps_since_improve >= model.param.train.early_stop_tolerance_steps) & (lr > min_lr + cmp_tolerance)
                    last_steps_improve = torch.where(need_drop_lr, eval_step, last_steps_improve)
                    lr = torch.where(need_drop_lr, lr * model.param.train.optimizer.optimizer_params.gamma, lr)
                    # Push the LR drop into the optimizer so subsequent opt.step() uses it.
                    opt.lr = lr

                    # Only stop when tolerance exceeded
                    early_stop_flags = early_stop_flags | current_early_stop_flags
                    # early_stop_flags = early_stop_flags | current_early_stop_flags | val_losses > model.param.train.optimizer.optimizer_params.max_grad

                    # On-device stop condition (0-d bool tensors; completion-% is
                    # compared on device so no host sync yet). `completed` counts
                    # models stopped so far — including those that hit tolerance
                    # this very eval, since current_early_stop_flags was just OR'd
                    # in above. All-reduce across ranks when distributed so the
                    # decision reflects the full ensemble, not just the local slice.
                    completed = early_stop_flags.sum()
                    all_stopped = early_stop_flags.all()
                    if distributed.is_dist():
                        dist.all_reduce(completed, op=dist.ReduceOp.SUM)
                        _all_t = all_stopped.to(torch.uint8)
                        dist.all_reduce(_all_t, op=dist.ReduceOp.MIN)
                        all_stopped = _all_t.bool()
                    completion_stop = completed * 100 > float(model.param.train.early_stop_completion_percentage) * ensemble_size

                    # Progress-bar scalars are cosmetic and each needs a GPU->CPU
                    # .item(); refresh them only every eval_display_interval evals,
                    # and only when a bar is actually shown. nan-safe reductions
                    # avoid the data-dependent boolean-mask indexing (itself a sync);
                    # posinf/neginf are pinned so only NaN is rewritten and a
                    # diverged candidate still reads as inf, not FLT_MAX.
                    display_interval = max(1, int(model.param.train.eval_display_interval))
                    if not tbar.disable and eval_step % display_interval == 0:
                        val_loss_mean = val_losses.nanmean().item()
                        val_loss_min = torch.nan_to_num(val_losses, nan=float("inf"), posinf=float("inf"), neginf=float("-inf")).min().item()
                        val_loss_max = torch.nan_to_num(val_losses, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf")).max().item()
                        tbar.set_postfix({
                            "epoch": epoch,
                            "step": global_step,
                            "lr": f"{lr.mean().item():.2e}",
                            "val_loss": f"[{val_loss_min:.3f}, {val_loss_mean:.3f}, {val_loss_max:.3f}]",
                            "completed": f"{int(completed.item())}/{ensemble_size}",
                        }, refresh=True)

                    # The single correctness sync per eval: has the whole ensemble
                    # stopped? The step budget is checked after every step below.
                    if bool((all_stopped | completion_stop).item()):
                        # Snap total to the current step so the bar renders as
                        # 100% on the final line — otherwise an early-stop at
                        # step 80 of a 500-step budget leaves a permanent
                        # "Training: 16%" record. The console log goes through
                        # TqdmLoggingHandler so it lands above the bar
                        # without tearing the redraw.
                        tbar.total = tbar.n
                        tbar.refresh()
                        logger.info(
                            f"All models of {neuron_model.__class__.__name__} "
                            f"early stopped at step {global_step}"
                        )
                        stop = True
                        break
                    eval_step += 1
                    del val_losses

                global_step += 1
                tbar.update(1)

                # The step budget holds at any step, not only at evaluations:
                # a fit runs at most train.steps optimizer steps (counted
                # across a resume, like global_step). The write-back and the
                # completed checkpoint after the loop end it as usual.
                if global_step >= int(model.param.train.steps):
                    tbar.total = tbar.n
                    tbar.refresh()
                    logger.info(
                        f"{neuron_model.__class__.__name__} reached train.steps ({global_step})"
                    )
                    stop = True
                    break

                save_interval = model.param.train.save_interval
                if (save_interval != -1 and global_step % save_interval == 0) or (step == train_dl_len - 1 and not model.param.train.skip_saving_at_epoch_end):
                    _ckpt_params = distributed.gather_ensemble_params(params_batch, shared_param_names) if dist_slice else params_batch
                    ensemble.write_back_params(neuron_model, _ckpt_params)
                    ensemble.write_back_shared_params(model, _ckpt_params)
                    last_ckpt = StepCheckpoint(
                        model=model,
                        opt=opt,
                        layer_idx=layer_idx,
                        neuron_model_idx=neuron_model_idx,
                        epoch=epoch,
                        step=step,
                        global_step=global_step,
                        best_val_losses=best_val_losses,
                        smoothed_val_losses=smoothed_val_losses,
                        last_steps_improve=last_steps_improve,
                        early_stop_flags=early_stop_flags,
                        lr=lr,
                        neuron_model_completed=stop,
                        layer_completed=False,
                        err=list(accumulator.err),
                        module_idxs=list(accumulator.module_idxs),
                    )
                    if not distributed.is_dist() or dist.get_rank() == 0:
                        self.save_checkpoint(last_ckpt)

            epoch += 1
            if stop:
                break
        tbar.close()
        stats = getattr(opt, "stats", None)
        if stats and stats["updates"]:
            logger.info(
                "%s fit: %d optimizer steps; %.1f%% of updates capped at max_step, "
                "%.1f%% of curvature pairs rejected",
                neuron_model.__class__.__name__, stats["steps"],
                100.0 * stats["capped"] / stats["updates"],
                100.0 * stats["rejected"] / max(1, stats["pairs"]),
            )

        # Restore the full ensemble shape before write-back and error evaluation.
        params_batch = distributed.gather_ensemble_params(params_batch, shared_param_names) if dist_slice else params_batch
        ensemble.write_back_params(neuron_model, params_batch)
        ensemble.write_back_shared_params(model, params_batch)
        last_ckpt = StepCheckpoint(
            model=model,
            opt=opt,
            layer_idx=layer_idx,
            neuron_model_idx=neuron_model_idx,
            epoch=epoch,
            step=step,
            global_step=global_step,
            best_val_losses=best_val_losses,
            smoothed_val_losses=smoothed_val_losses,
            last_steps_improve=last_steps_improve,
            early_stop_flags=early_stop_flags,
            lr=lr,
            neuron_model_completed=True,
            layer_completed=False,
            err=list(accumulator.err),
            module_idxs=list(accumulator.module_idxs),
        )
        if not distributed.is_dist() or dist.get_rank() == 0:
            self.save_checkpoint(last_ckpt)
            if model.param.train.save_last_layer:
                self.save_checkpoint(last_ckpt, suffix="last")

        # Drop optimizer / gradient state before returning so the next neuron
        # model in the same layer starts from a clean slate. We cannot drop
        # params_batch (it is returned) or last_ckpt (it is returned).
        del opt, buffers_batch
        return accumulator, params_batch, last_ckpt

    # endregion

    def train_out_proj(self, model: SONN, train_dl: DataLoader, dev_dl: DataLoader) -> None:
        """Train model.out_proj with all other parameters frozen.

        Runs a standard PyTorch training loop — no vmapped ensemble, no custom
        optimizer — because out_proj is a single shared nn.Linear whose gradient
        comes from the full batch in one shot.

        Stops at cfg.max_steps, or after early_stop_patience evaluations in a
        row without an improvement of the dev loss. ReduceLROnPlateau lowers
        the learning rate when the dev loss stalls, down to lr_min. With
        `optimizer: lbfgs` the head is fitted full-batch instead, without the
        plateau schedule.

        The head ends on its last step, or with `cfg.keep_best_weights` on
        the weights of its lowest dev loss (a last step between two
        evaluations is evaluated too), copied back in place from where
        `cfg.best_weights_copy` keeps them. The model is then saved to the
        run folder's `model_last.ckpt`.
        """
        if model.out_proj is None:
            raise ValueError("model.out_proj is None; enable use_output_projection in the model config")

        cfg = model.param.train.out_proj_train
        device = model.device

        # Freeze everything, then unfreeze only out_proj.
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.out_proj.parameters():
            p.requires_grad_(True)

        # Logistic regression on top of frozen survivor outputs is convex —
        # LBFGS with a strong-Wolfe line search reaches the optimum in O(100)
        # full-batch closure evaluations, matching sklearn's solver. Branches
        # off here because it ignores the ReduceLROnPlateau schedule and runs
        # full-batch rather than the mini-batch loop used by Adam / SGD.
        if cfg.optimizer == "lbfgs":
            head_fit.train_out_proj_lbfgs(model, train_dl, dev_dl, cfg, batch_callback=self.batch_callback,
                                          run_dir=lambda: self._ensure_run_dir(model))
            for p in model.parameters():
                p.requires_grad_(True)
            # Same persistence as the Adam/SGD path — see note below tbar.close().
            self.save_model_checkpoint(model)
            return

        if cfg.optimizer == "adam":
            opt = torch.optim.Adam(
                model.out_proj.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "adamw":
            # Decoupled weight decay, as in train_finetune: under Adam the L2
            # term rides through the second-moment estimate and is rescaled
            # per parameter, under AdamW it is applied to the weights directly.
            opt = torch.optim.AdamW(
                model.out_proj.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "sgd":
            opt = torch.optim.SGD(
                model.out_proj.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9
            )
        else:
            raise ValueError(
                f"Unsupported out_proj optimizer: {cfg.optimizer!r} "
                "(supported: 'adam', 'adamw', 'sgd', 'lbfgs')."
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
            min_lr=cfg.lr_min,
            threshold=cfg.early_stop_min_delta,
        )

        # Column indices are fixed — err_values don't change after train_layer.
        k = model.out_proj.in_features
        cols = model._best_neuron_columns(model.layers[-1], k)
        cols_t = torch.tensor(cols, dtype=torch.long, device=device)

        # Precompute frozen SONN features once; the loop only trains the linear head.
        train_feat_dl = evaluation.precompute_dl(model, train_dl, device, skip_last_layer=False, batch_callback=self.batch_callback)
        dev_feat_dl   = evaluation.precompute_dl(model, dev_dl,   device, skip_last_layer=False, batch_callback=self.batch_callback)

        # Branch on model.type once so the inner loop's loss step doesn't.
        # Multi-class uses log_softmax + NLL; regression uses the raw scalar
        # output + whatever loss_fn was configured (NormMSE for the regressor
        # SONN ctor path).
        is_regressor = model.param.model.type == "regressor"
        head_dropout = float(getattr(cfg, "dropout", 0.0))

        def _head_loss(features_batch: torch.Tensor, targets_batch: torch.Tensor) -> torch.Tensor:
            sel = head_fit.select_out_proj_inputs(features_batch, cols_t, k)
            if head_dropout > 0.0 and model.out_proj.training:
                sel = F.dropout(sel, p=head_dropout, training=True)
            proj = model.out_proj(sel)
            if is_regressor:
                return model.loss_fn(proj.squeeze(-1), targets_batch).mean()
            log_probs = F.log_softmax(proj, dim=-1)
            return model.loss_fn(log_probs, targets_batch).mean()

        def _dev_loss() -> float:
            """Mean dev loss of the head, on the precomputed features."""
            model.out_proj.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vfeatures, vt in dev_feat_dl:
                    vfeatures = vfeatures.to(device=device)
                    vt        = vt.to(device=device)
                    val_loss += _head_loss(vfeatures, vt).item()
            return val_loss / len(dev_feat_dl)

        best_weights = keep_best.record(list(model.out_proj.parameters()), cfg,
                                        "out_proj", "out_proj",
                                        run_dir=lambda: self._ensure_run_dir(model))

        verbose = model.param.train.verbose
        tbar = tqdm(total=cfg.max_steps, desc="out_proj", file=sys.stdout,
                    ncols=140, disable=not verbose)

        step = 0
        best_val_loss = float("inf")
        evals_no_improve = 0
        stop = False
        train_loss_accum = 0.0
        train_steps_accum = 0

        while step < cfg.max_steps and not stop:
            model.out_proj.train()

            for batch in train_feat_dl:
                if step >= cfg.max_steps:
                    break

                features, targets = batch
                features = features.to(device=device)
                targets  = targets.to(device=device)

                loss = _head_loss(features, targets)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss_accum += loss.item()
                train_steps_accum += 1
                step += 1

                if step % cfg.eval_interval != 0:
                    continue

                # Validation pass — also on precomputed features.
                val_loss = _dev_loss()
                best_weights.update(step, val_loss)
                scheduler.step(val_loss)
                current_lr = opt.param_groups[0]["lr"]

                if val_loss < best_val_loss - cfg.early_stop_min_delta:
                    best_val_loss = val_loss
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1

                tbar.set_postfix({
                    "train_loss": f"{train_loss_accum / train_steps_accum:.4f}",
                    "val_loss":   f"{val_loss:.4f}",
                    "best":       f"{best_val_loss:.4f}",
                    "lr":         f"{current_lr:.2e}",
                    "no_imp":     evals_no_improve,
                })
                tbar.update(train_steps_accum)
                train_loss_accum = 0.0
                train_steps_accum = 0

                if evals_no_improve >= cfg.early_stop_patience:
                    tbar.total = tbar.n
                    tbar.refresh()
                    logger.info(f"out_proj early stop at step {step} — "
                                f"no improvement for {evals_no_improve} evals")
                    stop = True
                    break

                model.out_proj.train()

        tbar.close()

        # Evaluate a last step that fell between two evaluations; under
        # keep_best_weights the head then goes back to its best evaluation.
        if step > (best_weights.last_step or 0):
            best_weights.update(step, _dev_loss())
        best_weights.restore()

        # Restore gradients on all parameters so subsequent calls work normally.
        for p in model.parameters():
            p.requires_grad_(True)

        # Persist the trained out_proj into model_last.ckpt so a subsequent
        # load_model_checkpoint() call doesn't silently revert to the random
        # init that was saved at the end of train(). Without this every
        # downstream evaluation runs on a never-trained Linear head.
        self.save_model_checkpoint(model)

    def train_finetune(self, model: SONN, train_dl: DataLoader, dev_dl: DataLoader,
                       cfg: Any = None, val_dl: DataLoader | None = None) -> None:
        """End-to-end fine-tuning of all SONN parameters.

        After SONN's structural search is done every layer is a normal
        differentiable module — gradients flow through BasePolynomNeuron.weight,
        shared_proj / soft_binner / neuron_proj, and out_proj all at once.

        The step is launch-bound on these models (hundreds of tiny kernels),
        so on a CUDA device with adam / adamw it is captured once as a CUDA
        graph over a static buffer of the loader's batch shape and replayed
        per step, with the next batch copied into the buffer; see
        `training.finetune.FinetuneStep` and `OutProjTrainConfig.cuda_graph` /
        `data_on_device`. Evaluation, the plateau schedule and the early
        stop run outside the graph every `eval_interval` steps.

        `cfg` is an OutProjTrainConfig-shaped block of LR / scheduler /
        early-stop hyperparameters; it defaults to
        `model.param.train.finetune_train`, which exists so an end-to-end pass
        can use a different (usually much smaller) learning rate than the
        out_proj / layer_finetune passes that share `out_proj_train`.

        The model ends on its last step, or with `cfg.keep_best_weights` on
        the parameters of its lowest loss on the split the early stop reads
        (a last step between two evaluations is evaluated too), copied back
        in place, so a captured graph stays valid, from where
        `cfg.best_weights_copy` keeps them. The model is then saved to the
        run folder's `model_last.ckpt`.
        """
        cfg    = cfg if cfg is not None else model.param.train.finetune_train
        device = model.device

        # Early stop and lr schedule read the dev split, or the validation
        # split under train.stop_source 'val' (a split that selected nothing).
        stop_split = "val" if (evaluation.stop_source(self.config, val_dl) == "val") else "dev"
        stop_dl = val_dl if stop_split == "val" else dev_dl
        for p in model.parameters():
            p.requires_grad_(True)

        on_cuda = torch.device(device).type == "cuda"
        # Dropout on the head's input during the pass; a captured graph would
        # replay one mask, so the steps run eagerly while it is on.
        head_dropout = float(getattr(cfg, "dropout", 0.0))
        model.head_dropout = head_dropout
        use_graph = (bool(getattr(cfg, "cuda_graph", True)) and on_cuda
                     and cfg.optimizer in ("adam", "adamw") and head_dropout == 0.0)
        # A device-tensor lr is what lets the captured optimizer step pick up
        # the plateau schedule's changes (the value is copied in place).
        lr = torch.tensor(float(cfg.lr), device=device) if use_graph else cfg.lr
        if cfg.optimizer == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay, capturable=use_graph)
        elif cfg.optimizer == "adamw":
            # Decoupled weight decay. On a refinement pass over an already-fitted
            # network this is the better-behaved of the two: the penalty is
            # applied straight to the weights instead of entering the gradient
            # (and so the adaptive second-moment estimate), which keeps the decay
            # from being scaled differently per parameter.
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay, capturable=use_graph)
        elif cfg.optimizer == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
        else:
            raise ValueError(
                f"Unsupported optimizer for train_finetune: {cfg.optimizer!r} "
                "(supported: 'adam', 'adamw', 'sgd'). Note there is no full-batch "
                "LBFGS path here, unlike train_out_proj / layer_finetune."
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
            min_lr=cfg.lr_min, threshold=cfg.early_stop_min_delta,
        )

        def sync_lr() -> float:
            """ReduceLROnPlateau writes a float into the param group; copy it
            into the lr tensor the captured step reads and put the tensor back."""
            group = opt.param_groups[0]
            if isinstance(lr, torch.Tensor):
                if not isinstance(group["lr"], torch.Tensor):
                    lr.fill_(float(group["lr"]))
                    group["lr"] = lr
                return float(lr.item())
            return float(group["lr"])

        mode = str(getattr(cfg, "data_on_device", "auto")).lower()
        if mode not in ("auto", "true", "false"):
            raise ValueError(f"finetune_train.data_on_device must be 'auto', 'true' or 'false', got {mode!r}")
        batches, source_desc = finetune.finetune_batches(train_dl, device, mode if use_graph else "false", self.batch_callback)
        stepper = finetune.FinetuneStep(model, opt, model.loss_fn, finetune.finetune_prediction, use_graph)
        logger.info("end-to-end pass: %s; batches %s; batch size %d",
                    "CUDA graph (captured on the first batch)" if use_graph else "eager steps",
                    source_desc, int(train_dl.batch_size))

        def _stop_loss() -> float:
            """Mean loss of the model on the split the early stop reads."""
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vbatch in stop_dl:
                    vx, vt = self.batch_callback(vbatch) if self.batch_callback else vbatch
                    val_loss += model.loss_fn(
                        finetune.finetune_prediction(model, vx.to(device=device)),
                        vt.to(device=device),
                    ).mean().item()
            return val_loss / len(stop_dl)

        best_weights = keep_best.record(list(model.parameters()), cfg,
                                        "finetune", "finetune", split=stop_split,
                                        run_dir=lambda: self._ensure_run_dir(model))

        verbose = model.param.train.verbose
        tbar = tqdm(total=cfg.max_steps, desc="finetune", file=sys.stdout,
                    ncols=140, disable=not verbose)

        step = 0
        best_val_loss = float("inf")
        evals_no_improve = 0
        stop = False
        train_steps_accum = 0
        captured_logged = False

        model.train()
        while step < cfg.max_steps and not stop:
            x, y = next(batches)
            if x is None:          # epoch boundary
                continue
            stepper.step(x, y)
            if stepper.capture_error and not captured_logged:
                logger.warning("end-to-end pass: CUDA graph capture failed (%s); running eager steps",
                               stepper.capture_error)
                captured_logged = True
            elif use_graph and stepper.graph is not None and not captured_logged:
                logger.info("end-to-end pass: CUDA graph captured (%d parameters, batch %s)",
                            sum(p.numel() for p in model.parameters()), tuple(stepper.x_s.shape))
                captured_logged = True
            n_new = stepper.n_steps - train_steps_accum
            train_steps_accum = stepper.n_steps
            step += n_new

            if step % cfg.eval_interval != 0 and not (n_new > 1 and step // cfg.eval_interval > (step - n_new) // cfg.eval_interval):
                continue

            val_loss = _stop_loss()
            best_weights.update(step, val_loss)
            scheduler.step(val_loss)
            current_lr = sync_lr()

            if val_loss < best_val_loss - cfg.early_stop_min_delta:
                best_val_loss = val_loss
                evals_no_improve = 0
            else:
                evals_no_improve += 1

            train_loss = stepper.drain()
            train_steps_accum = 0
            tbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_loss":   f"{val_loss:.4f}",
                "best":       f"{best_val_loss:.4f}",
                "lr":         f"{current_lr:.2e}",
                "no_imp":     evals_no_improve,
            })
            tbar.n = min(step, cfg.max_steps)
            tbar.refresh()

            if evals_no_improve >= cfg.early_stop_patience:
                tbar.total = tbar.n
                tbar.refresh()
                logger.info(f"finetune early stop at step {step} — "
                            f"no improvement for {evals_no_improve} evals")
                stop = True
                break

            model.train()

        tbar.close()
        model.head_dropout = 0.0
        if use_graph:
            logger.info("end-to-end pass: %d steps replayed, %d eager (warm-up and tail batches)",
                        stepper.graph_steps, stepper.eager_steps)

        # Evaluate a last step that fell between two evaluations; under
        # keep_best_weights every parameter then goes back, in place, to its
        # best evaluation's value.
        if step > (best_weights.last_step or 0):
            best_weights.update(step, _stop_loss())
        best_weights.restore()

        # The run folder holds what the run produced: without this save a
        # later load_model_checkpoint() returns the model from before the pass.
        self.save_model_checkpoint(model)

    def train_joint(self, model: SONN, train_dl: DataLoader, dev_dl: DataLoader,
                    verbose: bool = True, val_dl: DataLoader | None = None) -> SONN:
        """Grow the network with neurons selected jointly at every layer.

        The joint selection mode (`model.selection_mode: joint`, called by
        `train`). A greedy layer keeps the candidates that score best on
        their own; here each layer keeps the ones a sparsity penalty leaves
        when a pool of candidates is trained together with a head
        (`_train_joint_layer`):

        1. The layer's candidates are created, calibrated and fitted on
           their own, as `train_layer` does for a greedy layer. Those fits
           are the pool's starting weights; `train.joint_sparsity.prefilter`
           decides how many enter the pool (all finite ones, or
           prefilter_mult x `nbest_neurons` by a selection method).
        2. A temporary head over the whole pool is fitted with the neurons
           frozen (`train_out_proj`).
        3. The layer, the head and, under 'gates', the gates are trained
           together on the training split with loss = data loss + lam *
           penalty (`train.joint_sparsity`); the earlier layers are frozen
           and the layer's input is computed once. Every `eval_interval`
           steps the log shows lam, both terms, the loss on dev (or the
           validation split under `train.stop_source: val`) and the number
           of active neurons. Under `lam_control` 'anneal' / 'adaptive' lam
           is steered toward `nbest_neurons`: the pass stops once the count
           holds just above it, and rolls back to the closest state from
           above if the count falls below it (`training.joint_selection.CountControl`).
        4. The layer is cut to `nbest_neurons` of the active neurons (if
           more remain, the ones contributing most to the prediction: the
           norm of their head column times the standard deviation of their
           gated, scaled output on dev). Gates and head are discarded: the
           next layer reads the survivors' own outputs. With
           `train.layer_finetune` the survivors are then trained together
           with a temporary head (`training.finetune.train_layer_finetune`),
           as a greedy layer's are. The survivors' `err_values` become their own dev
           errors after the pass, before the fine-tune, as in a greedy layer.

        Each layer then gets an error for the growth rule, as greedy layers
        do (`GrowthCriterion`, `train.criterion_minimum_width`,
        `stop_train_min_delta`, `max_layer_count`), by
        `train.layer_err_source`: 'neuron' scores its survivors' own dev
        errors after the pass (reduced by `layer_err_criterion`), 'readout'
        the dev loss of a head refitted over them. Both are logged. When
        growth stops, the layers after the best one are dropped, and the
        model's head (`num_out_neurons` wide) is fitted over the last layer
        (`train_out_proj`). The model is saved as `model_last.ckpt`;
        `train_finetune` polishes it like any other.

        Parameters
        ----------
        model : SONN
            The model to train. Its existing layers are discarded.
        train_dl, dev_dl : DataLoader
            Training split (all fits) and dev split (candidate selection,
            the heads' early stop, the pass's log and the layer error).
        verbose : bool
            Log the time each layer takes, and show progress bars.
        val_dl : DataLoader, optional
            Validation split: its layer error is logged, and under
            `train.stop_source: val` it drives the growth rule and the pass's
            log instead of dev.

        Returns
        -------
        model : SONN
            The trained model (the same object).
        """
        if model.out_proj is None:
            raise ValueError("model.selection_mode='joint' needs model.use_output_projection=true")
        head_width = int(model.out_proj.in_features)
        n_out = int(model.out_proj.out_features)
        seed = int(model.param.train.seed)
        stop_source = evaluation.stop_source(self.config, val_dl)
        stop_split = "val" if stop_source == "val" else "dev"
        stop_dl = val_dl if stop_split == "val" else dev_dl
        source = str(model.param.train.layer_err_source)
        crit = model.param.train.layer_err_criterion
        growth = GrowthCriterion(
            width=int(model.param.train.criterion_minimum_width),
            epsilon=float(model.param.train.stop_train_epsilon_condition),
            min_delta=float(getattr(model.param.train, "stop_train_min_delta", 0.0)),
        )

        model.layers = nn.ModuleList()
        model.layer_val_err = []
        model.out_proj = None
        while True:
            layer_index = len(model.layers)
            layer = model.create_layer(layer_index)
            model.layers.append(layer)
            normalization.fit_layer_inputs(model, layer, train_dl, batch_callback=self.batch_callback)
            with timed_block(f"train joint layer #{layer_index}", verbose=verbose):
                self._train_joint_layer(model, layer, train_dl, dev_dl, stop_dl, stop_split, n_out, verbose)

            neuron_err = (layer.err_values.min() if crit == "top" else layer.err_values.mean()).item()
            readout = self._joint_readout(model, train_dl, [dev_dl] + ([val_dl] if val_dl is not None else []),
                                          dev_dl, n_out)
            model.out_proj = None
            layer.err = readout[0] if source == "readout" else neuron_err
            logger.info(f"Joint layer #{layer_index}: dev loss of a head over the survivors {fmt_err(readout[0])}; "
                        f"best survivor's own dev error {fmt_err(neuron_err)}; layer error ({source}) "
                        f"{fmt_err(layer.err)}")
            stop_err = layer.err
            if val_dl is not None:
                val_err = readout[1] if source == "readout" else evaluation.layer_val_err(model, layer, val_dl, batch_callback=self.batch_callback)
                layer.val_err = val_err
                model.layer_val_err.append(val_err)
                logger.info(f"Joint layer #{layer_index}: dev {fmt_err(layer.err)} | val {fmt_err(val_err)}")
                if stop_source == "val":
                    stop_err = val_err
            stopped = growth.update(layer_index, stop_err)
            logger.info(growth.describe(layer_index, stop_err))
            logger.info(f"Layer errors: {abbrev_floats([l.err for l in model.layers], fmt=fmt_err)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if stopped or not (layer_index < model.param.train.max_layer_count - 1):
                break

        keep = growth.kept_index(getattr(model.param.train, "layer_keep", "best"), len(model.layers))
        model.layer_err.clear()
        model.layer_err.extend(l.err for l in model.layers)
        for layer_idx in range(keep + 1, len(model.layers)):
            self.cleanup_layer_checkpoints(layer_idx)
        del model.layers[keep + 1:]
        for layer in model.layers:
            layer.set_neuron_module()

        model.out_proj = _linear_from_seed(head_width, n_out, seed).to(device=model.device)
        self.train_out_proj(model, train_dl, dev_dl)
        logger.info(f"joint selection: {len(model.layers)} layer(s) kept; {stop_split} loss of the "
                    f"{head_width}-wide head {evaluation.mean_loss(model, stop_dl, model.infer, batch_callback=self.batch_callback):.4f}")
        self.save_model_checkpoint(model)
        return model

    def _joint_last_layer(self, model: SONN, train_dl: DataLoader, dev_dl: DataLoader,
                          val_dl: DataLoader | None, verbose: bool) -> None:
        """`model.selection_mode: joint_last`: build the last layer of a grown
        greedy network again and select it as a joint layer.

        The layer is dropped and created anew at the same index, its
        candidates calibrated and fitted on the same input, then screened,
        trained under the sparsity penalty and cut to `nbest_neurons` by
        `_train_joint_layer`. The earlier layers stay as greedy selected
        them. Leaves a fresh, unfitted head of the model's width in
        `model.out_proj`."""
        stop_split = "val" if evaluation.stop_source(self.config, val_dl) == "val" else "dev"
        stop_dl = val_dl if stop_split == "val" else dev_dl
        head_width = int(model.out_proj.in_features)
        n_out = int(model.out_proj.out_features)
        idx = len(model.layers) - 1
        greedy_err = model.layers[idx].err
        del model.layers[idx]
        layer = model.create_layer(idx)
        model.layers.append(layer)
        normalization.fit_layer_inputs(model, layer, train_dl, batch_callback=self.batch_callback)
        with timed_block(f"train joint layer #{idx}", verbose=verbose):
            self._train_joint_layer(model, layer, train_dl, dev_dl, stop_dl, stop_split, n_out, verbose)
        errs = layer.err_values
        layer.err = (errs.min() if model.param.train.layer_err_criterion == "top" else errs.mean()).item()
        model.layer_err[idx] = layer.err
        layer.set_neuron_module()
        model.out_proj = _linear_from_seed(head_width, n_out, int(model.param.train.seed)).to(device=model.device)
        logger.info(f"joint_last: layer #{idx} of {len(model.layers)} selected jointly; best survivor's own dev "
                    f"error {fmt_err(layer.err)} (greedy {fmt_err(greedy_err)})")

    def _train_joint_layer(self, model: SONN, layer: SONNLayer, train_dl: DataLoader, dev_dl: DataLoader,
                           stop_dl: DataLoader, stop_split: str, n_out: int, verbose: bool) -> None:
        """Steps 1-4 of `train_joint` for the model's last layer: candidate
        fits and screen, the pool head, the joint pass, the cut to
        `nbest_neurons`, the survivors' own errors after the pass and the
        per-layer fine-tune (`train.layer_finetune`)."""
        js = model.param.train.joint_sparsity
        target = int(model.nbest_neurons)
        device = model.device
        n_candidates = len(layer)
        if str(js.prefilter) == "none":
            pool_size, method = n_candidates, "plain"
            screen = "every finite fit"
        else:
            pool_size = min(n_candidates, max(target, math.ceil(float(js.prefilter_mult) * target)))
            method = str(js.prefilter)
            screen = f"up to {pool_size} by {method}"
        self.train_layer(model, layer, train_dl, dev_dl, None, nbest=pool_size, selection_method=method,
                         joint_pool=True)
        logger.info(f"joint layer #{layer.layer_index}: pool of {len(layer)} of {n_candidates} candidates "
                    f"({screen}); keeping {target}")
        own_before = layer.err_values.clone()

        # A head over the whole pool, fitted over the frozen candidates.
        model.out_proj = _linear_from_seed(len(layer), n_out, int(model.param.train.seed)).to(device=device)
        self.train_out_proj(model, train_dl, dev_dl)
        # The layer's input does not change during the pass: compute it once.
        train_in = evaluation.precompute_dl(model, train_dl, device, batch_callback=self.batch_callback)
        stop_in = evaluation.precompute_dl(model, stop_dl, device, batch_callback=self.batch_callback)
        pool = joint_selection.joint_pool(model, train_in, js)
        model.out_proj = None

        fitted = None
        if pool.size <= target and str(js.lam_control) != "fixed":
            logger.info(f"joint layer #{layer.layer_index}: {pool.size} neurons, no more than it keeps; "
                        "nothing to select, so no pass")
        else:
            # Under restore_weights the pass only picks the survivors: their
            # coefficients go back to the candidates' own fits after the cut.
            fitted = ([p.detach().clone() for p in layer.parameters()]
                      if bool(js.restore_weights) else None)
            joint_selection.sparsify_joint_pool(model, pool, train_in, stop_in, stop_split, target, verbose)
        pool.eval()
        keep = joint_selection.joint_survivors(pool, target, stop_in)
        if fitted is not None:
            with torch.no_grad():
                for p, p0 in zip(layer.parameters(), fitted):
                    p.copy_(p0)
            logger.info(f"joint layer #{layer.layer_index}: coefficients restored to the candidates' own fits "
                        "(restore_weights)")
        pruning.keep_columns(layer, keep)
        layer.d_model = len(layer)
        before = own_before[keep.to(own_before.device)]
        layer.err_values = joint_selection.own_errors(model, layer, dev_dl, batch_callback=self.batch_callback)
        # As in a greedy layer, the errors are taken before the fine-tune,
        # which turns the survivors into a basis for a head: their own
        # errors after it no longer score them as predictors.
        if model.param.train.layer_finetune:
            finetune.train_layer_finetune(model, layer, train_dl, dev_dl, batch_callback=self.batch_callback,
                                          run_dir=lambda: self._ensure_run_dir(model))
        logger.info(f"joint layer #{layer.layer_index}: kept {len(layer)} of {pool.size}; survivors' own dev "
                    f"error before the pass min {fmt_err(before.min().item())} / median "
                    f"{fmt_err(before.median().item())}, after min {fmt_err(layer.err_values.min().item())} / "
                    f"median {fmt_err(layer.err_values.median().item())}")

    def _joint_readout(self, model: SONN, train_dl: DataLoader, dls: list[DataLoader], dev_dl: DataLoader,
                       n_out: int) -> list[float]:
        """Loss on each loader of a head fitted over every neuron of the last
        layer (`train_out_proj`, early stop on `dev_dl`). Leaves the head in
        `model.out_proj`."""
        model.out_proj = _linear_from_seed(len(model.layers[-1]), n_out,
                                           int(model.param.train.seed)).to(device=model.device)
        self.train_out_proj(model, train_dl, dev_dl)
        model.eval()
        return [evaluation.mean_loss(model, dl, model.infer, batch_callback=self.batch_callback) for dl in dls]

    def infer(
        self,
        model: SONN,
        test_dl: DataLoader,
        verbose: bool = True,
        use_compile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict over test_dl and return (predictions, targets), both in loader order.

        The predictions are one value per row for regression, a logit per
        row for binary models and a row of log-probabilities per sample for
        multi-class models (see `SONN.infer`). Each batch must be an
        `(x, y)` pair: `batch_callback` is not applied here.

        use_compile=True wraps model.infer with torch.compile before the loop,
        fusing the sequential per-layer kernel launches into a single optimized
        graph.  First batch is slower (compilation), subsequent batches are
        faster — worthwhile when test_dl has many batches or is called repeatedly.
        """
        infer_fn = torch.compile(model.infer) if use_compile else model.infer
        result = []
        targets = []
        model.eval()
        for step, batch in tqdm(enumerate(test_dl), disable=not verbose):
            x_inp, target = batch
            with torch.inference_mode():
                x = infer_fn(x_inp)
                result.append(x)
                targets.append(target)
        return torch.cat(result, dim=0), torch.cat(targets, dim=0)

    def prune(self, model: SONN) -> None:
        """Strip the network down to the neurons that actually reach the output.

        The last layer is cut to the columns the model reads, and every earlier
        layer is then cut to whatever the layers reading it still reference —
        its successor, and with `shortcut.prev_layers` later layers too. A
        layer nobody reads any more is deleted.

        How many columns the model reads is `SONN._readout_width`. Without a
        head, `SONN.infer` returns the best-error column and a single neuron
        suffices, except for a multi-class model with per-neuron projections
        (`use_neuron_proj`): its prediction sums every last-layer neuron's
        projected output, so the whole last layer stays. With `out_proj` it
        reads `in_features` columns, chosen by `_best_neuron_columns` — so
        collapsing the layer to one neuron would leave the head consuming
        zero-padding on every other input and silently destroy the model
        rather than raising. Keep exactly the columns the head consumes
        instead.

        The head needs no reindexing: the retained neurons keep their
        `err_values`, so re-running `_best_neuron_columns` on the pruned layer
        re-derives the same neurons in the same ascending-error order, which is
        the order the head's inputs were fitted in. Every kept layer's
        `d_model` is set to its new width.

        Raises
        ------
        ValueError
            If the model was built with `model.use_layer_norm`. Its LayerNorms
            normalize each row over every column of a layer's input, so
            removing columns would change the predictions; the model is left
            as it is.
        """
        if bool(model.param.model.use_layer_norm):
            raise ValueError(
                "Trainer.prune cannot keep the predictions of a model built with "
                "model.use_layer_norm: each LayerNorm normalizes over every column of a "
                "layer's input, so removing columns changes it. The model is unchanged."
            )
        # Every layer's input is a concatenation of blocks — the outputs of
        # the layers in its `input_layers`, then the raw inputs (see the
        # layout note above SONN.forward) — and a layer can be read by several
        # later ones under `shortcut.prev_layers`. A flat src index is only
        # meaningful against the block widths it was built for, so each layer's
        # src_idxs are decoded into (source, index inside the source's block)
        # right after the layer itself is pruned — its sources, all earlier,
        # still have their pre-prune widths then — and re-encoded against the
        # final widths at the end. Source -1 is the raw-input block.
        n_layers = len(model.layers)
        sources = [model.layer_sources(j) for j in range(n_layers)]
        decoded: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

        def decode(j: int) -> None:
            if j == 0:
                return
            blocks = model.input_blocks(j)
            codes = torch.tensor([-1 if s is None else s for s, _ in blocks], dtype=torch.long)
            widths = torch.tensor([w for _, w in blocks], dtype=torch.long)
            ends = widths.cumsum(0)
            starts = ends - widths
            per_module = []
            for module in model.layers[j].neuron_models:
                src = module.src_idxs.detach().cpu().long()
                block = torch.searchsorted(ends, src, right=True)
                per_module.append((codes[block], src - starts[block]))
            decoded[j] = per_module

        last_layer = model.layers[-1]
        n_keep = model._readout_width(last_layer)

        # Sorted, not left in topk's error order: module_idxs is laid out
        # module-by-module and the rest of this method (and a second prune()
        # call) relies on that ordering being preserved.
        used = last_layer.err_values.topk(n_keep, largest=False).indices.sort().values
        pruning.keep_columns(last_layer, used)
        decode(n_layers - 1)

        # Walk layers from second-to-last down to 0. By the time layer i is
        # reached every layer that reads it is final, so the columns of i
        # still referenced are known. Layers read by nobody are only marked
        # here and dropped after the walk, so positions stay put meanwhile.
        deleted: set[int] = set()
        for i in range(n_layers - 2, -1, -1):
            cur_layer = model.layers[i]
            readers = [j for j in range(i + 1, n_layers) if j not in deleted and i in sources[j][0]]
            refs = [local[source == i] for j in readers for source, local in decoded[j]]
            used = torch.unique(torch.cat(refs)) if refs else torch.empty(0, dtype=torch.long)

            if used.numel() == 0:
                # No surviving layer reads cur_layer's outputs (the best path
                # uses other layers or the original features through the
                # shortcut only). The layer is dead weight — delete it.
                deleted.add(i)
                continue

            # `used` is sorted (torch.unique returns sorted), so kept column
            # used[r] becomes column r.
            pruning.keep_columns(cur_layer, used)

            for j in readers:
                for source, local in decoded[j]:
                    mask = source == i
                    local[mask] = torch.searchsorted(used, local[mask])
            decode(i)

        # Re-encode every surviving layer's src_idxs against the final block
        # widths, without the deleted layers' blocks, and store the layout
        # under the new positions.
        keep = [j for j in range(n_layers) if j not in deleted]
        new_pos = {old: new for new, old in enumerate(keep)}
        for j in keep:
            layer = model.layers[j]
            if j > 0:
                input_layers = [s for s in sources[j][0] if s not in deleted]
                input_raw = sources[j][1]
                starts: dict[int, int] = {}
                offset = 0
                for s in input_layers:
                    starts[s] = offset
                    offset += len(model.layers[s])
                if input_raw:
                    starts[-1] = offset
                for (source, local), module in zip(decoded[j], layer.neuron_models):
                    src = torch.empty_like(local)
                    for code, start in starts.items():
                        mask = source == code
                        src[mask] = local[mask] + start
                    assert bool(torch.isin(source, torch.tensor(list(starts))).all()), \
                        "prune: a kept neuron reads a deleted layer"
                    module.src_idxs = src.to(device=module.src_idxs.device, dtype=module.src_idxs.dtype)
                layer.input_layers = [new_pos[s] for s in input_layers]
                layer.input_raw = input_raw
            else:
                layer.input_layers, layer.input_raw = [], True

        if deleted:
            # Drop the deleted layers and renumber the rest. Each
            # BasePolynomNeuron caches its own .layer_index (set at
            # construction, also persisted via params_metadata) and
            # plot_model + checkpoint metadata read it directly, so the
            # renumber has to propagate down into the neurons too.
            model.layers = nn.ModuleList([model.layers[j] for j in keep])
            for j, L in enumerate(model.layers):
                L.layer_index = j
                for nm in L.neuron_models:
                    nm.layer_index = j
            logger.info(
                f"prune: removed layer(s) at position(s) {sorted(deleted)} "
                "(outputs read by no surviving layer)."
            )

        # The widths the layers have now, as neuron_selection records them
        # (and as the checkpoint metadata saves them).
        for layer in model.layers:
            layer.d_model = len(layer)

