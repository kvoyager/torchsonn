import gc
import os
import sys
import torch
from torchsonn.model import SONN
from tqdm import tqdm
import logging
from torch import nn
from dataclasses import dataclass, field, fields, replace
from copy import deepcopy
from functools import partial
from torch.func import vmap, grad
from torch.utils.data import DataLoader
from pathlib import Path
from torchsonn.optimizers import optimizer_map, BaseOptimizer
from torchsonn.schedulers import scheduler_map, BaseScheduler
from torchsonn.layer import SONNLayer, NeuronModuleList
from torchsonn.neurons import BasePolynomNeuron
from torchsonn.data.dataset import SONNDataset
import numpy as np
import random

from torchsonn.utils import timed_block, abbrev_floats
from torchsonn.loss import bias_error, bias_error_l2, bias_error_js

logger = logging.getLogger(__name__)
import re
from typing import Optional, Any, Callable
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class LayerAccumulator:
    """Mutable per-layer state accumulated across neuron-model training calls.

    Only lives in memory — never serialized directly.  train_layer creates one
    at the start and passes it through every train_model_ensemble call so the
    growing err / module_idxs lists are available when building StepCheckpoints.
    """
    err: list[torch.Tensor] = field(default_factory=list)
    module_idxs: list[torch.Tensor] = field(default_factory=list)
    # Per-neuron-model (C_i, N_val) raw scalar outputs on the dev split.
    # neuron_selection cats along dim 0 → (C_total, N_val) and uses it for
    # OMP-style decorrelation alongside err / module_idxs.
    z_val: list[torch.Tensor] = field(default_factory=list)
    layer_completed: bool = False


@dataclass
class StepCheckpoint:
    """Immutable snapshot of all state needed to save or resume a training step.

    All fields except `scheduler` are required so the type checker can catch a
    missing assignment at construction time rather than at serialization time.
    Built explicitly at each save site inside train_model_ensemble; never
    mutated after construction (use dataclasses.replace() for variants).
    """
    # serialization anchor — also used for checkpoint_dir / keep_last_n lookups
    model: SONN
    opt: BaseOptimizer
    # positional context
    layer_idx: int
    neuron_model_idx: int
    # step counters
    epoch: int
    step: int
    global_step: int
    # per-ensemble training state
    best_val_losses: torch.Tensor
    smoothed_val_losses: torch.Tensor
    last_steps_improve: torch.Tensor
    early_stop_flags: torch.Tensor
    lr: torch.Tensor
    neuron_model_completed: bool
    # layer-level state (needed to resume across neuron-model iterations)
    layer_completed: bool
    err: list[torch.Tensor]
    module_idxs: list[torch.Tensor]
    # optional
    scheduler: Optional[BaseScheduler] = None

    def to_dict(self) -> dict:
        def safe_value(v):
            if hasattr(v, "state_dict"):
                return v.state_dict()
            return v
        return {
            f.name: safe_value(getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) is not None
        }


class Trainer:
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

    @staticmethod
    def _split_loader(dl: DataLoader, split: int) -> DataLoader:
        """Build a DataLoader over the same underlying tensors but restricted to one split.

        The original `copy(dl)` was shallow, so dl_a.dataset and dl_b.dataset aliased
        the same SONNDataset object — assigning .split mutated both. Here we make a
        fresh SONNDataset (sharing x/target storage) per split so they're independent.
        """
        ds = dl.dataset
        new_ds = SONNDataset(ds.x, ds.target, split=split)
        shuffle = isinstance(dl.sampler, torch.utils.data.RandomSampler)
        return DataLoader(
            new_ds,
            batch_size=dl.batch_size,
            shuffle=shuffle,
            num_workers=dl.num_workers,
            drop_last=dl.drop_last,
        )

    @classmethod
    def set_seed(cls, seed: int = 42) -> None:
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
    @classmethod
    def get_checkpoint_dir(cls, model: SONN) -> Path:
        checkpoint_dir = model.param.train.checkpoint_dir
        if checkpoint_dir == "":
            # After the src-layout move this file lives at
            # `<repo>/src/torchsonn/trainer.py`, so three `.parent`s reach the
            # repo root.
            checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"
        else:
            checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        return checkpoint_dir

    @classmethod
    def parse_checkpoint_step(cls, fname: str) -> tuple[int, int, int]:
        m = re.search(r"model_layer_(\d+)_neuron_(\d+)_step_(\d+)(?:_last)?\.ckpt$", fname)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Same arity as the match path so max()/sorted() compare apples to apples;
        # the (-1, -1, -1) sentinel sorts strictly below any real (layer, neuron,
        # step) tuple, so stray non-matching files never get picked as "latest".
        return -1, -1, -1

    def from_checkpoint(self, model: SONN) -> dict[str, Any] | None:
        checkpoint_dir = self.get_checkpoint_dir(model)
        checkpoints = os.listdir(checkpoint_dir)

        checkpoints = [checkpoint for checkpoint in checkpoints if not checkpoint.endswith("model_last.ckpt")]

        if not checkpoints:
            return

        latest_checkpoint = max(checkpoints, key=self.parse_checkpoint_step)
        checkpoint_data = torch.load(checkpoint_dir / latest_checkpoint, weights_only=False)
        return checkpoint_data

    def save_checkpoint(self, ckpt: StepCheckpoint, suffix: str = "") -> None:
        checkpoint_dir = self.get_checkpoint_dir(ckpt.model)
        if suffix:
            suffix = "_" + suffix
        checkpoint_filename = f"model_layer_{ckpt.layer_idx}_neuron_{ckpt.neuron_model_idx}_step_{ckpt.global_step}{suffix}.ckpt"
        d = ckpt.to_dict()
        d["config"] = ckpt.model.param
        torch.save(d, checkpoint_dir / checkpoint_filename)
        self.cleanup_checkpoints(checkpoint_dir, keep_last_n=ckpt.model.param.train.keep_last_n)

    def load_model_checkpoint(self, model: SONN, device: torch.device | str = "cpu") -> None:
        checkpoint_filename = f"model_last.ckpt"
        checkpoint_data = torch.load(self.get_checkpoint_dir(model) / checkpoint_filename, weights_only=False)
        # restore model architecture
        model.restore_from_checkpoint_metadata(checkpoint_data["model"])
        # load weights
        model.load_state_dict(checkpoint_data["model"], strict=False)
        model.to(device=device)

    def save_model_checkpoint(self, model: SONN) -> None:
        checkpoint_filename = f"model_last.ckpt"
        torch.save({
            'model': model.state_dict(),
        }, self.get_checkpoint_dir(model) / checkpoint_filename)

    def cleanup_checkpoints(self, checkpoint_dir: Path, keep_last_n: int = 10) -> None:
        """
        Deletes older checkpoints, keeping only the last `n` for each (layer_idx, neuron_model_idx).

        Args:
            checkpoint_dir (str): Path to directory containing .ckpt files.
            keep_last_n (int): Number of most recent checkpoints to keep per (layer, neuron).
        """

        checkpoints = os.listdir(checkpoint_dir)
        checkpoints = [checkpoint for checkpoint in checkpoints if not checkpoint.endswith("_last.ckpt")]
        checkpoints = sorted(checkpoints, key=self.parse_checkpoint_step)

        checkpoints_to_remove = checkpoints[:max(0, len(checkpoints) - keep_last_n)]

        for checkpoint in checkpoints_to_remove:
            os.remove(checkpoint_dir / checkpoint)

    def cleanup_layer_checkpoints(self, model: SONN, layer_idx: int) -> None:
        """
        Deletes all checkpoints for a layer

        Args:
            checkpoint_dir (str): Path to directory containing .ckpt files.
            keep_last_n (int): Number of most recent checkpoints to keep per (layer, neuron).
        """

        def filter_checkpoints_by_layer(filenames, idx):
            pattern = re.compile(rf"model_layer_{idx}_neuron_\d+_step_\d+(?:_last)?\.ckpt$")
            return [f for f in filenames if pattern.search(f)]

        checkpoint_dir = self.get_checkpoint_dir(model)
        checkpoints = os.listdir(checkpoint_dir)
        checkpoints_to_remove = filter_checkpoints_by_layer(checkpoints, layer_idx)

        for checkpoint in checkpoints_to_remove:
            os.remove(checkpoint_dir / checkpoint)
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
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    @staticmethod
    def _is_dist() -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    @staticmethod
    def _ensemble_slice(ensemble_size: int) -> tuple[int, int] | None:
        """Return (start, end) indices for this rank's ensemble slice, or None.

        Returns None when not running under torch.distributed or when
        ensemble_size is not divisible by world_size.  In the latter case a
        warning is emitted on rank 0 and the caller falls back to
        single-device mode for that neuron model.  Set nbest_neurons (and
        max_neuron_models) to a multiple of world_size to avoid the fallback.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return None
        world_size = dist.get_world_size()
        if world_size == 1:
            return None
        if ensemble_size % world_size != 0:
            if dist.get_rank() == 0:
                logger.warning(
                    f"ensemble_size={ensemble_size} is not divisible by "
                    f"world_size={world_size}; falling back to single-device "
                    "mode for this neuron model. Pad ensemble_size to a "
                    "multiple of world_size to enable ensemble-parallel training."
                )
            return None
        local_size = ensemble_size // world_size
        start = dist.get_rank() * local_size
        return start, start + local_size

    @staticmethod
    def _gather_ensemble_params(params: dict, shared_param_names: list) -> dict:
        """All-gather non-shared params from all ranks, concatenating along dim 0.

        Restores the full (ensemble_size, ...) shape after each rank trained
        its local slice.  Shared params (e.g. shared_proj_weight) are the
        same on all ranks after all-reduce and are passed through unchanged.
        Requires equal local_size on every rank, which _ensemble_slice
        guarantees by only returning a slice when ensemble_size % world_size == 0.
        """
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return params
        world_size = dist.get_world_size()
        shared = set(shared_param_names)
        result = {}
        for k, v in params.items():
            if k in shared:
                result[k] = v
                continue
            v_c = v.contiguous()
            gathered = [torch.empty_like(v_c) for _ in range(world_size)]
            dist.all_gather(gathered, v_c)
            result[k] = torch.cat(gathered, dim=0)
        return result

    @staticmethod
    def _allreduce_shared_grads(grads: dict, shared_param_names: list) -> None:
        """Average shared-param gradients across all ranks in-place.

        Called immediately after loss_fn_vmapped so every rank applies the
        same shared-param update regardless of which ensemble members it owns.
        Non-shared param grads are left untouched — each rank's optimizer
        already operates on its local slice independently.
        """
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return
        shared = set(shared_param_names)
        for k in shared:
            if k in grads:
                dist.all_reduce(grads[k], op=dist.ReduceOp.AVG)
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
        resume: bool = False,
    ) -> SONN:
        if self.class_weights is not None:
            model.set_class_weights(self.class_weights)

        min_error = sys.float_info.max
        error_stopped_decrease = False
        model.layers = nn.ModuleList()
        error_min_index = 0
        checkpoint_data = None
        # Per-layer snapshots of projection weights (multi-class only).
        # Each train_layer call overwrites model.shared_proj / model.neuron_proj
        # with the best neuron's state; we snapshot after each layer so we can
        # restore whichever layer achieved the minimum error before final save.
        shared_proj_states: dict[int, dict] = {}

        if resume:
            # load checkpoint
            checkpoint_data = self.from_checkpoint(model)
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

            # proceed until stop condition is fulfilled

            if layer.err < min_error:
                # layer error has been decreased, memorize the layer index
                error_min_index = layer.layer_index

            if (layer.err > min_error and
                layer.layer_index > 0 and
                layer.layer_index - error_min_index >= self.config.train.criterion_minimum_width):

                # layer error stopped decreasing
                error_stopped_decrease = True

            if layer.layer_index > 0 and layer.err < min_error and min_error > 0:
                if (min_error - layer.err) / min_error < self.config.train.stop_train_epsilon_condition:
                    # layer relative error decrease value is below stop condition
                    error_stopped_decrease = True

            min_error = min(min_error, layer.err)

            # if error does not decrease anymore or number of layers reached the limit
            # or the layer does not have any valid neuron - stop training
            if error_stopped_decrease or not (layer.layer_index < self.config.train.max_layer_count - 1):
                break

        model.layer_err.clear()
        for i in range(0, len(model.layers)):
            model.layer_err.append(model.layers[i].err)

        # delete unused layers keeping only error_min_index layers
        for layer_idx in range(error_min_index + 1, len(model.layers)):
            self.cleanup_layer_checkpoints(model, layer_idx)
        del model.layers[error_min_index + 1:]

        for layer in model.layers:
            layer.set_neuron_module()

        # Restore the projection that was co-trained with the optimal layer depth.
        # Each train_layer call overwrites model.shared_proj in the shared-proj path;
        # without this restore the checkpoint contains the projection from the
        # last trained layer (a different input space than layers[error_min_index]).
        if model.shared_proj is not None and error_min_index in shared_proj_states:
            model.shared_proj.load_state_dict(shared_proj_states[error_min_index])

        self.save_model_checkpoint(model)

        return model

    def train_layer(
            self,
            model: SONN,
            layer: SONNLayer,
            train_dl: DataLoader,
            dev_dl: DataLoader,
            checkpoint_data: dict[str, Any] | None,
    ) -> None:
        """Calculate neuron weights
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
            train_feat_dl = self._precompute_dl(model, train_dl, device)
            dev_feat_dl   = self._precompute_dl(model, dev_dl,   device)
        else:
            train_feat_dl = train_dl
            dev_feat_dl   = dev_dl

        if checkpoint_data is not None:
            accumulator.err = checkpoint_data["err"]
            accumulator.module_idxs = checkpoint_data["module_idxs"]
            loaded_neuron_model_idx = checkpoint_data["neuron_model_idx"]
            completed = checkpoint_data["neuron_model_completed"]
            if completed:
                if loaded_neuron_model_idx == len(layer.neuron_models) - 1:
                    # all neuron models have been trained
                    start_neuron_model_idx = 0
                else:
                    # only part of neuron models have been trained and each neuron model has been trained completely
                    start_neuron_model_idx = loaded_neuron_model_idx + 1
            else:
                # neuron model training has not been completed
                start_neuron_model_idx = loaded_neuron_model_idx
        else:
            start_neuron_model_idx = 0

        for neuron_model_idx, neuron_model in enumerate(layer.neuron_models[start_neuron_model_idx:]):
            neuron_model_idx += start_neuron_model_idx

            regularity_err = None
            bias_err = None

            if model.need_regularity_err:
                loss_fn_vmapped, eval_loss_fn_vmapped, pred_fn_vmapped, params_batch, buffers_batch, shared_param_names = (
                    self.create_loss_functions(model, neuron_model))
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
                trained_params_per_model[neuron_model_idx] = trained_params
                regularity_err = self.regularity_err(model, pred_fn_vmapped, trained_params, buffers_batch, dev_feat_dl, device, skip_model_fwd=precompute)
                # Cache the per-candidate raw scalar outputs on the full dev
                # split for neuron_selection's OMP decorrelation.
                accumulator.z_val.append(
                    self._compute_z_val(neuron_model, model, dev_feat_dl, device, precompute)
                )

            if model.need_bias_err:
                neuron_module_a = deepcopy(neuron_model)
                neuron_module_b = deepcopy(neuron_model)

                train_dl_a = self._split_loader(train_feat_dl, 0)
                train_dl_b = self._split_loader(train_feat_dl, 1)
                dev_dl_a   = self._split_loader(dev_feat_dl,   0)
                dev_dl_b   = self._split_loader(dev_feat_dl,   1)

                loss_fn_vmapped_a, eval_loss_fn_vmapped_a, pred_fn_vmapped_a, params_batch_a, buffers_batch_a, shared_param_names_a = (
                    self.create_loss_functions(model, neuron_model))
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
                trained_params_per_model.setdefault(neuron_model_idx, trained_params_a)

                loss_fn_vmapped_b, eval_loss_fn_vmapped_b, pred_fn_vmapped_b, params_batch_b, buffers_batch_b, shared_param_names_b = (
                    self.create_loss_functions(model, neuron_model))
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
                bias_err = self.bias_err(
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
                    skip_model_fwd=precompute)
                # If regularity branch ran above it already appended z_val for
                # this neuron_model_idx; skip here to keep the lists aligned
                # with err / module_idxs (one entry per neuron_model). Otherwise
                # use the split-A trained module as a representative.
                if not model.need_regularity_err:
                    accumulator.z_val.append(
                        self._compute_z_val(neuron_module_a, model, dev_feat_dl, device, precompute)
                    )

            module_err = model.get_error(model.criterion_type, regularity_err, bias_err)
            accumulator.module_idxs.append(
                torch.stack([
                   torch.empty(module_err.shape[0], dtype=torch.long, device=device).fill_(neuron_model_idx),
                   torch.arange(module_err.shape[0], device=device)],
                dim=1)
            )
            accumulator.err.append(module_err)

        err_values, module_idxs = self.neuron_selection(
            model, layer, accumulator, trained_params_per_model, device,
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
            # err_values is sorted by module_idx (not by error), and the retrain
            # path replaces it with an unsorted concat, so take the minimum
            # explicitly instead of trusting position 0.
            layer.err = err_values.min().item()
        elif model.param.train.layer_err_criterion == 'avg':
            layer.err = err_values.mean().item()
        else:
            raise NotImplementedError
        logger.info(f"Current layer error: {layer.err:.3f}")
        logger.info(f"Layer errors: {abbrev_floats([l.err for l in model.layers])}")

        # Optional per-layer joint fine-tune (surviving neurons' polynomial
        # weights + temporary (D, K) head trained end-to-end on CE). Runs
        # BEFORE the layer-completed checkpoint save so the persisted state
        # captures the fine-tuned weights. Skipped on the planned last layer
        # (max_layer_count - 1) because train_out_proj — which runs right
        # after train() returns — fits the real (D, K) head against this
        # layer's outputs anyway, so the temporary head fit here would just
        # be thrown away. Early-stopped runs that exit before the planned
        # last layer still get the fine-tune on whatever ended up being
        # final.
        if model.param.train.layer_finetune:
            self._train_layer_finetune(model, layer, train_dl, dev_dl)

        if last_ckpt is not None:
            layer_end_ckpt = replace(last_ckpt, layer_completed=True,
                err=list(accumulator.err), module_idxs=list(accumulator.module_idxs))
            self.save_checkpoint(layer_end_ckpt)
            del layer_end_ckpt

        # Release per-layer GPU state before the next layer's _precompute_dl
        # runs — otherwise trained_params_per_model (full ensemble params for
        # every candidate neuron model) and the optimizer state inside
        # last_ckpt stay resident in the CUDA caching allocator and stack up
        # layer over layer until OOM.
        del trained_params_per_model, accumulator, last_ckpt
        del train_feat_dl, dev_feat_dl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_z_val(
            self,
            neuron_model: BasePolynomNeuron,
            model: SONN,
            dev_dl: DataLoader,
            device: torch.device | str,
            skip_model_fwd: bool,
    ) -> torch.Tensor:
        """Per-candidate raw scalar outputs on the full dev split, shape (C, N_val).

        Used by neuron_selection's OMP-style decorrelation to skip candidates
        whose scalar response on the validation set is nearly collinear with
        already-selected survivors. Reads `neuron_model.weight` directly, so
        it must be called AFTER `train_model_ensemble` has written the trained
        ensemble weights back via `write_back_params`. Multi-class projection
        is intentionally bypassed — the raw polynomial scalar is what feeds
        the next layer and is the right space to decorrelate in.
        """
        chunks: list[torch.Tensor] = []
        neuron_model.eval()
        with torch.inference_mode():
            for x, _ in self._iter_eval_batches(model, dev_dl, device, skip_model_fwd):
                z = neuron_model(x)  # (batch, C)
                chunks.append(z.transpose(0, 1))  # (C, batch)
        return torch.cat(chunks, dim=1)  # (C, N_val)

    def neuron_selection(
            self,
            model: SONN,
            layer: SONNLayer,
            accumulator: LayerAccumulator,
            trained_params_per_model: dict[int, dict],
            device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick the nbest survivors across the layer's neuron-model ensemble and
        prune the rest in place.

        Concatenates per-neuron-model errors and module indices from the
        accumulator, drops NaNs, takes top-k (lowest error) up to
        model.nbest_neurons, re-sorts by module index, prunes each
        neuron_model down to its surviving rows, and restores model.shared_proj
        from the best neuron's trained params (shared-proj path only).

        Mutates `layer.neuron_models`, `layer.d_model`, and optionally
        `model.shared_proj` weight/bias in place. Returns the (err_values,
        module_idxs) tuple the caller attaches to `layer.err_values` /
        `layer.module_idxs`.
        """
        # make module_idxs been 2D tensor with rows correspond module idx and columns correspond to neuron model idx
        err = torch.cat(accumulator.err)
        module_idxs = torch.cat(accumulator.module_idxs)
        z_val = torch.cat(accumulator.z_val, dim=0)  # NEW: (C_total, N_val)

        # filter out nan values
        nan_idx = ~torch.isnan(err)
        err = err[nan_idx]
        module_idxs = module_idxs[nan_idx]
        z_val = z_val[nan_idx]

        nbest = min(err.shape[0], model.nbest_neurons)
        method = model.param.train.neuron_selection_method
        EPS = 1e-8

        if method == "plain":
            # Historical default: top-k by individual error.
            err_values, err_idxs = torch.topk(err, k=nbest, dim=0, largest=False)

        elif method == "omp_mixed":
            # Walk candidates in error order (best-error first) and keep any
            # whose residual after Gram-Schmidt against already-selected
            # survivors still carries enough novel signal on the dev set.
            orth_threshold = float(model.param.train.neuron_selection_orth_threshold)
            sort_order = torch.argsort(err)
            selected_local_idxs: list[int] = []
            basis: list[torch.Tensor] = []  # already-selected residuals
            for ci in sort_order.tolist():
                if len(selected_local_idxs) >= nbest:
                    break
                z = z_val[ci] - z_val[ci].mean()
                original_norm = z.norm().clamp_min(EPS)
                for b in basis:
                    proj = (z * b).sum() / (b * b).sum().clamp_min(EPS)
                    z = z - proj * b
                if z.norm() < orth_threshold * original_norm:
                    continue
                selected_local_idxs.append(ci)
                basis.append(z)
            err_idxs = torch.tensor(selected_local_idxs, device=err.device, dtype=torch.long)
            err_values = err[err_idxs]

        elif method == "omp":
            # Classical Orthogonal Matching Pursuit, vectorized over the
            # candidate axis. At every step, pick the candidate whose
            # orthogonalized residual has the largest norm, then project that
            # direction out of all remaining residuals. Pre-center each
            # candidate's dev outputs once — we score variance, not magnitude.
            R = z_val - z_val.mean(dim=1, keepdim=True)  # (C, N_val)
            C = R.size(0)
            available = torch.ones(C, dtype=torch.bool, device=R.device)
            selected_local_idxs = []
            neg_inf = torch.full((), float("-inf"), device=R.device, dtype=R.dtype)
            for _ in range(nbest):
                norms = R.norm(dim=1)
                norms = torch.where(available, norms, neg_inf)
                best_ci = int(torch.argmax(norms).item())
                if norms[best_ci].item() < EPS:
                    break
                selected_local_idxs.append(best_ci)
                available[best_ci] = False
                b = R[best_ci].clone()
                b_norm_sq = (b * b).sum().clamp_min(EPS)
                coefs = (R @ b) / b_norm_sq               # (C,)
                R = R - coefs.unsqueeze(1) * b            # (C, N_val)
            err_idxs = torch.tensor(selected_local_idxs, device=err.device, dtype=torch.long)
            err_values = err[err_idxs]

        else:
            raise NotImplementedError(
                f"Unknown neuron_selection_method: {method!r}; "
                "expected one of 'plain', 'omp_mixed', 'omp'."
            )

        topk_module_idxs = module_idxs[err_idxs]
        # Selection may produce fewer than the requested nbest (e.g. all
        # remaining residuals fell below EPS); keep the rest of the function
        # in sync with how many survivors were actually picked.
        nbest = err_idxs.shape[0]

        # Record the module-idx and within-module local-idx of the best neuron
        # (position 0 = smallest error) BEFORE the re-sort by module index.
        # Used below to copy the correct projection back to model.shared_proj /
        # model.neuron_proj so infer() uses the trained state.
        best_neuron_module_idx: int | None = int(topk_module_idxs[0, 0].item()) if nbest > 0 else None

        # now sort by module idx
        sort_indices = torch.sort(topk_module_idxs[:, 0], stable=True)[1]
        topk_module_idxs = topk_module_idxs[sort_indices]
        err_values = err_values[sort_indices]

        # select nbest neurons from the list,
        # delete unused neurons keeping only nbest neurons
        new_modules = []
        new_module_idxs: list[torch.Tensor] = []
        new_module_idx = 0
        for neuron_model_idx, neuron_model in enumerate(layer.neuron_models):
            cur_module_idx = topk_module_idxs[topk_module_idxs[:, 0] == neuron_model_idx]
            if cur_module_idx.shape[0] > 0:
                neuron_model.prune(cur_module_idx[:, 1])
                new_modules.append(neuron_model)
                new_module_idxs.append(
                    torch.stack([
                        torch.empty(cur_module_idx.shape[0], dtype=torch.long, device=device).fill_(new_module_idx),
                        torch.arange(cur_module_idx.shape[0], device=device)],
                        dim=1)
                )
                new_module_idx += 1
        module_idxs = torch.cat(new_module_idxs)
        layer.neuron_models = NeuronModuleList(new_modules)
        # Keep d_model in sync with the actual post-prune output count so the
        # next create_layer call uses the right number of inputs.
        layer.d_model = len(layer)
        # Build the per-layer LayerNorm with the post-shortcut-cat width so
        # SONN.forward can fold the raw input features into the normalization
        # alongside the layer's own outputs. No-op when use_layer_norm is off.
        ln_dim = layer.d_model + (model.d_model if model.param.model.shortcut else 0)
        layer.setup_layer_norm(ln_dim)

        # Restore the shared_proj that was co-trained with the selected neurons.
        # write_back_params only updates neuron_model.weight; shared_proj params
        # live on the model and need a separate copy from the best-neuron's
        # trained_params_batch so infer() uses the correct trained state.
        # neuron_proj params (proj_weight/proj_bias) live on the neuron modules
        # themselves and are updated by write_back_params automatically.
        if best_neuron_module_idx is not None:
            best_trained = trained_params_per_model.get(best_neuron_module_idx)
            if best_trained is not None and model.shared_proj is not None:
                if "shared_proj_weight" in best_trained:
                    with torch.no_grad():
                        model.shared_proj.weight.copy_(best_trained["shared_proj_weight"])
                        model.shared_proj.bias.copy_(best_trained["shared_proj_bias"])

        if model.retrain_required:
            raise NotImplementedError

        return err_values, module_idxs

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
        dist_slice = self._ensemble_slice(ensemble_size)
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

        opt = optimizer_cls(params_batch, shared_param_names, shared_param_lr_multiplier=shared_param_lr_multiplier, **optimizer_params)
        if model.param.train.scheduler.name is None:
            scheduler = None
        else:
            scheduler_cls = scheduler_map[model.param.train.scheduler.name]
            scheduler = scheduler_cls(opt, **model.param.train.scheduler.scheduler_params)

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
            if scheduler:
                scheduler.load_state_dict(checkpoint_data["scheduler"])

        train_dl_len = len(train_dl)
        step = 0
        # One persistent bar tracking the global step budget. The previous
        # design wrapped the inner DataLoader in tqdm, so with batch_size that
        # collapses an epoch to 1 batch (LBFGS / full-batch GD) every epoch
        # blasted the bar to 100% and printed a fresh line. A single
        # outer-level bar advanced by 1 per optimizer step gives a stable,
        # Lightning-style single-line readout regardless of batches per epoch.
        _rank0 = not self._is_dist() or dist.get_rank() == 0
        tbar = tqdm(
            total=int(model.param.train.steps),
            initial=int(global_step),
            file=sys.stdout,
            ncols=140,
            desc="Training",
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
                self._allreduce_shared_grads(grads, shared_param_names)
                params_batch = opt.step(params_batch, grads, active_mask=~early_stop_flags)

                if scheduler:
                    scheduler.step()
                    # Scheduler reassigns opt.lr to a fresh tensor; re-sync our view.
                    lr = opt.lr

                if global_step > 0 and ((eval_step_interval != -1 and global_step % eval_step_interval == 0) or (eval_step_interval == -1 and step == train_dl_len - 1)):

                    model.eval()
                    with torch.inference_mode():
                        val_losses = self.ds_loss(model, eval_loss_fn_vmapped, params_batch, buffers_batch, dev_dl, device, features_precomputed)

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

                    # Number of completed models (all-reduce across ranks when distributed
                    # so the count reflects the full ensemble, not just the local slice).
                    _local_done = early_stop_flags.sum()
                    if self._is_dist():
                        dist.all_reduce(_local_done, op=dist.ReduceOp.SUM)
                    completed_models = int(_local_done.item())
                    completion_percentage = 100.0 * completed_models / ensemble_size

                    # Update tqdm description with current step and val loss
                    val_losses_nonan = val_losses[~torch.isnan(val_losses)]
                    if val_losses_nonan.numel() > 0:
                        val_loss_mean = val_losses_nonan.mean().item()
                        val_loss_min = val_losses_nonan.min().item()
                        val_loss_max = val_losses_nonan.max().item()
                    else:
                        val_loss_mean = val_loss_min = val_loss_max = float("nan")
                    # Update tqdm postfix
                    tbar.set_postfix({
                        "epoch": epoch,
                        "step": global_step,
                        "lr": f"{lr.mean().item():.2e}",
                        "val_loss": f"[{val_loss_min:.3f}, {val_loss_mean:.3f}, {val_loss_max:.3f}]",
                        "completed": f"{completed_models}/{ensemble_size}"
                    }, refresh=True)

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

                    # All-reduce the "all stopped" flag so every rank agrees to stop simultaneously.
                    _all_stopped = early_stop_flags.all()
                    if self._is_dist():
                        _all_t = _all_stopped.to(torch.uint8)
                        dist.all_reduce(_all_t, op=dist.ReduceOp.MIN)
                        _all_stopped = _all_t.bool()
                    if _all_stopped or completion_percentage > model.param.train.early_stop_completion_percentage or global_step > model.param.train.steps:
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

                save_interval = model.param.train.save_interval
                if (save_interval != -1 and global_step % save_interval == 0) or (step == train_dl_len - 1 and not model.param.train.skip_saving_at_epoch_end):
                    _ckpt_params = self._gather_ensemble_params(params_batch, shared_param_names) if dist_slice else params_batch
                    self.write_back_params(neuron_model, _ckpt_params)
                    self._write_back_shared_params(model, _ckpt_params)
                    last_ckpt = StepCheckpoint(
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
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
                    if not self._is_dist() or dist.get_rank() == 0:
                        self.save_checkpoint(last_ckpt)

            epoch += 1
            if stop:
                break
        tbar.close()

        # Restore the full ensemble shape before write-back and error evaluation.
        params_batch = self._gather_ensemble_params(params_batch, shared_param_names) if dist_slice else params_batch
        self.write_back_params(neuron_model, params_batch)
        self._write_back_shared_params(model, params_batch)
        last_ckpt = StepCheckpoint(
            model=model,
            opt=opt,
            scheduler=scheduler,
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
        if not self._is_dist() or dist.get_rank() == 0:
            self.save_checkpoint(last_ckpt)
            if model.param.train.save_last_layer:
                self.save_checkpoint(last_ckpt, suffix="last")
        if checkpoint_data:
            checkpoint_data["neuron_model_completed"] = True

        # Drop optimizer / scheduler / gradient state before returning so the
        # next neuron model in the same layer starts from a clean slate. We
        # cannot drop params_batch (it is returned) or last_ckpt (it is returned).
        del opt, scheduler, buffers_batch
        return accumulator, params_batch, last_ckpt

    @staticmethod
    def _write_back_shared_params(model: "SONN", params_batch: dict) -> None:
        """Copy shared_proj params back into model.shared_proj after each optimizer step."""
        if model.shared_proj is not None and "shared_proj_weight" in params_batch:
            with torch.no_grad():
                model.shared_proj.weight.copy_(params_batch["shared_proj_weight"])
                model.shared_proj.bias.copy_(params_batch["shared_proj_bias"])

    @classmethod
    def write_back_params(
        cls,
        model: nn.Module,
        new_params: dict[str, torch.Tensor],
        new_buffers: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Copy updated batched parameters & buffers into the model.

        Args:
            model: nn.Module
            new_params: dict[str, Tensor] – updated batched parameters
            new_buffers: dict[str, Tensor] – updated buffers (optional)
        """
        # Write back parameters
        for name, param in model.named_parameters():
            if name in new_params:
                with torch.no_grad():
                    param.copy_(new_params[name])

        # Write back buffers
        if new_buffers is not None:
            for name, buf in model.named_buffers():
                if name in new_buffers:
                    buf.copy_(new_buffers[name])

    def create_loss_functions(
        self, model: SONN, module: BasePolynomNeuron,
    ) -> tuple[Callable, Callable, Callable, dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
        # prepare params & buffers as dicts
        params = dict(module.named_parameters())
        buffers = dict(module.named_buffers())
        buffers["src_idxs"] = module.src_idxs

        params_batch = params
        buffers_batch = buffers

        # set in_dims for dicts
        params_in_dims = {k: 0 for k in params_batch}
        buffers_in_dims = {k: 0 for k in buffers_batch}

        if isinstance(model.loss_fn, nn.NLLLoss) and model.soft_binner is None and model.shared_proj is not None:
            # shared_proj is trained jointly by all ensemble members (in_dims=None).
            # The optimizer's shared path averages their gradients into one update.
            # use_neuron_proj: proj_weight/proj_bias live on the neuron module and
            # are already in params via named_parameters() with in_dims=0.
            params["shared_proj_weight"] = model.shared_proj.weight
            params["shared_proj_bias"] = model.shared_proj.bias
            params_in_dims["shared_proj_weight"] = None
            params_in_dims["shared_proj_bias"] = None
            shared_param_names = ["shared_proj_weight", "shared_proj_bias"]
        else:
            shared_param_names = []

        # Ridge α applies to the training-loss partial only. The eval and
        # pred partials are pinned to 0.0 so the dev loss used for
        # neuron_selection, the layer-error reported in the log, and the
        # final R² / RMSE in report() all reflect pure data fit, not the
        # penalised training objective. This is the conventional ridge
        # tuning regime: "train regularized, evaluate raw".
        ridge_alpha = float(model.param.train.ridge_alpha)

        # create loss function that is vmapped & differentiated
        loss_fn_vmapped = vmap(
            grad(partial(model.compute_loss, module, model.loss_fn, model.param.model.type, model.soft_binner, ridge_alpha)),
            in_dims=(params_in_dims, buffers_in_dims, None, None)
        )

        # Create vmapped evaluation loss function (no gradient)
        eval_loss_fn_vmapped = vmap(
            partial(model.compute_loss, module, model.loss_fn, model.param.model.type, model.soft_binner, 0.0),
            in_dims=(params_in_dims, buffers_in_dims, None, None)
        )

        pred_fn_vmapped = vmap(
            partial(model.compute_loss, module, None, model.param.model.type, model.soft_binner, 0.0),
            in_dims=(params_in_dims, buffers_in_dims, None, None)
        )

        return loss_fn_vmapped, eval_loss_fn_vmapped, pred_fn_vmapped, params_batch, buffers_batch, shared_param_names

    def _precompute_dl(
        self,
        model: SONN,
        dl: DataLoader,
        device: torch.device | str,
        skip_last_layer: bool = True,
    ) -> DataLoader:
        """Run the frozen model over the full DataLoader once and cache features.

        skip_last_layer=True  → model(x, skip_last_layer=True): used by train_layer
                                 so the candidate ensemble sees frozen-prefix outputs.
        skip_last_layer=False → model(x): used by train_out_proj / train_finetune
                                 so the output head trains on full model features.

        Returns a new DataLoader yielding (features, targets).  Pass with
        skip_model_fwd=True so _iter_eval_batches skips the redundant forward.
        """
        all_x, all_targets = [], []
        model.eval()
        with torch.inference_mode():
            for batch in dl:
                x_inp, targets = self.batch_callback(batch) if self.batch_callback else batch
                x = model(x_inp.to(device=device), skip_last_layer=skip_last_layer)
                all_x.append(x.cpu())
                all_targets.append(targets)
        new_ds = SONNDataset(torch.cat(all_x, dim=0), torch.cat(all_targets, dim=0))
        shuffle = isinstance(dl.sampler, torch.utils.data.RandomSampler)
        return DataLoader(new_ds, batch_size=dl.batch_size, shuffle=shuffle,
                          num_workers=dl.num_workers, drop_last=dl.drop_last)

    def _iter_eval_batches(self, model: SONN, dl: DataLoader, device: torch.device | str,
                           skip_model_fwd: bool = False):
        """Shared eval-loop boilerplate for ds_loss / regularity_err / bias_err.

        Yields (x, targets) per batch where:
          * both tensors are on `device`
          * `x` is the SONN forward output with `skip_last_layer=True`
          * if skip_model_fwd=True the dl already contains pre-transformed
            features (from _precompute_dl) — the model forward is skipped.
        """
        for batch in tqdm(dl, total=len(dl), disable=True):
            x_inp, targets = self.batch_callback(batch) if self.batch_callback else batch
            x_inp = x_inp.to(device=device)
            targets = targets.to(device=device)
            if skip_model_fwd:
                yield x_inp, targets
            else:
                with torch.inference_mode():
                    x = model(x_inp, skip_last_layer=True)
                    yield x, targets

    def ds_loss(
            self,
            model: SONN,
            eval_loss_fn_vmapped: Callable,
            params_batch: dict[str, torch.Tensor],
            buffers_batch: dict[str, torch.Tensor],
            dl: DataLoader,
            device: torch.device | str,
            skip_model_fwd: bool = False) -> torch.Tensor:
        err_nom, err_denom = [], []
        for x, targets in self._iter_eval_batches(model, dl, device, skip_model_fwd):
            val_losses = eval_loss_fn_vmapped(params_batch, buffers_batch, x, targets)
            err_nom.append(val_losses)
            err_denom.append(torch.ones_like(val_losses))
        # Element-wise divide: per-neuron mean over batches.
        return torch.stack(err_nom, dim=1).sum(dim=-1) / torch.stack(err_denom, dim=1).sum(dim=-1)

    def regularity_err(
            self,
            model: SONN,
            pred_fn_vmapped: Callable,
            params_batch: dict[str, torch.Tensor],
            buffers_batch: dict[str, torch.Tensor],
            dl: DataLoader,
            device: torch.device | str,
            skip_model_fwd: bool = False) -> torch.Tensor:
        """Per-candidate regularity criterion, computed streaming per batch.

        Regression / binary: normalized MSE (matches regularity_error).
        Multi-class:         normalized cross-entropy (matches regularity_error_ce).

        The previous implementation concatenated per-batch preds into a
        (ensemble, N, K) tensor before calling the helpers, which for Otto-
        sized ensembles (~17k candidates after shortcut) materialized a 6+ GB
        tensor on GPU and then doubled it inside log_softmax — the OOM site
        at layer 5. Per-batch accumulation keeps GPU residency to a single
        batch's predictions.
        """
        eps = 1e-12

        if model.param.model.type == "multi-class":
            # streaming NCE: accumulate sum of -log p[i, y_i] across batches,
            # then divide by total N and by H(Y_B) from concatenated targets.
            sum_nll: torch.Tensor | None = None
            all_targets: list[torch.Tensor] = []
            total_N = 0
            K: int | None = None
            for x, targets in self._iter_eval_batches(model, dl, device, skip_model_fwd):
                logits = pred_fn_vmapped(params_batch, buffers_batch, x, targets)
                K = logits.size(-1)
                log_p = F.log_softmax(logits, dim=-1)
                del logits
                n_batch = targets.size(0)
                # broadcast targets across the leading candidate dims for gather along K
                y_idx = targets.view(*([1] * (log_p.dim() - 2)), n_batch, 1) \
                                .expand(*log_p.shape[:-1], 1)
                nll_batch = -log_p.gather(-1, y_idx).squeeze(-1)
                del log_p, y_idx
                partial = nll_batch.sum(dim=-1)
                del nll_batch
                sum_nll = partial if sum_nll is None else sum_nll + partial
                total_N += n_batch
                all_targets.append(targets)

            assert sum_nll is not None and K is not None, "empty dataloader passed to regularity_err"
            targets_cat = torch.cat(all_targets, dim=0)
            counts = torch.bincount(targets_cat, minlength=K).float()
            p_marg = counts / counts.sum()
            H_Y = torch.special.entr(p_marg).sum().clamp_min(eps)
            ce_model = sum_nll / total_N
            return ce_model / H_Y
        else:
            # streamed normalized MSE
            num: torch.Tensor | None = None
            denom_sum = torch.zeros((), device=device, dtype=torch.float32)
            for x, targets in self._iter_eval_batches(model, dl, device, skip_model_fwd):
                preds = pred_fn_vmapped(params_batch, buffers_batch, x, targets)
                diff_sq = ((preds - targets) ** 2).sum(dim=-1)
                del preds
                num = diff_sq if num is None else num + diff_sq
                denom_sum = denom_sum + (targets.to(denom_sum.dtype) ** 2).sum()
            assert num is not None, "empty dataloader passed to regularity_err"
            return num / denom_sum.clamp_min(eps)

    def bias_err(
            self,
            model: SONN,
            pred_fn_vmapped_a: Callable,
            pred_fn_vmapped_b: Callable,
            params_batch_a: dict[str, torch.Tensor],
            params_batch_b: dict[str, torch.Tensor],
            buffers_batch_a: dict[str, torch.Tensor],
            buffers_batch_b: dict[str, torch.Tensor],
            dl_a: DataLoader,
            dl_b: DataLoader,
            device: torch.device | str,
            bias_method: str = "l2",
            skip_model_fwd: bool = False,
    ) -> torch.Tensor:
        """Bias error between two neurons fitted on disjoint splits.

        Regression / binary: bias_error — normalized squared disagreement on scalars.
        Multi-class, bias_method="l2": bias_error_l2 on logits (ensemble, N, K).
        Multi-class, bias_method="js": bias_error_js on logits (ensemble, N, K).
        pred_fn_vmapped returns (ensemble, N) for regression and (ensemble, N, K)
        for multi-class, so no separate raw-pred function is needed.
        Both splits are concatenated before calling the criterion function.
        """
        is_multiclass = model.param.model.type == "multi-class"

        def _collect(dl):
            all_a, all_b, all_t = [], [], []
            for x, targets in self._iter_eval_batches(model, dl, device, skip_model_fwd):
                all_a.append(pred_fn_vmapped_a(params_batch_a, buffers_batch_a, x, targets))
                all_b.append(pred_fn_vmapped_b(params_batch_b, buffers_batch_b, x, targets))
                all_t.append(targets)
            return (
                torch.cat(all_a, dim=1),
                torch.cat(all_b, dim=1),
                torch.cat(all_t, dim=0),
            )

        a1, b1, t1 = _collect(dl_a)
        a2, b2, t2 = _collect(dl_b)
        preds_a = torch.cat([a1, a2], dim=1)
        preds_b = torch.cat([b1, b2], dim=1)

        if is_multiclass and bias_method == "js":
            return bias_error_js(preds_a, preds_b)
        elif is_multiclass:  # bias_method == "l2"
            return bias_error_l2(preds_a, preds_b)
        else:  # regression / binary
            return bias_error(preds_a, preds_b, torch.cat([t1, t2], dim=0))

    # endregion

    def train_out_proj(self, model: SONN, train_dl: DataLoader, dev_dl: DataLoader) -> None:
        """Train model.out_proj with all other parameters frozen.

        Runs a standard PyTorch training loop — no vmapped ensemble, no custom
        optimizer — because out_proj is a single shared nn.Linear whose gradient
        comes from the full batch in one shot.

        Stops at cfg.max_steps, or earlier if the early-stop criterion fires.
        ReduceLROnPlateau drops the LR when val loss stagnates; if it has
        already hit lr_min and val loss still hasn't improved for
        early_stop_patience consecutive evaluations, training halts.
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
            self._train_out_proj_lbfgs(model, train_dl, dev_dl, cfg)
            for p in model.parameters():
                p.requires_grad_(True)
            # Same persistence as the Adam/SGD path — see note below tbar.close().
            self.save_model_checkpoint(model)
            return

        if cfg.optimizer == "adam":
            opt = torch.optim.Adam(
                model.out_proj.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "sgd":
            opt = torch.optim.SGD(
                model.out_proj.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported out_proj optimizer: {cfg.optimizer!r}")

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
        train_feat_dl = self._precompute_dl(model, train_dl, device, skip_last_layer=False)
        dev_feat_dl   = self._precompute_dl(model, dev_dl,   device, skip_last_layer=False)

        # Branch on model.type once so the inner loop's loss step doesn't.
        # Multi-class uses log_softmax + NLL; regression uses the raw scalar
        # output + whatever loss_fn was configured (NormMSE for the regressor
        # SONN ctor path).
        is_regressor = model.param.model.type == "regressor"

        def _head_loss(features_batch: torch.Tensor, targets_batch: torch.Tensor) -> torch.Tensor:
            sel = self._select_out_proj_inputs(features_batch, cols_t, k)
            proj = model.out_proj(sel)
            if is_regressor:
                return model.loss_fn(proj.squeeze(-1), targets_batch).mean()
            log_probs = F.log_softmax(proj, dim=-1)
            return model.loss_fn(log_probs, targets_batch).mean()

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
                model.out_proj.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for vfeatures, vt in dev_feat_dl:
                        vfeatures = vfeatures.to(device=device)
                        vt        = vt.to(device=device)
                        val_loss += _head_loss(vfeatures, vt).item()

                val_loss /= len(dev_feat_dl)
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

        # Restore gradients on all parameters so subsequent calls work normally.
        for p in model.parameters():
            p.requires_grad_(True)

        # Persist the trained out_proj into model_last.ckpt so a subsequent
        # load_model_checkpoint() call doesn't silently revert to the random
        # init that was saved at the end of train(). Without this every
        # downstream evaluation runs on a never-trained Linear head.
        self.save_model_checkpoint(model)

    def _train_out_proj_lbfgs(self, model: SONN, train_dl: DataLoader, dev_dl: DataLoader, cfg) -> None:
        """Full-batch LBFGS path for train_out_proj.

        Materializes all (precomputed) train and dev features into single
        tensors so each closure call sees the exact convex gradient — this is
        what makes LBFGS work the way sklearn does. Outer step count is
        cfg.max_steps; each opt.step(closure) runs cfg.lbfgs_max_iter inner
        line-search iterations. weight_decay (if > 0) is added inside the
        closure as a manual L2 penalty.
        """
        device = model.device
        assert model.out_proj is not None  # checked by caller

        line_search = cfg.lbfgs_line_search or None
        opt = torch.optim.LBFGS(
            model.out_proj.parameters(),
            lr=cfg.lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=cfg.lbfgs_history_size,
            line_search_fn=line_search,
        )

        k = model.out_proj.in_features
        cols = model._best_neuron_columns(model.layers[-1], k)
        cols_t = torch.tensor(cols, dtype=torch.long, device=device)

        # Precompute frozen SONN features once; LBFGS then reads them every closure.
        train_feat_dl = self._precompute_dl(model, train_dl, device, skip_last_layer=False)
        dev_feat_dl   = self._precompute_dl(model, dev_dl,   device, skip_last_layer=False)

        # Concatenate the whole dataset into one full-batch tensor per split.
        def _cat_split(dl):
            xs, ys = [], []
            for features, targets in dl:
                xs.append(features.to(device=device))
                ys.append(targets.to(device=device))
            return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

        train_X, train_y = _cat_split(train_feat_dl)
        dev_X,   dev_y   = _cat_split(dev_feat_dl)
        train_X_sel = self._select_out_proj_inputs(train_X, cols_t, k)
        dev_X_sel   = self._select_out_proj_inputs(dev_X,   cols_t, k)
        del train_X, dev_X

        verbose = model.param.train.verbose
        tbar = tqdm(total=cfg.max_steps, desc="out_proj (lbfgs)", file=sys.stdout,
                    ncols=140, disable=not verbose)

        best_val_loss = float("inf")
        evals_no_improve = 0
        wd = float(cfg.weight_decay)
        # Multi-class uses log_softmax + NLL; regression uses the raw scalar
        # output + whatever loss_fn was configured (NormMSE).
        is_regressor = model.param.model.type == "regressor"

        def _head_loss(x_sel: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            proj = model.out_proj(x_sel)
            if is_regressor:
                return model.loss_fn(proj.squeeze(-1), y).mean()
            return model.loss_fn(F.log_softmax(proj, dim=-1), y).mean()

        def closure():
            opt.zero_grad()
            loss = _head_loss(train_X_sel, train_y)
            if wd > 0.0:
                # Manual L2 — PyTorch's LBFGS does not accept weight_decay.
                l2 = sum((p ** 2).sum() for p in model.out_proj.parameters())
                loss = loss + 0.5 * wd * l2
            loss.backward()
            return loss

        for step in range(1, cfg.max_steps + 1):
            model.out_proj.train()
            train_loss_t = opt.step(closure)
            train_loss_val = float(train_loss_t.item()) if torch.is_tensor(train_loss_t) else float(train_loss_t)

            if step % cfg.eval_interval != 0:
                tbar.update(1)
                continue

            model.out_proj.eval()
            with torch.no_grad():
                val_loss = _head_loss(dev_X_sel, dev_y).item()

            if val_loss < best_val_loss - cfg.early_stop_min_delta:
                best_val_loss = val_loss
                evals_no_improve = 0
            else:
                evals_no_improve += 1

            tbar.set_postfix({
                "train_loss": f"{train_loss_val:.4f}",
                "val_loss":   f"{val_loss:.4f}",
                "best":       f"{best_val_loss:.4f}",
                "no_imp":     evals_no_improve,
            })
            tbar.update(1)

            if evals_no_improve >= cfg.early_stop_patience:
                tbar.total = tbar.n
                tbar.refresh()
                logger.info(f"out_proj (lbfgs) early stop at step {step} — "
                            f"no improvement for {evals_no_improve} evals")
                break

        tbar.close()

    @staticmethod
    def _select_out_proj_inputs(
        features: torch.Tensor, cols_t: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Select and optionally zero-pad columns for out_proj input."""
        selected = features[:, cols_t]
        if selected.shape[1] < k:
            pad = torch.zeros(selected.shape[0], k - selected.shape[1],
                              device=features.device, dtype=features.dtype)
            selected = torch.cat([selected, pad], dim=-1)
        return selected

    def train_finetune(self, model: SONN, train_dl: DataLoader, dev_dl: DataLoader) -> None:
        """End-to-end fine-tuning of all SONN parameters.

        After SONN's structural search is done every layer is a normal
        differentiable module — gradients flow through BasePolynomNeuron.weight,
        shared_proj / soft_binner / neuron_proj, and out_proj all at once.

        Unlike out_proj training the full model forward runs on every step
        (frozen-feature precomputation is not applicable here).  Use a large
        batch size and/or torch.compile(model) to keep the GPU busy.

        Reuses OutProjTrainConfig (model.param.train.out_proj_train) for LR /
        scheduler / early-stop hyperparameters.
        """
        cfg    = model.param.train.out_proj_train
        device = model.device

        for p in model.parameters():
            p.requires_grad_(True)

        if cfg.optimizer == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer!r}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
            min_lr=cfg.lr_min, threshold=cfg.early_stop_min_delta,
        )

        verbose = model.param.train.verbose
        tbar = tqdm(total=cfg.max_steps, desc="finetune", file=sys.stdout,
                    ncols=140, disable=not verbose)

        step = 0
        best_val_loss = float("inf")
        evals_no_improve = 0
        stop = False
        train_loss_accum = 0.0
        train_steps_accum = 0

        while step < cfg.max_steps and not stop:
            model.train()

            for batch in train_dl:
                if step >= cfg.max_steps:
                    break

                x_inp, targets = self.batch_callback(batch) if self.batch_callback else batch
                x_inp   = x_inp.to(device=device)
                targets = targets.to(device=device)

                log_probs = model.infer(x_inp)
                loss      = model.loss_fn(log_probs, targets).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss_accum += loss.item()
                train_steps_accum += 1
                step += 1

                if step % cfg.eval_interval != 0:
                    continue

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for vbatch in dev_dl:
                        vx, vt = self.batch_callback(vbatch) if self.batch_callback else vbatch
                        val_loss += model.loss_fn(
                            model.infer(vx.to(device=device)), vt.to(device=device)
                        ).mean().item()

                val_loss /= len(dev_dl)
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
                    logger.info(f"finetune early stop at step {step} — "
                                f"no improvement for {evals_no_improve} evals")
                    stop = True
                    break

                model.train()

        tbar.close()

    def _train_layer_finetune(
            self,
            model: SONN,
            layer: SONNLayer,
            train_dl: DataLoader,
            dev_dl: DataLoader,
    ) -> None:
        """Per-layer joint fine-tune that runs after neuron_selection.

        Trains the surviving neurons' polynomial `weight`s together with a
        temporary `nn.Linear(layer.d_model, num_classes)` head against the
        model's class-weighted CE on the dev split. The temporary head exists
        only for this pass — it's discarded once fine-tuning completes; the
        proper classification head (out_proj / shared_proj / neuron_proj) is
        trained separately at end of run.

        Implementation mirrors `train_out_proj`: pre-computes layers 0..N-1
        features once (the current layer was already appended to model.layers
        before train_layer ran, so `skip_last_layer=True` strips exactly the
        layer we want to fine-tune), then iterates the per-layer + head path
        on the cached features. Only `nm.weight` is unfrozen per neuron —
        proj_weight / proj_bias are intentionally left alone since the loss
        doesn't flow through them in this pass.
        """
        # Reuse the out_proj_train hyperparam block — same shape, same
        # adam/sgd/lbfgs + ReduceLROnPlateau knobs. Keeps the YAML surface
        # small; if you need different settings per pass, the easiest path
        # is to clone the block before launching.
        cfg         = model.param.train.out_proj_train
        device      = model.device
        # Multi-class uses log_softmax + NLL → head outputs num_classes
        # logits. Regression / binary uses raw scalar output + the model's
        # configured loss_fn (NormMSE for regressor) → head outputs 1.
        is_regressor = model.param.model.type == "regressor"
        head_out_dim = 1 if is_regressor else model.param.model.num_classes

        # Freeze the world, then unfreeze exactly the parameters this pass
        # is allowed to touch: the surviving neurons' polynomial weights and
        # the temporary head. Restored at the end of the function.
        for p in model.parameters():
            p.requires_grad_(False)
        head = torch.nn.Linear(layer.d_model, head_out_dim).to(device=device)
        trainable_params: list[torch.nn.Parameter] = list(head.parameters())
        for nm in layer.neuron_models:
            nm.weight.requires_grad_(True)
            trainable_params.append(nm.weight)

        # Features through layers 0..N-1. Current `layer` is model.layers[-1],
        # so skip_last_layer=True is "everything but this one".
        train_feat_dl = self._precompute_dl(model, train_dl, device, skip_last_layer=True)
        dev_feat_dl   = self._precompute_dl(model, dev_dl,   device, skip_last_layer=True)

        # LBFGS runs full-batch with a closure — see the dedicated helper.
        # It still calls the same setup above (precomputed features +
        # trainable_params + head), so we hand them off rather than duplicate
        # the freeze/precompute boilerplate.
        if cfg.optimizer == "lbfgs":
            self._train_layer_finetune_lbfgs(
                model, layer, head, trainable_params,
                train_feat_dl, dev_feat_dl, cfg,
            )
            for p in model.parameters():
                p.requires_grad_(True)
            return

        if cfg.optimizer == "adam":
            opt = torch.optim.Adam(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "sgd":
            opt = torch.optim.SGD(
                trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer for layer_finetune: {cfg.optimizer!r} "
                "(supported: 'adam', 'sgd', 'lbfgs')."
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
            min_lr=cfg.lr_min, threshold=cfg.early_stop_min_delta,
        )

        clamp = model.param.model.output_clamp_value

        def fwd(features: torch.Tensor) -> torch.Tensor:
            """Replicates SONN.forward's per-layer path for `layer` only,
            treating it as the last layer (no shortcut concat, hence no
            post-cat LayerNorm) — matches how the eventual classification
            head will see this layer's output at inference."""
            z = layer(features)
            z = torch.clamp(z, -clamp, clamp)
            # LayerNorm only fires here when shortcut is globally disabled —
            # mirrors SONN.forward's `apply_shortcut or not shortcut` guard.
            if layer.layer_norm is not None and not model.param.model.shortcut:
                z = layer.layer_norm(z)
            proj = head(z)
            # Regression: return raw scalar (N,); the configured loss_fn
            # (NormMSE) compares against the (N,) target directly. Multi-
            # class: log_softmax + NLL via the model's NLLLoss head.
            if is_regressor:
                return proj.squeeze(-1)
            return F.log_softmax(proj, dim=-1)

        verbose = model.param.train.verbose
        tbar = tqdm(
            total=cfg.max_steps, desc=f"layer {layer.layer_index} finetune",
            file=sys.stdout, ncols=140, disable=not verbose,
        )

        step = 0
        best_val_loss = float("inf")
        evals_no_improve = 0
        stop = False
        train_loss_accum = 0.0
        train_steps_accum = 0

        while step < cfg.max_steps and not stop:
            for nm in layer.neuron_models:
                nm.train()
            head.train()

            for batch in train_feat_dl:
                if step >= cfg.max_steps:
                    break

                features, targets = batch
                features = features.to(device=device)
                targets  = targets.to(device=device)

                log_probs = fwd(features)
                loss = model.loss_fn(log_probs, targets).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss_accum += loss.item()
                train_steps_accum += 1
                step += 1

                if step % cfg.eval_interval != 0:
                    continue

                for nm in layer.neuron_models:
                    nm.eval()
                head.eval()

                val_loss = 0.0
                with torch.no_grad():
                    for vbatch in dev_feat_dl:
                        vf, vt = vbatch
                        vf = vf.to(device=device); vt = vt.to(device=device)
                        val_loss += model.loss_fn(fwd(vf), vt).mean().item()
                val_loss /= len(dev_feat_dl)
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
                    logger.info(
                        f"layer {layer.layer_index} finetune early stop at step {step} — "
                        f"no improvement for {evals_no_improve} evals"
                    )
                    stop = True
                    break

                for nm in layer.neuron_models:
                    nm.train()
                head.train()

        tbar.close()

        # Restore grads on every model parameter so the next train_layer
        # iteration (and subsequent train_out_proj / train_finetune passes)
        # don't start with this pass's freeze state. The temporary head
        # goes out of scope here.
        for p in model.parameters():
            p.requires_grad_(True)

    def _train_layer_finetune_lbfgs(
            self,
            model: SONN,
            layer: SONNLayer,
            head: torch.nn.Linear,
            trainable_params: list[torch.nn.Parameter],
            train_feat_dl: DataLoader,
            dev_feat_dl: DataLoader,
            cfg,
    ) -> None:
        """Full-batch LBFGS path for `_train_layer_finetune`.

        Concatenates the precomputed (layers 0..N-1) features into a single
        tensor so each closure call sees the exact gradient — same regime as
        `_train_out_proj_lbfgs`. Unlike out_proj, the objective here is not
        convex (bilinear interaction between the polynomial `weight`s and the
        temporary head's `weight`), so LBFGS converges to a local minimum;
        for a single layer-level refinement that's usually fine.

        `head` and `trainable_params` are constructed by the caller —
        `trainable_params` contains `head.weight`, `head.bias`, and each
        surviving neuron's `weight`. weight_decay (if > 0) is added inside
        the closure as a manual L2 penalty, since LBFGS doesn't accept it.
        """
        device = model.device
        clamp = model.param.model.output_clamp_value

        line_search = cfg.lbfgs_line_search or None
        opt = torch.optim.LBFGS(
            trainable_params,
            lr=cfg.lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=cfg.lbfgs_history_size,
            line_search_fn=line_search,
        )

        # Concatenate the whole train/dev splits into single full-batch
        # tensors so every closure sees the exact convex-side of the gradient.
        def _cat_split(dl: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
            xs, ys = [], []
            for features, targets in dl:
                xs.append(features.to(device=device))
                ys.append(targets.to(device=device))
            return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

        train_X, train_y = _cat_split(train_feat_dl)
        dev_X,   dev_y   = _cat_split(dev_feat_dl)

        # Mirror the Adam/SGD path's regressor branch. `head` was already
        # built with the correct out_dim (1 for regressor, num_classes for
        # multi-class) by `_train_layer_finetune` before dispatching here.
        is_regressor = model.param.model.type == "regressor"

        def fwd(features: torch.Tensor) -> torch.Tensor:
            """Same fine-tune forward as the Adam/SGD path — kept inline so
            the closure stays self-contained."""
            z = layer(features)
            z = torch.clamp(z, -clamp, clamp)
            if layer.layer_norm is not None and not model.param.model.shortcut:
                z = layer.layer_norm(z)
            proj = head(z)
            if is_regressor:
                return proj.squeeze(-1)
            return F.log_softmax(proj, dim=-1)

        wd = float(cfg.weight_decay)

        def closure():
            opt.zero_grad()
            log_probs = fwd(train_X)
            loss = model.loss_fn(log_probs, train_y).mean()
            if wd > 0.0:
                # Manual L2 — PyTorch's LBFGS does not accept weight_decay.
                # Applied to every trainable parameter (head weight/bias plus
                # surviving neurons' polynomial weights).
                l2 = sum((p ** 2).sum() for p in trainable_params)
                loss = loss + 0.5 * wd * l2
            loss.backward()
            return loss

        verbose = model.param.train.verbose
        tbar = tqdm(
            total=cfg.max_steps,
            desc=f"layer {layer.layer_index} finetune (lbfgs)",
            file=sys.stdout, ncols=140, disable=not verbose,
        )

        best_val_loss = float("inf")
        evals_no_improve = 0

        for step in range(1, cfg.max_steps + 1):
            for nm in layer.neuron_models:
                nm.train()
            head.train()

            train_loss_t = opt.step(closure)
            train_loss_val = (
                float(train_loss_t.item()) if torch.is_tensor(train_loss_t) else float(train_loss_t)
            )

            if step % cfg.eval_interval != 0:
                tbar.update(1)
                continue

            for nm in layer.neuron_models:
                nm.eval()
            head.eval()
            with torch.no_grad():
                val_loss = model.loss_fn(fwd(dev_X), dev_y).mean().item()

            if val_loss < best_val_loss - cfg.early_stop_min_delta:
                best_val_loss = val_loss
                evals_no_improve = 0
            else:
                evals_no_improve += 1

            tbar.set_postfix({
                "train_loss": f"{train_loss_val:.4f}",
                "val_loss":   f"{val_loss:.4f}",
                "best":       f"{best_val_loss:.4f}",
                "no_imp":     evals_no_improve,
            })
            tbar.update(1)

            if evals_no_improve >= cfg.early_stop_patience:
                tbar.total = tbar.n
                tbar.refresh()
                logger.info(
                    f"layer {layer.layer_index} finetune (lbfgs) early stop at step {step} "
                    f"— no improvement for {evals_no_improve} evals"
                )
                break

        tbar.close()

    def infer(
        self,
        model: SONN,
        test_dl: DataLoader,
        verbose: bool = True,
        use_compile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference over test_dl and return (log_probs, targets).

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

        last_layer = model.layers[-1]
        best_module_idx, best_neuron_idx = model.get_best_neuron_model(last_layer)
        best_module = last_layer.neuron_models[best_module_idx]
        best_module.prune(best_neuron_idx)
        last_layer.neuron_models = nn.ModuleList([best_module])
        # Keep module_idxs / err_values in sync with the pruned neuron list so
        # a second prune() call (e.g. the iris tutorial that prunes for the
        # `_pruned_model` plot after already pruning for the regular one)
        # doesn't index the modules with stale rows.
        device_l = best_module.weight.device
        best_err_pos = last_layer.err_values.topk(1, largest=False).indices
        last_layer.err_values = last_layer.err_values[best_err_pos]
        last_layer.module_idxs = torch.tensor([[0, 0]], dtype=torch.long, device=device_l)

        # Walk layers from second-to-last down to 0 with an index, because we
        # may delete entries from model.layers mid-iteration when a layer
        # contributes nothing to its downstream neighbor.
        i = len(model.layers) - 2
        while i >= 0:
            cur_layer = model.layers[i]
            next_layer = model.layers[i + 1]
            cur_num_neuron_models = cur_layer.module_idxs.shape[0]

            # Indices in next_layer's src_idxs that point at cur_layer outputs
            # vs. at the original-feature shortcut block. The split point is
            # cur_num_neuron_models — see SONN.forward's `cat([x, x_inp])`.
            #
            # Flatten each module's src_idxs to 1-D before cat so neuron
            # families with different arities (pair-based LinearCov /
            # Quadratic / Cubic at dim=2 vs PolyQuadratic at dim=k) can
            # coexist in the same layer. We only need the set of unique
            # *index values* here, not the per-neuron tuple shape — every
            # downstream use of `uniq_src_idxs` is `< cur_num_neuron_models`,
            # which treats it as a flat index pool.
            uniq_src_idxs = torch.unique(
                torch.cat([m.src_idxs.reshape(-1) for m in next_layer.neuron_models])
            )
            used = uniq_src_idxs[uniq_src_idxs < cur_num_neuron_models]

            if used.numel() == 0:
                # cur_layer's outputs aren't referenced by next_layer (the
                # surviving best path uses original features through the
                # shortcut only). The layer is dead weight — delete it.
                #
                # next_layer's input layout was:
                #   [cur_layer_output (cur_num), x_inp (d_model)]
                # After deleting cur_layer it becomes:
                #   [prev_layer_output (prev_cur_num), x_inp (d_model)]  if a prev layer exists
                #   x_inp (d_model)                                       otherwise
                # so the shortcut block shifts by (prev_cur_num - cur_num).
                prev_cur_num = (
                    model.layers[i - 1].module_idxs.shape[0] if i > 0 else 0
                )
                shift = prev_cur_num - cur_num_neuron_models
                if shift != 0:
                    for module in next_layer.neuron_models:
                        module.src_idxs = (module.src_idxs + shift).to(
                            device=module.src_idxs.device
                        )

                # Drop cur_layer from the ModuleList and renumber the rest.
                # Each BasePolynomNeuron caches its own .layer_index (set at
                # construction, also persisted via params_metadata) and
                # plot_model + checkpoint metadata read it directly, so the
                # renumber has to propagate down into the neurons too.
                model.layers = nn.ModuleList(
                    [L for j, L in enumerate(model.layers) if j != i]
                )
                for j, L in enumerate(model.layers):
                    L.layer_index = j
                    for nm in L.neuron_models:
                        nm.layer_index = j

                logger.info(
                    f"prune: removed layer at position {i} "
                    f"({cur_num_neuron_models} unreferenced modules); "
                    f"shifted downstream src_idxs by {shift}."
                )
                i -= 1
                continue

            new_module_idxs = cur_layer.module_idxs[used]
            new_num_neuron_models = new_module_idxs.shape[0]
            new_modules = []
            # Build the post-prune module_idxs with renumbered (new_module_idx,
            # arange(K)) rows — same shape contract as train_layer's output —
            # so a second prune() call sees consistent (module_idxs ↔ modules)
            # state. `used` is sorted (torch.unique returns sorted) and
            # train_layer lays module_idxs out module-by-module, so the loop's
            # iteration order matches the new layer's concatenated output
            # order; cur_layer.err_values[used] is therefore already correctly
            # ordered for the new layer.
            renumbered_idxs_rows = []
            dev = cur_layer.module_idxs.device
            new_idx = 0
            for module_idx in torch.unique(new_module_idxs[:, 0]):
                cur_module_idx = new_module_idxs[new_module_idxs[:, 0] == module_idx]
                new_module = cur_layer.neuron_models[module_idx]
                new_module.prune(cur_module_idx[:, 1])
                new_modules.append(new_module)
                k = cur_module_idx.shape[0]
                renumbered_idxs_rows.append(torch.stack([
                    torch.full((k,), new_idx, dtype=torch.long, device=dev),
                    torch.arange(k, dtype=torch.long, device=dev),
                ], dim=1))
                new_idx += 1
            cur_layer.neuron_models = NeuronModuleList(new_modules)
            back_map = {k: v for k, v in enumerate(["-".join([str(x) for x in item.tolist()]) for item in cur_layer.module_idxs])}
            forw_map = {v: k for k, v in enumerate(["-".join([str(x) for x in item.tolist()]) for item in new_module_idxs])}

            def remap_func(x):
                try:
                    # x is an index of the prev layer output
                    res = forw_map[back_map[x]]
                except KeyError:
                    # x is an index of the model inputs
                    diff = cur_num_neuron_models - new_num_neuron_models
                    res = x - diff
                return res

            for module in next_layer.neuron_models:
                src_idx = module.src_idxs.cpu()
                src_idx.apply_(remap_func)
                module.src_idxs = src_idx.to(device=model.device)

            # Swap the per-layer metadata in AFTER back_map / forw_map
            # snapshotted the pre-prune state, so a subsequent prune() call
            # sees module_idxs whose within-indices match the now-smaller
            # modules.
            if cur_layer.err_values is not None:
                cur_layer.err_values = cur_layer.err_values[used]
            cur_layer.module_idxs = torch.cat(renumbered_idxs_rows)
            i -= 1

