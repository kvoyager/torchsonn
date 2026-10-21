"""torch.distributed helpers for ensemble parallel candidate training."""
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def init_distributed(backend: str = "nccl") -> None:
    """Initialize torch.distributed for multi-GPU / multi-node training.

    Call once before Trainer.train(), then launch with torchrun:
        torchrun --nproc_per_node=<N> --nnodes=<M> script.py

    Sets LOCAL_RANK as the active CUDA device so each rank uses a
    distinct GPU within the node.
    """
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def ensemble_slice(ensemble_size: int) -> tuple[int, int] | None:
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


def gather_ensemble_params(params: dict, shared_param_names: list) -> dict:
    """All-gather non-shared params from all ranks, concatenating along dim 0.

    Restores the full (ensemble_size, ...) shape after each rank trained
    its local slice.  Shared params (e.g. shared_proj_weight) are the
    same on all ranks after all-reduce and are passed through unchanged.
    Requires equal local_size on every rank, which ensemble_slice
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


def allreduce_shared_grads(grads: dict, shared_param_names: list) -> None:
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
