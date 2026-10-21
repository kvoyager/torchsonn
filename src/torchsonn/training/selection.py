"""Survivor selection for a grown layer.

`select_candidates` holds the selection algorithms as plain functions over
tensors. `neuron_selection` gathers their inputs from a layer's accumulator,
calls it, and prunes the layer down to the survivors.
"""
import logging
from typing import Callable, Iterable

import torch
from torch import nn

from torchsonn.config.schemas import TrainConfig
from torchsonn.layer import NeuronModuleList, SONNLayer
from torchsonn.model import SONN
from torchsonn.neurons import BasePolynomNeuron

logger = logging.getLogger(__name__)

EPS = 1e-8


def _plain(err: torch.Tensor, z_val: torch.Tensor, nbest: int, target: torch.Tensor | None,
           orth_threshold: float, min_gain: float, reserve_frac: float) -> torch.Tensor:
    # Historical default: top-k by individual error.
    _, err_idxs = torch.topk(err, k=nbest, dim=0, largest=False)
    return err_idxs


def _greedy_orth(err: torch.Tensor, z_val: torch.Tensor, nbest: int, target: torch.Tensor | None,
                 orth_threshold: float, min_gain: float, reserve_frac: float) -> torch.Tensor:
    # Walk candidates in error order (best-error first) and keep any
    # whose residual after Gram-Schmidt against already-selected
    # survivors still carries enough novel signal on the dev set.
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
    return torch.tensor(selected_local_idxs, device=err.device, dtype=torch.long)


def _max_volume(err: torch.Tensor, z_val: torch.Tensor, nbest: int, target: torch.Tensor | None,
                orth_threshold: float, min_gain: float, reserve_frac: float) -> torch.Tensor:
    # Greedy maximum volume selection, vectorized over the candidate
    # axis. At every step, pick the candidate whose orthogonalized
    # residual has the largest norm, then project that direction out
    # of all remaining residuals. Pre-center each candidate's dev
    # outputs once: we score variance, not magnitude.
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
    return torch.tensor(selected_local_idxs, device=err.device, dtype=torch.long)


def _omp(err: torch.Tensor, z_val: torch.Tensor, nbest: int, target: torch.Tensor | None,
         orth_threshold: float, min_gain: float, reserve_frac: float) -> torch.Tensor:
    # Orthogonal Matching Pursuit against the dev targets: at every
    # step pick the candidate whose orthogonalized output explains the
    # most of the target residual, then project its direction out of
    # the residual and of every remaining candidate. After the first
    # survivor, stops early once the best gain falls below
    # min_gain of the target's centered energy.
    R = z_val - z_val.mean(dim=1, keepdim=True)  # (C, N_val)
    y = target.to(dtype=R.dtype)  # (N_val, O)
    r = y - y.mean(dim=0, keepdim=True)  # target residual
    r0 = (r * r).sum()
    C = R.size(0)
    available = torch.ones(C, dtype=torch.bool, device=R.device)
    selected_local_idxs = []
    neg_inf = torch.full((), float("-inf"), device=R.device, dtype=R.dtype)
    for _ in range(nbest):
        norms = R.norm(dim=1).clamp_min(EPS)
        gain = ((R @ r) ** 2).sum(dim=1) / norms ** 2  # residual target energy explained
        gain = torch.where(available & (norms > EPS), gain, neg_inf)
        best_ci = int(torch.argmax(gain).item())
        if not torch.isfinite(gain[best_ci]):
            break
        if selected_local_idxs and gain[best_ci].item() < min_gain * r0.item():  # the rest is noise
            break
        selected_local_idxs.append(best_ci)
        available[best_ci] = False
        b = R[best_ci] / norms[best_ci]  # unit direction
        r = r - b[:, None] * (b @ r)[None, :]  # deflate the target
        R = R - (R @ b).unsqueeze(1) * b  # deflate the candidates
    return torch.tensor(selected_local_idxs, device=err.device, dtype=torch.long)


def _omp_mixed(err: torch.Tensor, z_val: torch.Tensor, nbest: int, target: torch.Tensor | None,
               orth_threshold: float, min_gain: float, reserve_frac: float) -> torch.Tensor:
    # Hybrid selection in two phases over one shared Gram-Schmidt state.
    #   Phase 1, target aware OMP: fill the first (1 - reserve_frac) of
    #   the nbest slots with the candidates that explain the most dev
    #   target residual after the selected set is projected out.
    #   Phase 2, max volume reserve: fill the remaining slots with the
    #   candidates carrying the most variance the selected set cannot
    #   explain. This phase ignores the target on purpose: it keeps
    #   directions alive that may only become informative at a later
    #   layer, which pure OMP would discard.
    # Both phases skip near collinear picks with orth_threshold.
    n_omp = nbest - int(round(reserve_frac * nbest))

    R = z_val - z_val.mean(dim=1, keepdim=True)  # (C, N_val) residualized candidates
    full_norm = R.norm(dim=1).clamp_min(EPS)  # (C,) reference for the collinearity guard
    y = target.to(dtype=R.dtype)  # (N_val, O)
    r = y - y.mean(dim=0, keepdim=True)  # (N_val, O) target residual
    r0 = (r * r).sum().item()
    C = R.size(0)
    available = torch.ones(C, dtype=torch.bool, device=R.device)
    neg_inf = torch.full((), float("-inf"), device=R.device, dtype=R.dtype)
    selected_local_idxs = []

    # Phase 1: target aware OMP.
    while len(selected_local_idxs) < n_omp:
        norms = R.norm(dim=1).clamp_min(EPS)
        ok = available & (norms / full_norm >= orth_threshold)
        if not bool(ok.any()):
            break
        gain = ((R @ r) ** 2).sum(dim=1) / norms ** 2  # residual target energy explained
        gain = torch.where(ok, gain, neg_inf)
        ci = int(torch.argmax(gain).item())
        if gain[ci].item() < min_gain * r0:  # the rest does not explain y
            break
        selected_local_idxs.append(ci)
        available[ci] = False
        b = R[ci] / norms[ci]
        r = r - b[:, None] * (b @ r)[None, :]  # deflate the target
        R = R - (R @ b).unsqueeze(1) * b  # deflate the candidates

    # Phase 2: max volume reserve on the OMP deflated candidates.
    # Runs to nbest even if phase 1 stopped early on min_gain: once y
    # is explained, the remaining slots are best spent on novelty.
    while len(selected_local_idxs) < nbest:
        norms = R.norm(dim=1)
        ok = available & (norms / full_norm >= orth_threshold)
        if not bool(ok.any()):
            break
        score = torch.where(ok, norms, neg_inf)
        ci = int(torch.argmax(score).item())
        selected_local_idxs.append(ci)
        available[ci] = False
        b = R[ci] / norms[ci].clamp_min(EPS)
        R = R - (R @ b).unsqueeze(1) * b  # r is no longer needed
    return torch.tensor(selected_local_idxs, device=err.device, dtype=torch.long)


SELECTION_METHODS: dict[str, Callable[..., torch.Tensor]] = {
    "plain": _plain,
    "greedy_orth": _greedy_orth,
    "max_volume": _max_volume,
    "omp": _omp,
    "omp_mixed": _omp_mixed,
}

# Methods that score candidates against the dev targets.
TARGET_AWARE_METHODS = frozenset({"omp", "omp_mixed"})


def select_candidates(
        method: str,
        err: torch.Tensor,
        z_val: torch.Tensor,
        nbest: int,
        target: torch.Tensor | None = None,
        orth_threshold: float = TrainConfig.neuron_selection_orth_threshold,
        min_gain: float = TrainConfig.neuron_selection_min_gain,
        reserve_frac: float = TrainConfig.neuron_selection_reserve_frac,
) -> torch.Tensor:
    """Pick up to `nbest` survivors from a candidate pool.

    Parameters
    ----------
    method
        One of the keys of `SELECTION_METHODS`; see
        `train.neuron_selection_method` for what each one does.
    err
        (C,) candidate errors, without NaNs.
    z_val
        (C, N_val) candidate outputs on the dev split.
    nbest
        Upper bound on the survivors, at most C. The decorrelating methods
        may return fewer.
    target
        (N_val, O) dev targets, required by the methods in
        `TARGET_AWARE_METHODS` and ignored by the others.
    orth_threshold, min_gain, reserve_frac
        `train.neuron_selection_orth_threshold`, `_min_gain` and
        `_reserve_frac`; the defaults are the config defaults.

    Returns
    -------
    torch.Tensor
        Long tensor of candidate indices into `err`, in pick order.
    """
    select = SELECTION_METHODS.get(method)
    if select is None:
        raise NotImplementedError(
            f"Unknown neuron_selection_method: {method!r}; "
            f"expected one of {', '.join(repr(m) for m in SELECTION_METHODS)}."
        )
    if method in TARGET_AWARE_METHODS and target is None:
        raise ValueError(f"neuron_selection_method {method!r} needs the dev targets.")
    return select(err, z_val, nbest, target,
                  float(orth_threshold), float(min_gain), float(reserve_frac))


def selection_target(model: SONN, y: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dev targets as an (N_val, O) matrix for the target aware selection
    methods: one column per output, a multi-class target one-hot encoded."""
    if model.param.model.type == "multi-class":
        y = nn.functional.one_hot(y.long().view(-1), int(model.param.model.num_classes))
    y = y.to(dtype=dtype)
    return y if y.dim() == 2 else y.view(-1, 1)


def compute_z_val(
        neuron_model: BasePolynomNeuron,
        batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-candidate raw scalar outputs on the full dev split, shape (C, N_val),
    and the dev targets in the same order.

    `batches` yields (layer input, target) pairs over the dev split. Used by
    the decorrelating selection methods to skip candidates whose scalar
    response on the validation set is nearly collinear with already selected
    survivors. Reads `neuron_model.weight` directly, so it must be called
    AFTER the trained ensemble weights are written back. Multi-class
    projection is intentionally bypassed: the raw polynomial scalar is what
    feeds the next layer and is the right space to decorrelate in.
    """
    chunks: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    neuron_model.eval()
    with torch.inference_mode():
        for x, y in batches:
            z = neuron_model(x)  # (batch, C)
            chunks.append(z.transpose(0, 1))  # (C, batch)
            targets.append(y)
    return torch.cat(chunks, dim=1), torch.cat(targets, dim=0)  # (C, N_val), (N_val, ...)


def neuron_selection(
        model: SONN,
        layer: SONNLayer,
        accumulator: "LayerAccumulator",
        trained_params_per_model: dict[int, dict],
        device: torch.device | str,
        nbest: int | None = None,
        method: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the nbest survivors across the layer's neuron-model ensemble and
    prune the rest in place.

    `nbest` and `method` default to `model.nbest_neurons` and
    `train.neuron_selection_method`.

    Concatenates per-neuron-model errors, module indices and dev outputs from
    the accumulator, drops NaNs, picks the survivors with
    `select_candidates`, re-sorts them by module index, prunes each
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
    z_val = torch.cat(accumulator.z_val, dim=0)  # (C_total, N_val)

    # filter out nan values
    nan_idx = ~torch.isnan(err)
    err = err[nan_idx]
    module_idxs = module_idxs[nan_idx]
    z_val = z_val[nan_idx]

    nbest = min(err.shape[0], model.nbest_neurons if nbest is None else int(nbest))
    train_cfg = model.param.train
    method = train_cfg.neuron_selection_method if method is None else method
    target = selection_target(model, accumulator.y_val, z_val.dtype) if method in TARGET_AWARE_METHODS else None

    err_idxs = select_candidates(
        method, err, z_val, nbest, target,
        orth_threshold=train_cfg.neuron_selection_orth_threshold,
        min_gain=train_cfg.neuron_selection_min_gain,
        reserve_frac=train_cfg.neuron_selection_reserve_frac,
    )
    err_values = err[err_idxs]

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
    for nm in new_modules:
        report = nm.fit_report()
        if report:
            logger.info("Layer #%d %s", layer.layer_index, report)
    # Keep d_model in sync with the actual post-prune output count so the
    # next create_layer call uses the right number of inputs.
    layer.d_model = len(layer)
    # Build the per-layer LayerNorm with the width of the next layer's
    # input (post-shortcut-cat) so SONN.forward can fold the raw input
    # features and older layers' outputs into the normalization alongside
    # this layer's own. `layer` is the last of model.layers here, so the
    # next layer would be created right after it. No-op when
    # use_layer_norm is off.
    layer.setup_layer_norm(model.next_input_width())

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

    return err_values, module_idxs
