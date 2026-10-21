"""Evaluation over a loader: features of the frozen model, candidate losses
and the criteria the growth rule reads."""
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchsonn.data.dataset import SONNDataset
from torchsonn.layer import SONNLayer
from torchsonn.loss import bias_error, bias_error_js, bias_error_l2
from torchsonn.model import SONN


def stop_source(config: Any, val_dl: DataLoader | None) -> str:
    """`train.stop_source` of `config`: the split, `dev` or `val`, that the
    growth rule and the end-to-end pass stop on."""
    source = str(getattr(config.train, "stop_source", "dev")).lower()
    if source not in ("dev", "val"):
        raise ValueError(f"train.stop_source must be 'dev' or 'val', got {config.train.stop_source!r}")
    if source == "val" and val_dl is None:
        raise ValueError("train.stop_source='val' needs a validation loader (val_dl)")
    return source


def layer_val_err(model: SONN, layer: SONNLayer, dl: DataLoader, *,
                  batch_callback: Callable | None) -> float:
    """Best surviving neuron's error of the (last) layer on a split, under the
    model's loss: the counterpart of `layer.err` (`layer_err_criterion:
    top`, `layer_err_source: neuron`) on rows that selected nothing.
    Regression / binary only; a multi-class model gets NaN (its neurons
    are scored through a projection this helper does not rebuild)."""
    if model.param.model.type == "multi-class":
        return float("nan")
    device = layer.neuron_models[0].device
    was_training = model.training
    model.eval()
    total: torch.Tensor | None = None
    rows = 0
    with torch.no_grad():
        for batch in dl:
            x_inp, targets = batch_callback(batch) if batch_callback else batch
            out = model(x_inp.to(device=device))                       # (B, d_layer)
            targets = targets.to(device=device)
            losses = torch.stack([model.loss_fn(out[:, j], targets).mean() for j in range(out.shape[1])])
            n = out.shape[0]
            total = losses * n if total is None else total + losses * n
            rows += n
    model.train(was_training)
    if total is None or rows == 0:
        return float("nan")
    return float((total / rows).min().item())


def reservoir_add(
    reservoir: torch.Tensor | None, filled: int, seen: int, feats: torch.Tensor,
    max_rows: int, gen: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Feed one batch into a reservoir sample of `max_rows` rows.

    Algorithm R, vectorized per batch: the first `max_rows` rows fill the
    reservoir; a later row at 0-based position `p` over the whole stream
    replaces a uniformly random slot with probability `max_rows / (p + 1)`.
    Two replacements landing on one slot inside a batch resolve to the
    later row, a negligible bias for a k-means start. `seen` is the
    number of rows streamed before this batch; the generator is on the
    CPU so the sample is reproducible for a fixed loader order.
    """
    b = feats.shape[0]
    if reservoir is None:
        reservoir = torch.empty((max_rows, feats.shape[1]), dtype=feats.dtype, device=feats.device)
    take = min(b, max_rows - filled)
    if take > 0:
        reservoir[filled:filled + take] = feats[:take]
        filled += take
    rest = feats[take:]
    if rest.shape[0] > 0:
        pos = torch.arange(seen + take, seen + b, dtype=torch.float64)
        keep = torch.rand(pos.shape[0], generator=gen, dtype=torch.float64) < (max_rows / (pos + 1.0))
        n_keep = int(keep.sum().item())
        if n_keep > 0:
            slots = torch.randint(0, max_rows, (n_keep,), generator=gen)
            reservoir[slots.to(device=feats.device)] = rest[keep.to(device=feats.device)]
    return reservoir, filled


def split_loader(dl: DataLoader, split: int) -> DataLoader:
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


def precompute_dl(
    model: SONN,
    dl: DataLoader,
    device: torch.device | str,
    skip_last_layer: bool = True,
    *,
    batch_callback: Callable | None,
) -> DataLoader:
    """Run the frozen model over the full DataLoader once and cache features.

    skip_last_layer=True  → model(x, skip_last_layer=True): used by train_layer
                             so the candidate ensemble sees frozen-prefix outputs.
    skip_last_layer=False → model(x): used by train_out_proj / train_finetune
                             so the output head trains on full model features.

    Returns a new DataLoader yielding (features, targets).  Pass with
    skip_model_fwd=True so iter_eval_batches skips the redundant forward.
    """
    all_x, all_targets = [], []
    model.eval()
    with torch.inference_mode():
        for batch in dl:
            x_inp, targets = batch_callback(batch) if batch_callback else batch
            x = model(x_inp.to(device=device), skip_last_layer=skip_last_layer)
            all_x.append(x.cpu())
            all_targets.append(targets)
    new_ds = SONNDataset(torch.cat(all_x, dim=0), torch.cat(all_targets, dim=0))
    shuffle = isinstance(dl.sampler, torch.utils.data.RandomSampler)
    return DataLoader(new_ds, batch_size=dl.batch_size, shuffle=shuffle,
                      num_workers=dl.num_workers, drop_last=dl.drop_last)


def iter_eval_batches(model: SONN, dl: DataLoader, device: torch.device | str,
                      skip_model_fwd: bool = False, *, batch_callback: Callable | None):
    """Shared eval-loop boilerplate for ds_loss / regularity_err / bias_err.

    Yields (x, targets) per batch where:
      * both tensors are on `device`
      * `x` is the SONN forward output with `skip_last_layer=True`
      * if skip_model_fwd=True the dl already contains pre-transformed
        features (from precompute_dl) — the model forward is skipped.
    """
    for batch in tqdm(dl, total=len(dl), disable=True):
        x_inp, targets = batch_callback(batch) if batch_callback else batch
        x_inp = x_inp.to(device=device)
        targets = targets.to(device=device)
        if skip_model_fwd:
            yield x_inp, targets
        else:
            with torch.inference_mode():
                x = model(x_inp, skip_last_layer=True)
                yield x, targets


def ds_loss(
        model: SONN,
        eval_loss_fn_vmapped: Callable,
        params_batch: dict[str, torch.Tensor],
        buffers_batch: dict[str, torch.Tensor],
        dl: DataLoader,
        device: torch.device | str,
        skip_model_fwd: bool = False,
        *,
        batch_callback: Callable | None) -> torch.Tensor:
    """
    Mean evaluation loss of every candidate over a loader.

    Parameters
    ----------
    model : SONN
        The model being grown.
    eval_loss_fn_vmapped : callable
        Per-candidate loss, from `create_loss_functions`.
    params_batch, buffers_batch : dict of str to tensor
        Stacked candidate parameters and buffers.
    dl : DataLoader
        The split to evaluate on.
    device : torch.device or str
        Device the batches are moved to.
    skip_model_fwd : bool
        The loader already yields the layer's input features, so the
        model forward is skipped. Default False.

    Returns
    -------
    loss : (num_candidates,) tensor
        Per-candidate loss, averaged over the batches.
    """
    err_nom, err_denom = [], []
    for x, targets in iter_eval_batches(model, dl, device, skip_model_fwd, batch_callback=batch_callback):
        val_losses = eval_loss_fn_vmapped(params_batch, buffers_batch, x, targets)
        err_nom.append(val_losses)
        err_denom.append(torch.ones_like(val_losses))
    # Element-wise divide: per-neuron mean over batches.
    return torch.stack(err_nom, dim=1).sum(dim=-1) / torch.stack(err_denom, dim=1).sum(dim=-1)


def regularity_err(
        model: SONN,
        pred_fn_vmapped: Callable,
        params_batch: dict[str, torch.Tensor],
        buffers_batch: dict[str, torch.Tensor],
        dl: DataLoader,
        device: torch.device | str,
        skip_model_fwd: bool = False,
        *,
        batch_callback: Callable | None) -> torch.Tensor:
    """Per-candidate regularity criterion, computed streaming per batch.

    Regression / binary: normalized MSE (matches regularity_error).
    Multi-class:         normalized cross-entropy (matches regularity_error_ce).

    The sums are accumulated batch by batch instead of concatenating the
    predictions into one (ensemble, N, K) tensor first: for Otto-sized
    ensembles (~17k candidates after shortcut) that tensor would take
    over 6 GB of GPU memory, twice that inside log_softmax. GPU memory
    stays at one batch's predictions.
    """
    eps = 1e-12

    if model.param.model.type == "multi-class":
        # streaming NCE: accumulate sum of -log p[i, y_i] across batches,
        # then divide by total N and by H(Y_B) from concatenated targets.
        sum_nll: torch.Tensor | None = None
        all_targets: list[torch.Tensor] = []
        total_N = 0
        K: int | None = None
        for x, targets in iter_eval_batches(model, dl, device, skip_model_fwd, batch_callback=batch_callback):
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
        # streamed normalized MSE — mirrors loss/errors.py::regularity_error,
        # which cannot be called directly here because the eval set is only
        # ever seen one batch at a time.
        num: torch.Tensor | None = None
        # float64 accumulators: under `variance` the denominator is
        # `Σy² - (Σy)²/N`, and that subtraction cancels — on a target whose
        # mean dominates its spread the two terms can differ by several
        # decades, more than float32 can absorb.
        n_total = 0
        sum_y = torch.zeros((), device=device, dtype=torch.float64)
        sum_y2 = torch.zeros((), device=device, dtype=torch.float64)
        for x, targets in iter_eval_batches(model, dl, device, skip_model_fwd, batch_callback=batch_callback):
            preds = pred_fn_vmapped(params_batch, buffers_batch, x, targets)
            diff_sq = ((preds - targets) ** 2).sum(dim=-1)
            del preds
            num = diff_sq if num is None else num + diff_sq
            t = targets.to(torch.float64)
            n_total += t.numel()
            sum_y = sum_y + t.sum()
            sum_y2 = sum_y2 + (t * t).sum()
        assert num is not None, "empty dataloader passed to regularity_err"
        if model.error_centered:
            denom_sum = sum_y2 - sum_y * sum_y / max(n_total, 1)
        else:
            denom_sum = sum_y2
        return num / denom_sum.clamp_min(eps).to(num.dtype)


def bias_err(
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
        *,
        batch_callback: Callable | None,
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
        for x, targets in iter_eval_batches(model, dl, device, skip_model_fwd, batch_callback=batch_callback):
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
        return bias_error_l2(preds_a, preds_b, logits=True)
    else:  # regression / binary
        return bias_error(preds_a, preds_b, torch.cat([t1, t2], dim=0),
                          centered=model.error_centered)


def mean_loss(model: SONN, dl: DataLoader, predict: Callable[[torch.Tensor], torch.Tensor], *,
              batch_callback: Callable | None) -> float:
    """Mean over the loader's batches of the model's loss of `predict(x)`.
    `batch_callback=None` for a loader of precomputed features."""
    device = model.device
    total = 0.0
    with torch.no_grad():
        for batch in dl:
            x, y = batch_callback(batch) if batch_callback else batch
            total += model.loss_fn(predict(x.to(device=device)), y.to(device=device)).mean().item()
    return total / len(dl)
