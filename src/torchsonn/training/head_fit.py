"""The output head fit with LBFGS, and the head's input columns."""
import logging
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchsonn.model import SONN
from torchsonn.training import evaluation, keep_best

logger = logging.getLogger(__name__)


def train_out_proj_lbfgs(model: SONN, train_dl: DataLoader, dev_dl: DataLoader, cfg, *,
                        batch_callback: Callable | None, run_dir: Callable[[], Path]) -> None:
    """Full-batch LBFGS path for `Trainer.train_out_proj`.

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
    train_feat_dl = evaluation.precompute_dl(model, train_dl, device, skip_last_layer=False, batch_callback=batch_callback)
    dev_feat_dl   = evaluation.precompute_dl(model, dev_dl,   device, skip_last_layer=False, batch_callback=batch_callback)

    # Concatenate the whole dataset into one full-batch tensor per split.
    def _cat_split(dl):
        xs, ys = [], []
        for features, targets in dl:
            xs.append(features.to(device=device))
            ys.append(targets.to(device=device))
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    train_X, train_y = _cat_split(train_feat_dl)
    dev_X,   dev_y   = _cat_split(dev_feat_dl)
    train_X_sel = select_out_proj_inputs(train_X, cols_t, k)
    dev_X_sel   = select_out_proj_inputs(dev_X,   cols_t, k)
    del train_X, dev_X

    best_weights = keep_best.record(list(model.out_proj.parameters()), cfg,
                                    "out_proj", "out_proj (lbfgs)",
                                    run_dir=run_dir)

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
        loss.backward()
        loss = loss.detach()
        if wd > 0.0:
            # Manual L2 — PyTorch's LBFGS does not accept weight_decay.
            # Backward'd separately from the data term.
            l2 = torch.stack([(p ** 2).sum() for p in model.out_proj.parameters()]).sum()
            penalty = 0.5 * wd * l2
            penalty.backward()
            loss = loss + penalty.detach()
        return loss

    step = 0
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
        best_weights.update(step, val_loss)

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

    # Evaluate a last step that fell between two evaluations; under
    # keep_best_weights the head then goes back to its best evaluation.
    if step > (best_weights.last_step or 0):
        model.out_proj.eval()
        with torch.no_grad():
            best_weights.update(step, _head_loss(dev_X_sel, dev_y).item())
    best_weights.restore()


def select_out_proj_inputs(
    features: torch.Tensor, cols_t: torch.Tensor, k: int
) -> torch.Tensor:
    """Select and optionally zero-pad columns for out_proj input."""
    selected = features[:, cols_t]
    if selected.shape[1] < k:
        pad = torch.zeros(selected.shape[0], k - selected.shape[1],
                          device=features.device, dtype=features.dtype)
        selected = torch.cat([selected, pad], dim=-1)
    return selected
