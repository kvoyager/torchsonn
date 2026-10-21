"""End-to-end and per-layer fine-tuning: the training step (eager or a
replayed CUDA graph), its batch source and the per-layer pass."""
import logging
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchsonn.layer import SONNLayer
from torchsonn.model import SONN
from torchsonn.training import evaluation, keep_best

logger = logging.getLogger(__name__)


class FinetuneStep:
    """One end-to-end training step, eager or as a replayed CUDA graph.

    The graph is captured once over static buffers of the loader's batch
    shape (`x_s`, `y_s`, a static loss); every later step copies the next
    batch into the buffers and replays. A batch of a different shape (the
    tail of an epoch) runs the same forward / backward / optimizer step
    eagerly, on the same parameters, gradient tensors and optimizer state,
    so no row is dropped. The loss is accumulated on the device and read
    back only when `drain` is called (no per-step host sync).

    Capture recipe: three eager warm-up steps on a side stream (allocator
    and cuBLAS workspaces), then `torch.cuda.graph` on that stream with the
    optimizer's gradients kept allocated (`zero_grad(set_to_none=False)`)
    and a capturable Adam / AdamW whose lr is a device tensor. The warm-up
    steps are real steps on the first batch and are counted.
    """

    def __init__(self, model: SONN, opt: torch.optim.Optimizer, loss_fn: nn.Module,
                 pred_fn: Callable, use_graph: bool) -> None:
        self.model, self.opt, self.loss_fn, self.pred_fn = model, opt, loss_fn, pred_fn
        self.use_graph = bool(use_graph)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.x_s: torch.Tensor | None = None
        self.y_s: torch.Tensor | None = None
        self.loss_s: torch.Tensor | None = None
        self.stream = torch.cuda.Stream() if self.use_graph else None
        device = model.device
        self.loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.n_steps = 0
        self.graph_steps = 0
        self.eager_steps = 0
        self.capture_error: str | None = None

    def _eager(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.opt.zero_grad(set_to_none=False)
        loss = self.loss_fn(self.pred_fn(self.model, x), y).mean()
        loss.backward()
        self.opt.step()
        return loss.detach()

    def _rehearse(self, x: torch.Tensor, y: torch.Tensor) -> str | None:
        """Run one eager step with PyTorch's synchronization debug mode set to
        raise, as a stand-in for the capture's own restriction: a step that
        synchronizes with the host (`.item()`, `.cpu()`, `nonzero`, ...)
        cannot be captured, and a capture that fails leaves the CUDA context
        unusable for the rest of the process (no API call recovers it), so
        the check has to happen before. Returns the reason the step is not
        capturable, or None. The debug mode does not flag every
        synchronizing operation, so the capture itself stays guarded too.
        The rehearsal is a real, counted step. One plain forward first warms
        the model's lazy caches (the head's column index is computed from
        host-side tensors on its first use after a checkpoint load), which
        would otherwise be flagged as a sync of the step itself."""
        with torch.no_grad():
            self.pred_fn(self.model, x)
        torch.cuda.set_sync_debug_mode("error")
        try:
            self.loss_sum += self._eager(x, y).to(torch.float64)
            self.n_steps += 1
            self.eager_steps += 1
        except RuntimeError as e:
            return f"{type(e).__name__}: {e}"
        finally:
            torch.cuda.set_sync_debug_mode("default")
        return None

    def _capture(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x_s = x.detach().clone()
        self.y_s = y.detach().clone()
        s = self.stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.loss_sum += self._eager(self.x_s, self.y_s).to(torch.float64)
                self.n_steps += 1
                self.eager_steps += 1
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        self.opt.zero_grad(set_to_none=False)
        with torch.cuda.graph(graph, stream=s):
            # The zeroing must be part of the graph: a replay accumulates into
            # the same static gradient tensors as the previous one.
            self.opt.zero_grad(set_to_none=False)
            loss = self.loss_fn(self.pred_fn(self.model, self.x_s), self.y_s).mean()
            loss.backward()
            self.opt.step()
        self.loss_s = loss.detach()
        self.graph = graph

    def step(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Train on one batch (already on the device)."""
        if self.use_graph and (self.x_s is None or x.shape == self.x_s.shape):
            if self.graph is None:
                reason = self._rehearse(x, y)
                if reason is not None:
                    # The flagged rehearsal stopped mid-step: redo this batch
                    # eagerly from clean gradients and stay eager from here on.
                    self.capture_error = reason
                    self.use_graph = False
                    self.graph = None
                    self.opt.zero_grad(set_to_none=False)
                    self.loss_sum += self._eager(x, y).to(torch.float64)
                    self.n_steps += 1
                    self.eager_steps += 1
                    return
                try:
                    self._capture(x, y)
                except Exception as e:  # noqa: BLE001
                    # Not recoverable: after a failed capture every CUDA call
                    # in this process fails. Say so instead of cascading.
                    raise RuntimeError(
                        "end-to-end pass: CUDA graph capture failed after a clean rehearsal "
                        f"({type(e).__name__}: {e}); the CUDA context is now unusable for this "
                        "process. Re-run with finetune_train.cuda_graph=false.") from e
            else:
                self.x_s.copy_(x, non_blocking=True)
                self.y_s.copy_(y, non_blocking=True)
            self.graph.replay()
            self.loss_sum += self.loss_s.to(torch.float64)
            self.graph_steps += 1
        else:
            self.loss_sum += self._eager(x, y).to(torch.float64)
            self.eager_steps += 1
        self.n_steps += 1

    def drain(self) -> float:
        """Mean training loss over the steps since the last call."""
        if self.n_steps == 0:
            return float("nan")
        mean = (self.loss_sum / self.n_steps).item()
        self.loss_sum.zero_()
        self.n_steps = 0
        return mean


def finetune_batches(train_dl: DataLoader, device: torch.device, mode: str, batch_callback: Callable | None):
    """Yield (x, y) batches on the device for the end-to-end pass, once per
    epoch, from one of two sources (see OutProjTrainConfig.data_on_device):
    the whole split resident on the device, index-selected in a fresh
    permutation per epoch (or in order when the loader does not shuffle),
    or streamed from the loader with a non-blocking copy. Returns the
    generator and a description for the log."""
    on_cuda = torch.device(device).type == "cuda"
    resident = False
    desc = "streamed from the loader"
    if on_cuda and mode in ("auto", "true"):
        xs, ys = [], []
        rows = 0
        for batch in train_dl:
            x, y = batch_callback(batch) if batch_callback else batch
            xs.append(x)
            ys.append(y)
            rows += x.shape[0]
        nbytes = sum(t.numel() * t.element_size() for t in xs) + sum(t.numel() * t.element_size() for t in ys)
        free, _ = torch.cuda.mem_get_info(device)
        if mode == "true" or nbytes * 2 < free // 4:
            X = torch.cat(xs, 0).to(device=device)
            Y = torch.cat(ys, 0).to(device=device)
            resident = True
            desc = f"{rows} rows resident on the device ({nbytes / 1e6:.1f} MB)"
        del xs, ys
    shuffle = isinstance(getattr(train_dl, "sampler", None), torch.utils.data.RandomSampler)
    batch_size = int(train_dl.batch_size)

    def epochs():
        while True:
            if resident:
                n = X.shape[0]
                order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
                for start in range(0, n, batch_size):
                    idx = order[start:start + batch_size]
                    yield X.index_select(0, idx), Y.index_select(0, idx)
            else:
                for batch in train_dl:
                    x, y = batch_callback(batch) if batch_callback else batch
                    if on_cuda and not x.is_pinned():
                        x, y = x.pin_memory(), y.pin_memory()
                    yield x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)
            yield None, None   # epoch boundary

    return epochs(), desc


def finetune_prediction(model: SONN, x: torch.Tensor) -> torch.Tensor:
    """Model output shaped to match the target, for the end-to-end pass.

    `SONN.infer` collapses every path to one prediction per sample: the
    head's output where there is one, otherwise the best-error neuron's
    column of the last layer (the value a headless regressor / binary
    model is scored on). The scalar losses therefore compare (N,) against
    (N,) directly, and the column indexing inside `infer` keeps the
    gradient path to that neuron and, through it, to every layer feeding
    it.
    """
    return model.infer(x)


def train_layer_finetune(
        model: SONN,
        layer: SONNLayer,
        train_dl: DataLoader,
        dev_dl: DataLoader,
        freeze_neurons: bool = False,
        *,
        batch_callback: Callable | None,
        run_dir: Callable[[], Path],
) -> float:
    """Per-layer joint fine-tune that runs after neuron_selection.

    Returns the layer's readout error (see `train.layer_err_source`): the
    dev loss of the weights the survivors end with, not finite when no
    evaluation gave a finite loss. They end on the last step (a last step
    between two evaluations is evaluated too), or with
    `out_proj_train.keep_best_weights` on the lowest dev loss evaluated,
    copied back from where `best_weights_copy` keeps them. With frozen
    survivors it is the lowest dev loss the temporary head reached.
    With `freeze_neurons=True` only the temporary head is trained and the
    survivors' weights are left exactly as fitted: a pure measurement of
    the layer's readout, used by `layer_err_source: readout` when
    `layer_finetune` is off.

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
    on the cached features. Every parameter the neuron owns is unfrozen
    (`weight`, and the RBF family's centers and widths) except
    proj_weight / proj_bias, which are intentionally left alone since the
    loss doesn't flow through them in this pass.
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
    neuron_params: list[torch.nn.Parameter] = []
    if not freeze_neurons:
        # Every parameter the neuron owns, not `weight` alone: the RBF
        # family also carries learnable centers and widths. Unchanged for
        # the polynomial families, whose only parameter is `weight`.
        for nm in layer.neuron_models:
            for name, p in nm.named_parameters():
                if name in ("proj_weight", "proj_bias"):
                    continue
                p.requires_grad_(True)
                neuron_params.append(p)
    trainable_params: list[torch.nn.Parameter] = list(head.parameters()) + neuron_params
    # Under keep_best_weights only the survivors' weights are put back at
    # the end: the temporary head is discarded, and with frozen neurons
    # there is nothing to copy.
    lbfgs_tag = " (lbfgs)" if cfg.optimizer == "lbfgs" else ""
    best_weights = keep_best.record(neuron_params, cfg,
                                    f"layer_{layer.layer_index}_finetune",
                                    f"layer {layer.layer_index} finetune{lbfgs_tag}",
                                    run_dir=run_dir)

    # Features through layers 0..N-1. Current `layer` is model.layers[-1],
    # so skip_last_layer=True is "everything but this one".
    train_feat_dl = evaluation.precompute_dl(model, train_dl, device, skip_last_layer=True, batch_callback=batch_callback)
    dev_feat_dl   = evaluation.precompute_dl(model, dev_dl,   device, skip_last_layer=True, batch_callback=batch_callback)

    # LBFGS runs full-batch with a closure — see the dedicated helper.
    # It still calls the same setup above (precomputed features +
    # trainable_params + head), so we hand them off rather than duplicate
    # the freeze/precompute boilerplate.
    if cfg.optimizer == "lbfgs":
        readout_err = _train_layer_finetune_lbfgs(
            model, layer, head, trainable_params,
            train_feat_dl, dev_feat_dl, cfg, best_weights,
        )
        for p in model.parameters():
            p.requires_grad_(True)
        return readout_err

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
        # LayerNorm only fires here when nothing is concatenated to a
        # layer's output (`shortcut: false`) — mirrors SONN.forward's
        # last-layer guard.
        if layer.layer_norm is not None and not model.concat_inputs:
            z = layer.layer_norm(z)
        proj = head(z)
        # Regression: return raw scalar (N,); the configured loss_fn
        # (NormMSE) compares against the (N,) target directly. Multi-
        # class: log_softmax + NLL via the model's NLLLoss head.
        if is_regressor:
            return proj.squeeze(-1)
        return F.log_softmax(proj, dim=-1)

    def _dev_loss() -> float:
        """Mean dev loss of the layer and the temporary head."""
        for nm in layer.neuron_models:
            nm.eval()
        head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vf, vt in dev_feat_dl:
                vf = vf.to(device=device); vt = vt.to(device=device)
                val_loss += model.loss_fn(fwd(vf), vt).mean().item()
        return val_loss / len(dev_feat_dl)

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

    # Evaluate a last step that fell between two evaluations, so the
    # readout error is that of the weights the layer keeps; under
    # keep_best_weights the survivors go back to their best evaluation.
    if step > (best_weights.last_step or 0):
        best_weights.update(step, _dev_loss())
    best_weights.restore()

    # Restore grads on every model parameter so the next train_layer
    # iteration (and subsequent train_out_proj / train_finetune passes)
    # don't start with this pass's freeze state. The temporary head
    # goes out of scope here.
    for p in model.parameters():
        p.requires_grad_(True)
    return best_weights.final_loss


def _train_layer_finetune_lbfgs(
        model: SONN,
        layer: SONNLayer,
        head: torch.nn.Linear,
        trainable_params: list[torch.nn.Parameter],
        train_feat_dl: DataLoader,
        dev_feat_dl: DataLoader,
        cfg,
        best_weights: keep_best.BestWeights,
) -> float:
    """Full-batch LBFGS path for `train_layer_finetune`.

    Concatenates the precomputed (layers 0..N-1) features into a single
    tensor so each closure call sees the exact gradient — same regime as
    `head_fit.train_out_proj_lbfgs`. Unlike out_proj, the objective here is not
    convex (bilinear interaction between the polynomial `weight`s and the
    temporary head's `weight`), so LBFGS converges to a local minimum;
    for a single layer-level refinement that's usually fine.

    `head` and `trainable_params` are constructed by the caller —
    `trainable_params` contains `head.weight`, `head.bias`, and each
    surviving neuron's `weight`. weight_decay (if > 0) is added inside
    the closure as a manual L2 penalty, since LBFGS doesn't accept it.
    `best_weights` records the evaluations (and, under keep_best_weights,
    copies the survivors' best one); the return value is its
    `final_loss`, as in the Adam / SGD path.
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
    # multi-class) by `train_layer_finetune` before dispatching here.
    is_regressor = model.param.model.type == "regressor"

    def fwd(features: torch.Tensor) -> torch.Tensor:
        """Same fine-tune forward as the Adam/SGD path — kept inline so
        the closure stays self-contained."""
        z = layer(features)
        z = torch.clamp(z, -clamp, clamp)
        if layer.layer_norm is not None and not model.concat_inputs:
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

    def _dev_loss() -> float:
        for nm in layer.neuron_models:
            nm.eval()
        head.eval()
        with torch.no_grad():
            return model.loss_fn(fwd(dev_X), dev_y).mean().item()

    step = 0
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

        val_loss = _dev_loss()
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
            logger.info(
                f"layer {layer.layer_index} finetune (lbfgs) early stop at step {step} "
                f"— no improvement for {evals_no_improve} evals"
            )
            break

    tbar.close()

    # Evaluate a last step that fell between two evaluations, so the
    # readout error is that of the weights the layer keeps; under
    # keep_best_weights the survivors go back to their best evaluation.
    if step > (best_weights.last_step or 0):
        best_weights.update(step, _dev_loss())
    best_weights.restore()
    return best_weights.final_loss
