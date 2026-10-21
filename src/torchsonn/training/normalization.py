"""Calibration from the training split: each layer's input statistics and
the regressor's loss scale."""
import logging
from typing import Callable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from torchsonn.layer import SONNLayer
from torchsonn.loss import NormMSE
from torchsonn.model import SONN
from torchsonn.training import distributed, evaluation

logger = logging.getLogger(__name__)


def fit_layer_inputs(model: SONN, layer: SONNLayer, train_dl: DataLoader, *,
                     batch_callback: Callable | None) -> None:
    """Per-layer input pass: calibrate every neuron model on the layer's
    actual inputs before the layer trains.

    The inputs have to be the features that reach *this* layer: the
    (preprocessed) raw inputs for layer 0, the frozen prefix's outputs —
    plus the shortcut-concatenated originals — deeper down. So this runs
    once per layer, after create_layer and before any of the layer's
    neurons train, and measures `model(x, skip_last_layer=True)` exactly
    as train_layer will feed it. Three things come out of it, each only
    when some neuron model in the layer asks (see the hooks on
    `BasePolynomNeuron`):

      * mean / std per input feature (`needs_input_stats`): the sigma
        squash of the orthogonal families. Streamed in float64.
      * a seeded reservoir sample of up to `train.input_sample_rows` rows
        (`needs_input_sample`), the whole split when it is smaller, taken
        in the same pass. Handed to `fit_input_sample(sample, stream)`.
      * a streaming pass (`needs_input_stream`, and only when the mode
        resolves to 'stream': `train.rbf_kmeans_mode` 'stream', or 'auto'
        with more rows than the cap): `train.rbf_kmeans_passes` further
        passes over the split feeding `stream_input_batch(x_batch)`, then
        `finish_input_stream()` once.

    Costs one forward pass over the training split per layer, plus one
    per streaming pass. Skipped outright when no neuron in the layer asks
    for anything, so configs that use none of it pay nothing.
    """
    modules = list(layer.neuron_models)
    want_stats = any(nm.needs_input_stats for nm in modules)
    want_sample = any(nm.needs_input_sample for nm in modules)
    if not (want_stats or want_sample):
        return

    cfg = model.param.train
    max_rows = int(cfg.input_sample_rows)
    if want_sample and max_rows <= 0:
        raise ValueError(f"train.input_sample_rows must be > 0, got {max_rows}")
    mode = str(cfg.rbf_kmeans_mode).lower()
    if mode not in ("auto", "sample", "stream"):
        raise ValueError(
            f"train.rbf_kmeans_mode must be 'auto', 'sample' or 'stream', got {cfg.rbf_kmeans_mode!r}")

    device = modules[0].device
    # Accumulate in float64 whatever the inputs' type is: a running sum of
    # squares over a large split bleeds precision badly in float32, and
    # this runs once per layer so the extra width is free.
    total: torch.Tensor | None = None
    total_sq: torch.Tensor | None = None
    count = 0
    reservoir: torch.Tensor | None = None
    filled = 0
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + 7919 * (layer.layer_index + 1))

    was_training = model.training
    model.eval()
    # no_grad rather than inference_mode: these statistics get copied into
    # buffers that autograd reads during training, and inference tensors
    # are barred from autograd-enabled ops.
    with torch.no_grad():
        for batch in train_dl:
            x_inp, _ = batch_callback(batch) if batch_callback else batch
            feats = model(x_inp.to(device=device), skip_last_layer=True)
            f64 = feats.double()
            batch_sum = f64.sum(dim=0)
            batch_sq = (f64 * f64).sum(dim=0)
            total = batch_sum if total is None else total + batch_sum
            total_sq = batch_sq if total_sq is None else total_sq + batch_sq
            if want_sample:
                reservoir, filled = evaluation.reservoir_add(reservoir, filled, count, feats, max_rows, gen)
            count += feats.shape[0]
    model.train(was_training)

    if count == 0:
        logger.warning(
            "Layer #%d input pass skipped: training set is empty. "
            "Neurons keep their identity stats (mean 0, std 1).",
            layer.layer_index,
        )
        return

    mean = total / count
    # Var as E[x^2] - E[x]^2. Clamped because the cancellation can land a
    # hair below zero for a near-constant feature; SigmaSquashNorm then
    # reads the resulting std == 0 as a constant feature and neutralizes it.
    std = (total_sq / count - mean * mean).clamp(min=0.0).sqrt()

    if want_stats:
        # Back from the float64 accumulators to the parameters' type.
        dtype = torch.get_default_dtype()
        layer.fit_input_stats(mean.to(dtype=dtype), std.to(dtype=dtype))
        logger.info(
            "Layer #%d squash calibrated on %d training samples over %d input "
            "feature(s); |mean| max %.4g, std range [%.4g, %.4g]",
            layer.layer_index, count, mean.numel(),
            mean.abs().max().item(), std.min().item(), std.max().item(),
        )

    if not want_sample:
        return
    sample = reservoir[:filled]
    stream = mode == "stream" or (mode == "auto" and count > max_rows)
    seed = int(cfg.seed) + 7919 * (layer.layer_index + 1)
    for nm in modules:
        if nm.needs_input_sample:
            nm.fit_input_sample(sample, stream=stream, seed=seed, iters=int(cfg.rbf_kmeans_iters))
    logger.info(
        "Layer #%d input sample: %d of %d training rows kept (%s); k-means mode %s",
        layer.layer_index, filled, count,
        "whole split" if filled == count else "seeded reservoir",
        "stream" if stream else "sample",
    )
    if not stream:
        return
    streamers = [nm for nm in modules if nm.needs_input_stream]
    if not streamers:
        return
    passes = int(cfg.rbf_kmeans_passes)
    if passes < 1:
        raise ValueError(f"train.rbf_kmeans_passes must be >= 1, got {passes}")
    model.eval()
    with torch.no_grad():
        for _ in range(passes):
            for batch in train_dl:
                x_inp, _ = batch_callback(batch) if batch_callback else batch
                feats = model(x_inp.to(device=device), skip_last_layer=True)
                for nm in streamers:
                    nm.stream_input_batch(feats)
    model.train(was_training)
    for nm in streamers:
        nm.finish_input_stream()
    logger.info("Layer #%d streaming pass done: %d pass(es) over %d rows for %d neuron model(s)",
                layer.layer_index, passes, count, len(streamers))


def fit_target_scale(model: SONN, train_dl: DataLoader, device: torch.device | str, *,
                     batch_callback: Callable | None) -> None:
    """Fix the NormMSE denominator to the training-set target spread.

    One streaming pass over `train_dl`'s targets — no model forward, the
    labels are all this needs. Sets `model.loss_fn.scale` to the per-sample
    variance of the training targets (or their mean square under
    `error_normalization: energy`), so the regressor's training loss is
    `Σ(y-ŷ)² / (n_batch · scale)`.

    Fixing the scale rather than recomputing it per batch is what keeps the
    loss batch-size-independent, safe on a degenerate batch, and on the same
    scale as the dev-set regularity criterion — see NormMSE's docstring.

    `device` is where the sums are accumulated (`train.device`). No-op
    unless the model is a regressor (only that path uses NormMSE).
    """
    if not isinstance(model.loss_fn, NormMSE):
        return

    # float64: `sum_y2 - sum_y**2 / n` cancels, and on a target whose mean
    # dominates its spread the two terms can differ by several decades —
    # more than float32 can absorb.
    n = 0
    sum_y = torch.zeros((), dtype=torch.float64, device=device)
    sum_y2 = torch.zeros((), dtype=torch.float64, device=device)
    for batch in train_dl:
        _, targets = batch_callback(batch) if batch_callback else batch
        t = targets.to(device=device, dtype=torch.float64).reshape(-1)
        n += t.numel()
        sum_y += t.sum()
        sum_y2 += (t * t).sum()

    acc = torch.stack([torch.as_tensor(float(n), dtype=torch.float64, device=device),
                       sum_y, sum_y2])
    if distributed.is_dist():
        # Every rank must use the same denominator, or the per-neuron losses
        # they compare are on different scales. Correct whether or not the
        # loader is sharded: `scale` is invariant to a common factor on all
        # three accumulators, so summing an unsharded loader across W ranks
        # gives the same answer as not reducing at all.
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)

    n, sum_y, sum_y2 = acc.tolist()
    if n <= 0:
        logger.warning("empty train dataloader; leaving the NormMSE denominator per-batch")
        return

    scale = (sum_y2 - sum_y * sum_y / n) / n if model.error_centered else sum_y2 / n
    if not (scale > model.loss_fn.eps):
        logger.warning(
            "training targets have no spread (measured scale %.3g); leaving the NormMSE "
            "denominator per-batch. A constant target makes the normalized error undefined.",
            scale,
        )
        return

    model.loss_fn.scale = scale
    logger.info(
        "NormMSE denominator fixed to the training-set target %s: %.6g",
        "variance" if model.error_centered else "mean square", scale,
    )
