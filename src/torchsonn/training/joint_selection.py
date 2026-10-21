"""Joint selection of a layer's survivors: the gated pool over the layer's
candidates, its training under the sparsity penalty, and the cut."""
import logging
import math
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchsonn.layer import SONNLayer
from torchsonn.model import SONN, _linear_from_seed
from torchsonn.modules import HardConcreteGate, diversity_loss, group_lasso_penalty, group_norms
from torchsonn.training import ensemble, evaluation, finetune

logger = logging.getLogger(__name__)


class CountControl:
    """lam control toward a target number of active neurons (`train.joint_sparsity.lam_control`).

    The joint pass calls `observe` at every evaluation with the number of
    active neurons and the dev loss. It tracks the state to roll back to,
    the one with the fewest active neurons not below `target` (on equal
    counts, the lower loss), and moves `lam` after the warm-up: times
    `lam_step` while more than `hi` = max(target, floor(target x (1 +
    margin))) neurons are active and neither the count nor the penalty has
    fallen since the previous evaluation (the penalty by at least 1%): the
    neurons lag behind lam, so while they are still moving lam waits. It is
    divided by `lam_step` while fewer than `target` are active ('adaptive'
    only), held otherwise. A sparsity penalty almost never brings a closed neuron
    back, so the count only falls: the band stops the pass before it falls
    through the target, and the copy undoes an overshoot.
    """

    def __init__(self, mode: str, target: int, margin: float, patience: int,
                 lam_step: float, lam: float) -> None:
        self.mode = mode
        self.target = int(target)
        self.hi = max(self.target, int(math.floor(self.target * (1.0 + float(margin)))))
        self.patience = int(patience)
        self.lam_step = float(lam_step)
        self.lam = float(lam)
        self.best_count: int | None = None
        self.best_loss = float("inf")
        self.best_step: int | None = None
        self.in_band = 0
        self.below = 0
        self.prev_active: int | None = None
        self.prev_penalty: float | None = None

    def observe(self, step: int, active: int, loss: float, adjust: bool,
                penalty: float | None = None) -> tuple[bool, str | None]:
        """Record an evaluation. Returns (whether this state is the new copy to
        roll back to, 'band' / 'below' when the pass should stop, else None).
        `adjust` lets lam move (False during the warm-up); `penalty` is the
        penalty's current value (without lam)."""
        better = active >= self.target and (
            self.best_count is None or active < self.best_count
            or (active == self.best_count and loss < self.best_loss))
        if better:
            self.best_count, self.best_loss, self.best_step = active, loss, step
        self.in_band = self.in_band + 1 if self.target <= active <= self.hi else 0
        self.below = self.below + 1 if active < self.target else 0
        falling = (self.prev_active is not None and active < self.prev_active) or (
            penalty is not None and self.prev_penalty is not None and penalty < 0.99 * self.prev_penalty)
        self.prev_active, self.prev_penalty = active, penalty
        if self.mode == "fixed":
            return better, None
        if self.in_band >= self.patience:
            return better, "band"
        if self.below >= self.patience:
            return better, "below"
        if adjust:
            if active > self.hi and not falling:
                self.lam *= self.lam_step
            elif active < self.target and self.mode == "adaptive":
                self.lam /= self.lam_step
        return better, None


class JointPool:
    """One layer's pool in the joint selection mode (`Trainer.train_joint`).

    The layer being built (the model's last; the earlier layers are frozen,
    and the pool reads the layer's input, computed once) and a linear head
    over all of its columns, in the layer's column order. Each column is
    standardized with fixed statistics of the training split, (output -
    `center`) * `scale`, before the head, so the head's weights compare the
    neurons on one scale and the level of the prediction sits in the head's
    bias; under 'gates' a hard-concrete gate multiplies the standardized
    column, and the gate rather than the weight scale carries a neuron's
    importance. Every candidate predicts the target on its own, so the raw
    columns share a large common level: a gate on a raw column would scale
    that level, and the gate noise would move the whole prediction. Under
    'group_lasso' neuron j's group is its coefficient row together with its
    column of the head: zeroing the group removes the neuron's deviation
    from its mean from the prediction.
    """

    def __init__(self, model: SONN, head: nn.Linear, center: torch.Tensor, scale: torch.Tensor,
                 js: Any) -> None:
        self.model = model
        self.layer = model.layers[-1]
        self.clamp = float(model.param.model.output_clamp_value)
        self.head = head
        self.center = center
        self.scale = scale
        self.kind = str(js.kind)
        self.multiclass = model.param.model.type == "multi-class"
        self.gate: HardConcreteGate | None = None
        if self.kind == "gates":
            self.gate = HardConcreteGate(head.in_features, beta=float(js.beta), gamma=float(js.gamma),
                                         zeta=float(js.zeta), init_open=float(js.init_open))
            self.gate.to(device=scale.device, dtype=scale.dtype)
        self.bounds = [0]
        for module in self.layer.neuron_models:
            self.bounds.append(self.bounds[-1] + module.num_neurons)
        # Each neuron's head column norm when the pass starts; 'group_lasso'
        # measures a neuron's importance against it.
        self.head_norms0 = self.head.weight.detach().norm(dim=0).clone() if self.gate is None else None

    @property
    def size(self) -> int:
        return int(self.head.in_features)

    def tensors(self) -> list[torch.Tensor]:
        """Everything the pass trains: the layer's parameters, the head and the gates."""
        gate = [self.gate.log_alpha] if self.gate is not None else []
        return list(self.layer.parameters()) + list(self.head.parameters()) + gate

    @torch.no_grad()
    def state(self) -> list[torch.Tensor]:
        """A copy of `tensors()`, for `load`."""
        return [t.detach().clone() for t in self.tensors()]

    @torch.no_grad()
    def load(self, state: list[torch.Tensor]) -> None:
        """Copy a `state()` back in place."""
        for t, saved in zip(self.tensors(), state):
            t.copy_(saved)

    def groups(self) -> list[list[torch.Tensor]]:
        """Per neuron module, the parts of its neurons' groups: the coefficient
        rows and the matching columns of the head (transposed views)."""
        head_t = self.head.weight.T
        return [[module.weight, head_t[a:b]] for module, a, b
                in zip(self.layer.neuron_models, self.bounds[:-1], self.bounds[1:])]

    def group_norms(self) -> torch.Tensor:
        return torch.cat([group_norms(parts) for parts in self.groups()])

    def penalty(self) -> torch.Tensor:
        """The expected number of open gates, or the sum of the group norms."""
        if self.gate is not None:
            return self.gate.expected_l0()
        return sum(group_lasso_penalty(parts) for parts in self.groups())

    def diversity(self, h: torch.Tensor, z: torch.Tensor | None, tau: float) -> torch.Tensor:
        """`diversity_loss` of the layer's outputs `h`, each pair weighted by
        the gates `z`, or without gates by the norms of the head columns."""
        g = z if z is not None else self.head.weight.norm(dim=0)
        return diversity_loss(h, g, tau=tau)

    def train(self) -> None:
        self.model.train()
        if self.gate is not None:
            self.gate.train()

    def eval(self) -> None:
        self.model.eval()
        if self.gate is not None:
            self.gate.eval()

    def eval_gates(self) -> torch.Tensor | None:
        """The deterministic gates, or None without gates."""
        return self.gate.deterministic() if self.gate is not None else None

    def outputs(self, x_in: torch.Tensor) -> torch.Tensor:
        """The layer's (clamped) outputs on its input `x_in`, as `SONN.forward` computes them."""
        return torch.clamp(self.layer(x_in), -self.clamp, self.clamp)

    def predict(self, x_in: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """The pool's prediction from the layer's input, shaped like
        `SONN.infer`'s: (B,) for a regressor, (B, C) log-probabilities for a
        multi-class model."""
        return self.readout(self.outputs(x_in), z)

    def readout(self, h: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """`predict` from the layer's outputs `h`."""
        u = (h - self.center) * self.scale
        if z is not None:
            u = u * z
        out = self.head(u)
        return F.log_softmax(out, dim=-1) if self.multiclass else out.squeeze(-1)

    @torch.no_grad()
    def importance(self) -> torch.Tensor:
        """Per neuron: its deterministic gate, or the norm of its head column
        over the norm it started the pass with.

        Under 'group_lasso' the penalty covers the whole group, so the head
        column cannot shrink while the coefficients grow to make up for it,
        but whether the head still reads a neuron is told by its column
        alone: the coefficient rows carry each neuron's constant term (about
        the target's mean for every candidate), which Adam takes thousands
        of steps to shrink and which the centred column does not pass on."""
        if self.gate is not None:
            return self.gate.deterministic()
        return self.head.weight.norm(dim=0) / self.head_norms0.clamp_min(1e-12)

    @torch.no_grad()
    def histogram(self) -> str:
        """How the neurons' importances spread, for the log."""
        imp = self.importance()
        bins = [(0.0, 0.05), (0.05, 0.5), (0.5, 0.95), (0.95, float("inf"))]
        counts = [int((imp == 0).sum())] + [int(((imp > lo) & (imp <= hi)).sum()) for lo, hi in bins]
        what = "gates" if self.gate is not None else "head col/start"
        return (f"{what} =0: {counts[0]}, (0,.05]: {counts[1]}, (.05,.5]: {counts[2]}, "
                f"(.5,.95]: {counts[3]}, >.95: {counts[4]}")


def own_errors(model: SONN, layer: SONNLayer, dl: DataLoader, *,
               batch_callback: Callable | None) -> torch.Tensor:
    """Each neuron's own error on `dl` by the regularity criterion (the
    dev error a 'validate' selection ranks candidates by), in column order."""
    device = model.device
    errs = []
    model.eval()
    with torch.no_grad():
        for module in layer.neuron_models:
            _, _, pred_fn, params, buffers, _ = ensemble.create_loss_functions(model, module)
            errs.append(evaluation.regularity_err(model, pred_fn, params, buffers, dl, device, batch_callback=batch_callback).detach())
    return torch.cat(errs).to(device=layer.module_idxs.device)


def joint_pool(model: SONN, train_in: DataLoader, js: Any) -> JointPool:
    """The pool over the model's last layer and its fitted wide head.

    Measures each column's mean and standard deviation on the training
    split (`train_in`: the layer's input, precomputed) and moves the
    head's weights into the layer's column order (the fitted head reads
    the columns in ascending-error order), rescaled to the standardized
    columns, with the means moved into the bias, so the prediction is
    unchanged.
    """
    layer = model.layers[-1]
    k = len(layer)
    device = model.device
    clamp = float(model.param.model.output_clamp_value)
    n = 0
    s1 = torch.zeros(k, dtype=torch.float64, device=device)
    s2 = torch.zeros(k, dtype=torch.float64, device=device)
    model.eval()
    with torch.inference_mode():
        for x_in, _ in train_in:
            out = torch.clamp(layer(x_in.to(device=device)), -clamp, clamp).double()
            s1 += out.sum(dim=0)
            s2 += (out * out).sum(dim=0)
            n += out.shape[0]
    mean = s1 / n
    std = (s2 / n - mean * mean).clamp_min(0.0).sqrt()
    # A constant column gets the scale of a column 1e6 times quieter than
    # the loudest, not an infinite one.
    std = std.clamp_min(1e-6 * float(std.max().clamp_min(1e-12)))
    dtype = model.out_proj.weight.dtype
    scale = (1.0 / std).to(dtype=dtype)
    center = mean.to(dtype=dtype)

    cols = torch.as_tensor(model._best_neuron_columns(layer, k), dtype=torch.long, device=device)
    head = _linear_from_seed(k, int(model.out_proj.out_features), int(model.param.train.seed)).to(device=device)
    with torch.no_grad():
        w = model.out_proj.weight
        head.weight.zero_()
        head.weight[:, cols] = w / scale[cols]
        head.bias.copy_(model.out_proj.bias + w @ center[cols])
    return JointPool(model, head, center, scale, js)


def sparsify_joint_pool(model: SONN, pool: JointPool, train_in: DataLoader,
                        stop_in: DataLoader, stop_split: str, target: int, verbose: bool) -> None:
    """Train the pool, the head and the gates together under the sparsity penalty.

    `train_in` / `stop_in` hold the layer's input, precomputed: the
    earlier layers are frozen. AdamW over the layer's parameters and the
    head (`lr`, `weight_decay`) and over the gates (`gate_lr`, no
    decay), for at most
    `max_steps` steps. lam rises linearly from 0 over the first
    `lam_warmup_frac` of the steps, then follows `lam_control` against
    `target` (`CountControl`), which can also stop the pass early or
    roll it back. The loss is data loss + lam * penalty: the expected
    number of open gates under 'gates' (one gate sample per step, from a
    generator seeded by `train.seed`), the sum of the group norms under
    'group_lasso'. Both go through AdamW, so a neuron shrinks only where
    the penalty's gradient outweighs its data gradient; a proximal step
    (a shrink of lam * lr for every group alike) does not tell the
    neurons apart under Adam's normalized steps, and on California
    housing it shrank all 120 at once.
    """
    js = model.param.train.joint_sparsity
    device = model.device
    for p in pool.layer.parameters():
        p.requires_grad_(True)
    param_groups = [{"params": list(pool.layer.parameters()) + list(pool.head.parameters()),
                     "lr": float(js.lr), "weight_decay": float(js.weight_decay)}]
    if pool.gate is not None:
        param_groups.append({"params": [pool.gate.log_alpha], "lr": float(js.gate_lr), "weight_decay": 0.0})
    opt = torch.optim.AdamW(param_groups)
    generator = torch.Generator(device=device).manual_seed(int(model.param.train.seed))
    batches, _ = finetune.finetune_batches(train_in, device, "auto", None)

    max_steps = int(js.max_steps)
    eval_interval = int(js.eval_interval)
    threshold = float(js.prune_threshold)
    lam_max = float(js.lam)
    div_lam = float(js.diversity_lam)
    div_tau = float(js.diversity_tau)
    warmup = float(js.lam_warmup_frac) * max_steps
    control = CountControl(str(js.lam_control), target, float(js.count_margin),
                            int(js.count_patience), float(js.lam_step), lam_max)
    # The state to roll back to; the pool as it starts has every neuron.
    best_state = pool.state()
    control.best_count, control.best_step = pool.size, 0

    def lam_at(step: int) -> float:
        return control.lam if step >= warmup else lam_max * step / warmup

    steer = "" if control.mode == "fixed" else (
        f", steered to {control.target}-{control.hi} active ({control.mode}, x{control.lam_step:g} per evaluation)")
    diverse = f", diversity {div_lam:g} (tau {div_tau:g})" if div_lam > 0 else ""
    logger.info(f"joint pass, layer #{pool.layer.layer_index}: {js.kind}, lam {lam_max:g} "
                f"(warm-up {int(warmup)} steps){steer}{diverse}, "
                f"{pool.size} neurons, at most {max_steps} steps")
    tbar = tqdm(total=max_steps, desc="joint", file=sys.stdout, ncols=140,
                disable=not (verbose and model.param.train.verbose))
    step = 0
    active = pool.size
    data_sum, data_n, div_sum = 0.0, 0, 0.0
    while step < max_steps:
        x, y = next(batches)
        if x is None:          # epoch boundary
            continue
        lam = lam_at(step)
        pool.train()
        z = pool.gate(generator) if pool.gate is not None else None
        h = pool.outputs(x)
        data = model.loss_fn(pool.readout(h, z), y).mean()
        loss = data + lam * pool.penalty()
        if div_lam > 0:
            div = pool.diversity(h, z, div_tau)
            loss = loss + div_lam * div
            div_sum += float(div.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        data_sum += float(data.item())
        data_n += 1
        if step % eval_interval != 0 and step != max_steps:
            continue

        pool.eval()
        val_loss = evaluation.mean_loss(model, stop_in, lambda v: pool.predict(v, pool.eval_gates()),
                                       batch_callback=None)
        with torch.no_grad():
            penalty = float(pool.penalty())
        active = int((pool.importance() > threshold).sum())
        diverse = f" | diversity {div_sum / data_n:.4f}" if div_lam > 0 else ""
        logger.info(f"joint step {step}: lam {lam:.3g} | train data loss {data_sum / data_n:.4f} | "
                    f"lam*penalty {lam * penalty:.4g}{diverse} | {stop_split} loss {val_loss:.4f} | "
                    f"active {active}/{pool.size} | {pool.histogram()}")
        tbar.set_postfix({"data": f"{data_sum / data_n:.4f}", stop_split: f"{val_loss:.4f}",
                          "lam": f"{lam:.2e}", "active": active})
        tbar.n = step
        tbar.refresh()
        data_sum, data_n, div_sum = 0.0, 0, 0.0
        better, stop = control.observe(step, active, val_loss, adjust=step >= warmup, penalty=penalty)
        if better:
            best_state = pool.state()
        if stop == "band":
            logger.info(f"joint pass: {active} active, in [{control.target}, {control.hi}] for "
                        f"{control.patience} evaluations; stopping at step {step}")
            break
        if stop == "below":
            logger.info(f"joint pass: fewer than {control.target} active for {control.patience} "
                        f"evaluations; rolling back to step {control.best_step} "
                        f"({control.best_count} active)")
            pool.load(best_state)
            break
    else:
        if control.mode != "fixed" and active < control.target:
            logger.info(f"joint pass: {active} active at the last step, fewer than {control.target}; "
                        f"rolling back to step {control.best_step} ({control.best_count} active)")
            pool.load(best_state)
    tbar.close()


@torch.no_grad()
def joint_contribution(pool: JointPool, dl: DataLoader) -> torch.Tensor:
    """How much each pool neuron moves the prediction: the norm of its head
    column times the standard deviation, over the loader (the layer's
    input, precomputed), of its scaled and (deterministically) gated output."""
    device = pool.model.device
    z = pool.eval_gates()
    n = 0
    s1 = torch.zeros(pool.size, dtype=torch.float64, device=device)
    s2 = torch.zeros(pool.size, dtype=torch.float64, device=device)
    for x_in, _ in dl:
        u = (pool.outputs(x_in.to(device=device)) - pool.center) * pool.scale
        if z is not None:
            u = u * z
        u = u.double()
        s1 += u.sum(dim=0)
        s2 += (u * u).sum(dim=0)
        n += u.shape[0]
    mean = s1 / n
    std = (s2 / n - mean * mean).clamp_min(0.0).sqrt()
    return pool.head.weight.detach().double().norm(dim=0) * std


@torch.no_grad()
def joint_survivors(pool: JointPool, target: int, dl: DataLoader) -> torch.Tensor:
    """The pool columns to keep, sorted: the active neurons, cut to the
    `target` with the largest contribution on `dl` when there are more
    (`joint_contribution`), and the most important one when none is
    active."""
    threshold = float(pool.model.param.train.joint_sparsity.prune_threshold)
    imp = pool.importance()
    keep = (imp > threshold).nonzero().squeeze(1)
    if keep.numel() == 0:
        logger.warning(f"joint selection: no neuron is active (threshold {threshold}); "
                       "keeping the most important one")
        keep = imp.argmax().view(1)
    if keep.numel() > target:
        contrib = joint_contribution(pool, dl)[keep]
        order = torch.argsort(contrib, descending=True, stable=True)
        logger.info(f"joint selection: {keep.numel()} active neurons, more than the layer keeps; "
                    f"keeping the {target} with the largest contribution (head weight x output std "
                    f"on dev: kept >= {float(contrib[order[target - 1]]):.4g}, "
                    f"cut <= {float(contrib[order[target]]):.4g})")
        keep = keep[order[:target]]
    return keep.sort().values
