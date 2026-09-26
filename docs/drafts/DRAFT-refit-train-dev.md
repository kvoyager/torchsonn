# DRAFT: Refit coefficients on train + dev after structure selection

Status: draft for review, no code written. Patches are gated: each is
implemented only on explicit approval, one at a time, and marked DONE here
with the date and the measured numbers.

## 1. Problem

The trainer uses the dev rows for one thing only: choosing. They score
candidate neurons (the regularity criterion), pick the survivors, decide when
to stop adding layers, and early-stop the head fit and the end-to-end pass.
They never fit a coefficient. On the California-housing 60/20/20 split that is
12384 rows fitting and 4128 rows watching, so every coefficient in the network
is estimated from 75% of the data that was available for estimation.

Classic GMDH handles this with a final step that keeps the selected structure
and re-solves the coefficients on the whole sample (Ivakhnenko's "refit on all
data"; gmdhpy does not do it, torchsonn does not either). The structure is what
needed a held-out set; the coefficients, once the structure is fixed, do not.

Expected gain: a third more rows for every least-squares fit. With the current
best config (`california_housing_legendre_finetune.yaml`) the single-run
noise floor is ±0.002 MSE and the 5-member ensemble resolves ~0.002, so an
effect of 0.005–0.01 would be clearly measurable. Lower bound if it does
nothing: zero, since the structure and the head architecture are untouched.

## 2. Where each stage gets its data today

| stage | fits on | watches | notes |
|---|---|---|---|
| squash calibration (`fit_layer_squash`) | train | – | per layer, mean/std of that layer's inputs |
| NormMSE scale (`_fit_target_scale`) | train | – | fixed denominator, once per run |
| candidate fits (`train_model_ensemble`) | train | dev (val loss early stop) | vmapped LBFGS, `train.steps` cap |
| neuron selection, layer growth | – | dev | `err_values`, `criterion_minimum_width`, epsilon rule |
| head fit (`train_out_proj`, LBFGS path) | train | dev (early stop) | saves `model_last.ckpt` |
| end-to-end pass (`train_finetune`) | train | dev (early stop) | AdamW, plateau LR; keeps last weights, no best-state restore; not checkpointed |
| prediction | – | – | `SONN.infer`, head or best column |

Facts the design relies on:

- `_precompute_dl(model, dl, device, skip_last_layer=True)` runs the frozen
  prefix over any loader and caches the features layer `k` consumes. Refitting
  layer `k` needs exactly this, on the union loader, after layers `< k` have
  been refit.
- `layer.err_values` and `layer.module_idxs` define the survivor ordering that
  `_best_neuron_columns` uses to feed `out_proj`. They are search-time
  quantities and must stay frozen through a refit, otherwise the head's input
  order changes under it.
- The Legendre / linear_cov / quadratic / polyquad neurons are linear in their
  coefficients and the loss is squared error plus ridge, so a per-neuron refit
  is a convex problem: LBFGS run to convergence lands on the unique optimum,
  and no early stopping is needed. A pair neuron has 8 coefficients against
  16512 rows, so overfitting is not a concern even at `ridge_alpha: 0`.
- The linear head is convex for the same reason.
- The end-to-end pass is not convex and today needs dev to know when to stop.

## 3. Proposal

An opt-in stage `refit`, run after the structural search and before
prediction, on the union of the train and dev loaders:

**Ordering rule (review comment, 2026-09-26):** once any coefficient has been
fit on the dev rows, the dev loss is no longer a held-out signal, so no
dev-watched decision may follow a union fit. The refit stage therefore runs
strictly *after* the last dev-monitored step, whatever that step is (head fit,
or end-to-end pass), and nothing after the refit may early-stop on dev. This
is not test leakage: the test rows are never touched and the reported test
metrics stay honest. It is a decision-quality problem: a dev loss computed on
rows the weights were fit to is optimistic, so a stopping rule based on it
runs too long or stops in the wrong place. The first version of this draft
violated the rule in Patch 4 (refit, then a dev-monitored end-to-end pass);
Patches 3 and 4 below are rewritten to respect it.

**Fixed (structure):** layer count, neuron families, `src_idxs` (which inputs
each neuron reads), survivor sets and their `err_values` / `module_idxs`
ordering, squash calibration, NormMSE scale, head shape.

**Re-solved (coefficients):** every surviving neuron's `weight` (and its
`proj_weight`/`proj_bias` where the family has them), layer by layer from
layer 0 up, each on features produced by the already-refit prefix; then the
head. The end-to-end pass is a separate, later decision (Patch 3).

Refit cost is small: per layer it re-solves 16 survivors, not 140–280
candidates, so roughly a tenth of the search time per layer.

## 4. Patches

| # | patch | status | date | measured |
|---|---|---|---|---|
| 1 | `Trainer.refit_layers` + `train.refit` config | proposed | | |
| 2 | head refit on the union (`train_out_proj` without dev) | proposed | | |
| 3 | end-to-end pass on the union with a recorded budget | proposed, gated on 1–2 | | |
| 4 | tutorial wiring + ensemble measurement | proposed | | |

### Patch 1: `Trainer.refit_layers(model, union_dl)`

- New config: `train.refit: bool = False`, `train.refit_steps: int` (LBFGS
  step cap per layer; convergence normally ends it earlier via the existing
  train-loss tolerance window `train_loss_tol` / `train_loss_window`).
- For `k` in layers, in order: build `feat_dl = _precompute_dl(model,
  union_dl, device, skip_last_layer=True)` with `model.layers[:k+1]` as the
  prefix (same trick `train_layer` uses: the last layer is the one being
  fitted), then optimize each neuron module of layer `k` on `feat_dl` with the
  vmapped training loss (`create_loss_functions`, ridge included), starting
  from the search-time weights. No dev loader, no eval, no selection; the
  ensemble here is the survivor set itself.
- Untouched: `err_values`, `module_idxs`, `neuron_idxs`, squash parameters,
  `loss_fn.scale`, `layer.err`, `model.layer_err`.
- Ends with `save_model_checkpoint(model)` so the refit weights are what
  `load_model_checkpoint` returns from then on.
- Tests: (a) refit with `union_dl == train_dl` reproduces the search-time
  training loss within tolerance and does not change `err_values` or the
  layer output width; (b) refit on a union with extra rows lowers the
  union training loss and leaves `prune()` / `infer()` shapes intact; (c) a
  multi-class model refits its `proj_weight` / `proj_bias` too, or raises a
  clear NotImplemented for the `shared_proj` / `soft_binner` paths if those
  are left out of the first version (see Decision D6).

### Patch 2: head refit on the union

- `train_out_proj(model, train_dl, dev_dl=None)`: with no dev loader the
  LBFGS path runs to convergence (`max_steps` cap, train-loss tolerance), the
  Adam/SGD path runs `max_steps` with no plateau schedule. Then saves the
  checkpoint as today.
- Called by the refit stage right after Patch 1, on the union loader, so the
  head is solved over the refit survivors' outputs.
- Test: with `dev_dl=None` the head converges to the same solution a
  closed-form ridge over the precomputed features gives, within tolerance.

### Patch 3 (gated): the end-to-end pass under the ordering rule

The end-to-end pass is dev-monitored, so with `refit` on it can only come
*before* the refit, or run on the union with no dev signal at all. Three
legal pipelines, to be decided from the Patch 4 measurements:

- **3a, pass first, then refit.** Search → head fit (dev) → end-to-end pass
  on train (dev early stop, best-dev weights restored) → refit layers and
  head on the union (Patches 1–2), warm-started from the fine-tuned weights.
  Every dev decision is made before the first union fit. Cost: the convex
  refit partly undoes what the non-convex pass did to the neuron weights,
  because it re-solves each neuron for its own least-squares optimum rather
  than for the joint readout. Whether the net effect is positive is exactly
  what Patch 4 measures.
- **3b, refit, then budget replay on the union.** Search → head fit (dev) →
  end-to-end pass on train (dev; record `best_step`) → refit layers and head
  on the union → rerun the end-to-end pass on the union for `best_step` steps
  from the refit weights, same optimizer settings, no scheduler, no early
  stop. Legal because the union pass takes no dev decision; the budget was
  decided before any union fit. Ignores that the plateau schedule shaped the
  original run, and costs a second pass.
- **3c, no end-to-end pass.** Search → refit layers and head on the union.
  Cheapest and cleanest; loses whatever the pass is worth on top of the
  refit model.

Illegal, and what the first version of this draft proposed: refit on the
union, then a dev-monitored end-to-end pass.

Regardless of the option, `train_finetune` should restore the best-dev
weights rather than keep the last ones (today it does not); that is a
separate one-line fix worth doing first, since 3a and 3b both start the
refit from "the fine-tuned weights" and today those are the last-step ones.

### Patch 4: tutorial wiring and measurement

- `california_housing.py`: with `train.refit`, the union loader (train + dev
  datasets concatenated, batch size as configured) is built once per member,
  and the stages run in the order of the chosen Patch 3 option. In every
  option the refit call comes after the last dev-monitored step, and no
  dev-monitored step follows it. Applies per ensemble member.
- Measurement: the 5-member ensemble of
  `california_housing_legendre_finetune.yaml`, same seeds, same test rows,
  in four variants: baseline (no refit), 3c (refit, no end-to-end pass),
  3a (pass, then refit) and 3b (refit, then budget replay). Report
  single-member mean ± std and ensemble MSE / MAE for each. Baseline today:
  members 0.2010 ± 0.0021, ensemble 0.1901 / 0.2740. Four ensembles at
  ~35 minutes each on the GPU.
- Record the numbers in this table and in the config header's results table.

## 5. Decisions

- **D1 Squash calibration: keep the search-time one.** It is part of the
  structure each neuron's coefficients were fitted around; recalibrating on
  the union would shift every input by a few thousandths of a sigma and force
  the refit to compensate for no benefit. Train and dev come from the same
  shuffle, so their moments match anyway.
- **D2 NormMSE scale: keep from train.** It only scales the loss; the
  least-squares optimum is unchanged, and the ridge term's effective strength
  stays what the search used.
- **D3 Survivor ordering (`err_values`) frozen.** Recomputing errors on the
  union would reorder the head's inputs; the head is refit anyway, but the
  pruned model's readout and the plots depend on the ordering, and there is
  nothing to gain from changing it.
- **D4 Warm start from the search-time weights**, not from a fresh init:
  convex problem, same optimum, far fewer steps.
- **D5 Refit is opt-in** (`train.refit: False` default) so every existing
  config and test is unaffected.
- **D6 First version covers the regressor and binary paths** (weights, plus
  `proj_weight` / `proj_bias` for neurons that carry them). The multi-class
  `shared_proj` / `soft_binner` paths raise NotImplemented until measured on
  a classification tutorial; they have their own per-layer state
  (`shared_proj_states`) that a refit would need to reproduce.
- **D7 Checkpointing:** the refit weights replace `model_last.ckpt`. The
  end-to-end weights stay in memory only, as today, unless Patch 3 says
  otherwise.
- **D8 `resume` is unaffected:** refit runs after the search completes and
  reads no layer checkpoints.
- **D9 Ordering rule enforced in code, not only by convention.** The refit
  stage sets a flag on the trainer (e.g. `self._dev_consumed = True`); any
  later call that early-stops on a dev loader (`train_out_proj` with a dev
  loader, `train_finetune` with a dev loader) raises with a message naming
  the rule. A pipeline that wants the pass after the refit must use the
  budget-replay form (3b), which takes no dev loader.

## 6. Risks and open questions

- If the end-to-end pass is what carries most of the gain today, option 3a
  may give part of it back (the convex refit re-solves neurons for their own
  optimum, not the joint readout) and option 3c forgoes it entirely. Only 3b
  keeps both, at the cost of a second pass and a replayed budget. Patch 4
  measures all three against the baseline before any option is made the
  default.
- Under the ordering rule the union stage is blind: after the refit no
  held-out rows remain to catch a pathological fit, and the dev-loss numbers
  the tutorial logs during the search cannot be compared with anything
  computed after the refit. The test set stays the only honest yardstick.
- The union has no held-out rows, so nothing in the refit stage can detect
  a pathological fit. Convexity and the 8-coefficient neurons make that a
  theoretical concern for the polynomial families; it is real for the
  end-to-end pass, which is why Patch 3 is gated.
- Timing: ~2 s per layer for 16 Legendre survivors on the GPU, so ~30 s for
  a 15-layer model plus the head; negligible next to the 7-minute member.
- Wider question, out of scope here: the same argument says the *search*
  could use k-fold selection instead of a single dev split. That is a
  different, larger draft.
