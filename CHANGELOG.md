# Changelog

## 0.1.4

### Added — `rbf`: Gaussian RBF neuron family with learnable, k-means-initialized centers

A local basis next to the global polynomial families. `RBFNeuron`
(`ref_functions: - rbf`, alias `gauss`) puts `centers` Gaussian bumps over
its `dim`-tuple of inputs, standardized per slot, and fits the bump centers
and widths *together* with the output weights by the existing vmapped
candidate fit (they are parameters with the `num_neurons` axis; nothing in
the fit loop changed). The design row is the normalized bumps (a partition
of unity computed as a softmax, so the constant is in the span and dropped)
plus the standardized inputs as linear columns: `num_w = M + dim`, 18 for a
pair neuron at the default `M = 16`. Widths are carried in log form and
bounded to `(1/width_band, width_band)` times their start by a smooth tanh,
so a bump can neither collapse onto one row nor blur into the linear part.
Centers are carried as a displacement from their k-means start whose length
is squashed below `center_radius` times the local center spacing, the same
smooth bound, so a cell can move across its neighbourhood but never off the
data (unbounded centers were ejected by the per-tensor quasi-Newton step on
California housing: 37-68% of survivors' centers ended outside the data).
Options: `centers`, `placement` (`kmeans` | `grid`), `width`, `learn_centers`,
`learn_widths`, `center_radius`, `width_band`, `normalize`, `linear`,
`standardize`, `dim`.
`learn_*: false` turns the tensors into buffers (the fixed-basis RBF).

Initialization happens in the per-layer input pass: batched k-means++ over
the candidates on a seeded reservoir sample of the layer input, then Lloyd
iterations (`train.rbf_kmeans_iters`); above `train.input_sample_rows` (or
with `train.rbf_kmeans_mode: stream`) the centers are refined instead by
mini-batch k-means over the whole split, `train.rbf_kmeans_passes` passes,
never holding more than one batch. `placement: grid` uses the product of
per-slot quantile grids. Widths start at the local center spacing, floored
for tied slots. After selection the log reports how far each family's
survivors' centers moved and where their width scales sit. No CA run yet.

### Changed — per-layer input pass; `BaseTupleNeuron`

`Trainer.fit_layer_squash` is now `fit_layer_inputs` (old name kept): the
same streaming mean / std pass, which additionally keeps a seeded reservoir
sample of the layer input for families that ask (`needs_input_sample`) and
runs the optional streaming pass (`needs_input_stream`). New no-op hooks on
`BasePolynomNeuron`: `needs_input_stats` / `fit_input_stats` (old names
`needs_squash_stats` / `fit_squash` kept as aliases), `needs_input_sample`
/ `fit_input_sample`, `needs_input_stream` / `stream_input_batch` /
`finish_input_stream`, `fit_report`. The `dim`-tuple candidate enumeration
of the orthogonal families moved to `BaseTupleNeuron`, which they and the
RBF family inherit; class names, state keys and metadata are unchanged, so
existing checkpoints load. `train.layer_finetune` now unfreezes every
parameter a neuron owns (except the projection), not `weight` alone.

### Fixed — ensemble LBFGS: curvature condition and step cap

`BatchedLBFGS` stored every correction pair and took a fixed step with no
bound, so on a direction whose gradient and curvature vanish together (an
RBF bump losing its mass, where `y.s <= 0`) its quasi-Newton step pointed
away from the minimum and grew without limit; unbounded RBF centers were
thrown tens of standard units off the data. Two guards, both on by default:
a pair is kept only if `y.s > curvature_eps |s||y|` (`curvature_eps` 1e-8,
the standard skipping rule) and each member's update of each parameter
tensor is capped at `max_step` in norm (1.0). Both are `optimizer_params`
keys; `null` restores the old behaviour. The candidate-fit log reports the
share of updates capped and pairs rejected per family and layer. On the
polynomial families the guards touch 0.1% of updates and no pair, and the
California Legendre baseline reproduces (0.1884 / 0.2772 vs 0.1877 /
0.2791); on the RBF family they end the runaway (0-6% of centers beyond 5
std against 37-68%) with about 15% of pairs rejected.

### Changed — end-to-end pass as a captured CUDA graph (`finetune_train.cuda_graph`, `data_on_device`)

`Trainer.train_finetune` was launch-bound: hundreds of tiny kernels per
step, 15-18 ms of latency for about 1 ms of work. On a CUDA device with
adam / adamw the step (gradient zeroing, forward, backward, update) is now
captured once as a CUDA graph over a static buffer of the loader's batch
shape and replayed per step, the next batch copied into the buffer, so the
dataset size does not matter; a batch of another shape (an epoch's tail)
runs the same step eagerly, so no row is dropped. Adam / AdamW are built
capturable with a tensor learning rate that the plateau scheduler's value
is copied into. `data_on_device` keeps the split resident on the device
(index-selected in a fresh permutation per epoch) when it fits in a quarter
of the free memory, or streams it from the loader with pinned non-blocking
copies. CPU, sgd, `cuda_graph: false` and a capture failure take the eager
loop, which has lost its per-step host sync. The loss is accumulated on the
device. California, three runs each: the pass 60-80 s -> 8-11 s with the
test numbers unchanged to 0.0001 (Legendre 0.1891-0.1894, RBF-8 0.1893-
0.1894); a whole Legendre run takes about 90 s.

### Changed — `SONN.infer`: the head's input columns cached on the device

The top-k column index the head reads from the last layer was recomputed
from `layer.err_values` on the CPU on every call, one host-to-device copy
per forward. It is now cached as a long tensor on the model's device, keyed
on the layer, `k`, the identity of `err_values` and the neuron counts (so
selection and `prune` invalidate it), and applied with `index_select`. The
prediction path therefore captures into a CUDA graph unchanged (end-to-end
step on the California Legendre model: 2.7 ms replayed vs 14-18 ms eager).

### Added — validation split reporting (`val_dl`) and `train.stop_source`

`Trainer.train` and `train_finetune` accept an optional `val_dl`, a split
that selects nothing. After every layer the trainer logs the best surviving
neuron's loss on it next to the dev error (`Layer #k: dev .. | val .. (gap
..)`) and keeps it on `layer.val_err` / `model.layer_val_err`; the
evolution of the gap shows whether the search is fitting the dev split.
`train.stop_source: val` routes the growth criterion and the end-to-end
early stop to that split (default `dev`, unchanged behaviour). The
California tutorial carves the split with `VAL_SPLIT` (every k-th training
row), off by default: measured at 10% it costs the fits 0.006 MSE on the
Legendre baseline and reading the stop rules from the 1239-row split costs
0.003-0.010 more, while the dev-val gap narrows with depth for both
families (no sign of dev fitting under the train stop).

### Changed — `rbf`: deterministic k-means start (`seeding: pca_quantiles`)

Lloyd's cluster sums are a one-hot batched matrix product instead of
`scatter_add_`, whose summation order on CUDA varied between runs and, with
many near-tied assignments, drifted the iterations apart. The default start
is no longer a seeded k-means++ draw but the rows at the quantiles of each
candidate's projection on the first principal axis of its input tuple
(`seeding: pca_quantiles`; `kmeans++` keeps the seeded draw, and old
checkpoints restore with it). The same data now gives the same centers
under any seed: three runs of the California RBF-8 config are identical at
every layer and end at 0.1893-0.1894 on test, where the random start had
given 0.1824-0.1957.

### Added — `train.early_stop_source`: candidate fits can stop on the training loss

`early_stop_source: train` makes the per-candidate early stop and its
learning-rate drop watch the training loss of the current batch instead of
the dev loss (same smoothing, patience and thresholds); the dev split is
then used once per layer, by selection, which is the method's regularizer.
Default `dev` is the historical behaviour. On the California Legendre
baseline the two agree (81-step fits either way, 0.1891-0.1894 vs
0.1882-0.1884 over three runs, two layers shallower); on the RBF family the
dev stop had been holding non-repeatable candidate fits to a common answer,
which the train stop exposes (three runs from 0.1824 to 0.1957).

### Changed — layer-growth stop rule: `train.stop_train_min_delta`, window from the last accepted improvement

`Trainer.train`'s growth rule is now `GrowthCriterion`, testable on a
scripted sequence. A layer improves when it lowers the error of the last
accepted improvement by at least `max(stop_train_min_delta,
stop_train_epsilon_condition * best)`, so small steps down add up, and the
search stops after `criterion_minimum_width` consecutive layers without an
improvement; the kept depth is still the layer with the lowest error. One
behaviour change with the old settings: a new best that falls short of the
relative margin no longer stops the search on the spot, it counts toward
the window. `stop_train_min_delta` (absolute, default 0 = off) gives the
rule a margin the dev evaluation can resolve: with the relative 1e-3 alone
(0.00017 at an error of 0.17) float32 summation order decided between 9
and 11 layers on the California tutorial, 0.005 on test; with 0.002 and a
width of 3 the same config repeats to 0.0002 at a fixed depth
(0.1882-0.1884 / 0.2795 over three runs). Per-layer log line with the gain,
the margin and the window count.

### Changed — ensemble LBFGS: batched two-loop recursion

The correction history of the batched parameters is now a pair of
`(B, history_size, P)` tensors with a per-member fill count, and the two-loop
recursion, the curvature test and the step cap run as batched operations
over the whole ensemble instead of a Python loop over its members. Same
algorithm (equal to the loop version to 1e-10 in float64); a Legendre layer
of the California tutorial goes from 24-30 s to 3-5 s and an RBF-8 layer
from 90-96 s to 11-15 s, with the same fit steps and guard statistics.
Checkpoints written by the per-member version still load. The float32
iterates differ at the rounding level, which on that tutorial moves where
the dev-driven growth criterion (0.1% relative margin) and the end-to-end
early stop fire: the same config now grows to 11 layers instead of 9 and
reads 0.193 instead of 0.188 on test, reproducibly, see the config header.

### Fixed — ensemble optimizers with parameters of rank > 2

`adam`, `sgd`, `newton` and `newton_lm` shaped the per-member learning rate
as `(num_neurons, 1)`, which only broadcasts against 2-D parameters; it is
now viewed to the parameter's rank (identical for the existing weights).

### Added — `train.censor_target_at`: censored least squares for capped targets

`NormMSE` takes an optional `censor_at`, wired from `train.censor_target_at`.
Rows whose target is at or above the cap have their prediction clipped to the
cap before the squared error, so predicting above the cap is free and pushes
no gradient, while predicting below it is penalized as before; uncensored rows
are untouched. This is the Tobit likelihood with the noise scale taken to
zero, and it keeps the objective a convex piecewise quadratic, so every fit
path (vmapped neuron fits, LBFGS head, end-to-end pass) works unchanged. The
neuron-selection criterion is not affected. Clip predictions to the cap at
inference. Off by default.

Motivation, California housing: prices are recorded as 5.0 for every house
worth 5.0 or more (4.8% of rows). Under plain squared error those rows carried
24% of the test MSE and dragged the fitted surface down around them: uncapped
houses priced 4-5 were under-predicted by 0.74 on average.

### Added — `train.layer_err_source: readout`

The layer-growth criterion in `Trainer.train` compares `layer.err` across
layers, and until now that was always the best surviving neuron's dev error
(`layer_err_source: neuron`, the default and unchanged). For a model that is
read out through an output head that is the wrong quantity, and it interacts
badly with `layer_finetune`: the per-layer fine-tune turns the survivors into a
basis for a head, so each one alone gets worse while the readout gets better,
and the criterion reads that as a regression and stops the search early.

`layer_err_source: readout` makes `layer.err` the best dev loss of a head over
all survivors, i.e. the readout a headed model is scored on: the fine-tune's
own head when `train.layer_finetune` is on, otherwise a temporary head fitted
over the frozen survivors purely as a measurement (the neurons are untouched,
so the search grows as before and only the stop decision changes). `Trainer`
raises `ValueError` when it is set without `model.use_output_projection: true`
(inference would read the best neuron, so the criterion has to score that
neuron), and `NotImplementedError` for multi-class models.
`_train_layer_finetune` gained a `freeze_neurons` flag and returns that dev
loss.

**What changes for you.** Nothing unless you set the key. Measured on the
California-housing tutorial (`california_housing_legendre_finetune.yaml`),
the fair criterion did not rescue the per-layer fine-tune there: its readout
plateaus from layer 0, so the search still stops at 3 layers and the calibrated
15-layer search with the fine-tune off remains better (test MSE 0.1972 vs
0.2099). The numbers are recorded in that config's header.

### Fixed — headless regressor `SONN.infer` returned the whole last layer

`SONN.infer` collapses the last layer to one prediction per sample whenever a
head is present (`out_proj`, or the multi-class projection paths), but a
`type: regressor` or `type: binary` model *without* `use_output_projection`
fell through to `return out` and handed back the raw last layer, shape
`(N, nbest_neurons)`. Everything downstream that compares predictions against
the `(N,)` target either broke or compensated by hand: the California-housing
tutorial crashed in `mean_squared_error` with
`y_true and y_pred have different number of output (1!=8)`, the CCPP tutorial
re-selected the best column itself, the README told you to `prune()` first so
inference returns one column, and `Trainer._finetune_prediction` carried its
own copy of the same selection.

`infer` now returns the best-error neuron's column, shape `(N,)`, on the
headless regressor / binary path — the neuron the layer error was scored on
and the one `prune()` keeps, so pruned and unpruned models predict the same
values. The workarounds are removed.

**What changes for you.** Only headless `regressor` / `binary` models. If your
code indexed `infer`'s output with `[:, col]` or reshaped a `(N, 1)` pruned
output, drop that step: the result is already `(N,)`. Models with
`use_output_projection: true` and multi-class models are unchanged.

## 0.1.3

### Fixed — `bias_error_l2` reduced over the wrong axis on multi-class logits

`bias_error_l2` was written for the scalar contract in its docstring —
`(..., N)` neuron outputs, summed over the sample axis — but `Trainer.bias_err`
hands it the `(ensemble, N, K)` logits the multi-class path produces. It summed
over the class axis instead, returning a per-sample `(ensemble, N)` tensor
normalized by `K` rather than one error per candidate normalized by `N`.

Both bias-using criteria raised. Under `criterion_type: validate_bias` the
tensor broadcast against the `(ensemble,)` regularity error and failed in
`SONN.get_error`; under `criterion_type: bias` it reached neuron selection and
failed there on a mask shape. Neither silently selected on a malformed error.

The function now takes a `logits` flag: `logits=True` sums over classes *and*
samples so `(ensemble, N, K)` collapses to `(ensemble,)`, and takes `N` from
`shape[-2]` so the denominator stays the one-hot label energy `Σ‖y_i‖² = N`.
The flag is explicit rather than inferred from `z_A.dim()` because
`(ensemble, N)` and `(N, K)` are both 2-D and mean different things. The scalar
form is unchanged.

**What changes for you.** Only multi-class runs with `train.bias_ce_type: l2`
*and* a bias-using criterion. The default `bias_ce_type` is `js`, whose
`bias_error_js` was always correct, and the default `criterion_type` is
`validate`, which computes no bias error at all — so a default configuration is
unaffected, and no previously-completed run produced a wrong number.

## 0.1.2

### Changed — regression criteria now normalize by target variance

`regularity_error`, `bias_error` and the `NormMSE` training loss divide by
`Σ(y - ȳ)²` instead of `Σ y²`. The reported error is now a genuine fraction of
variance unexplained (1 - R²): invariant to a constant offset on `y`, ≈1 at the
mean predictor, and in `[0, ~1]` for any dataset.

`Σ y²` is Ivakhnenko's original, but its baseline is the *zero* predictor, which
only means something on already-centered targets. On a target with mean 454 and
sd 17 the denominator is ~700x the variance, so the whole no-skill→perfect range
collapses into a band near zero — the reported error moves by five orders of
magnitude under a constant offset that a linear-in-coefficients model is
provably invariant to.

Set `train.error_normalization: energy` to restore the previous behaviour and
keep exact parity with gmdhpy and other classical implementations.

**What changes for you.** If your targets were already centered (the CCPP and
Concrete tutorials z-score theirs), nothing meaningful moves. If they were not,
expect the reported error to rise by roughly `1 + ȳ²/var`. California Housing,
the one tutorial that feeds raw targets, goes from a final layer error of 0.081
to 0.351 — the same model, honestly measured.

Depth can change too, and in the intended direction: the per-neuron early stop
compares the validation loss against an *absolute* `train.early_stop_patience`
(default 1e-4). On a scale-inflated loss that threshold can exceed the entire
improvement budget, stopping neurons before they fit. With the corrected scale
California Housing trains 15 layers instead of 13.

Candidate ranking within a layer is unaffected either way — the denominator is
constant across candidates in one evaluation — and the layer-stopping test
(`train.stop_train_epsilon_condition`) was already relative.

Multi-class is untouched: `regularity_error_ce` normalizes by `H(Y_B)` and the
bias criteria by label energy, both already proper no-skill baselines. Binary
classification goes through the regression criteria and does change — its
denominator becomes `n·p(1-p)`, the Bernoulli variance, which is the right
no-skill reference where `Σy² = n·p` was not.

### Added

- `train.error_normalization`: `"variance"` (default) | `"energy"`.
- `Trainer._fit_target_scale`: measures the training-set target spread once per
  `train()` and fixes the `NormMSE` denominator to it, so the training loss is
  batch-size-independent and on the same scale as the dev-set criterion.

### Fixed

- Layer-error logging fell back to `0.000` for any error below 5e-4. Small
  values now print in scientific notation (`torchsonn.utils.fmt_err`).