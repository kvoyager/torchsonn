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