# Changelog

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