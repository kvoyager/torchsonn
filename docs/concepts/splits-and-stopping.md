# Splits and stopping

TorchSONN makes many decisions from data: when each candidate's fit stops,
which candidates survive, how deep the network grows. This page says which
split each decision reads and how the two stop rules work, the one inside a
candidate's fit and the one between layers.

## What each split decides

| Split | Decides |
|---|---|
| train | every coefficient; the optimizer minimizes the training loss on it |
| dev | when each candidate's fit stops (by default), the [criterion](criteria.md) and [selection](selection.md), the growth rule, and when the head fit and the end-to-end pass stop |
| validation (optional) | nothing by default; its error is reported per layer. Under `train.stop_source: val`, the growth rule and the end-to-end pass's early stop |
| test | nothing; it measures the finished model |

Every decision the dev split makes is a choice of the best option by dev
error, so the dev error of the result is optimistic: at every layer it is
the minimum over many candidates. The validation split, which chooses
nothing, shows by how much. [The algorithm](algorithm.md#the-splits-across-a-run)
draws the splits across a whole run.

## Stopping a candidate's fit

All candidates of a family are fitted together, and each stops on its own:

1. Every `train.eval_step_interval` steps, each candidate's loss is
   evaluated: on the dev split by default, or on the current training batch
   under `train.early_stop_source: train`.
2. The loss is smoothed over evaluations by an exponential moving average
   with weight `train.eval_smoothing_factor` (0.2) on the newest value.
3. An evaluation counts as an improvement when the smoothed loss falls below
   the best so far by more than `train.early_stop_patience`, an absolute
   amount (1e-4 by default).
4. After `train.early_stop_tolerance_steps` evaluations in a row without an
   improvement, the candidate's learning rate is multiplied by `gamma` from
   `train.optimizer.optimizer_params` (0.5) and the count restarts. Once the
   learning rate is down to `min_lr`, the candidate stops.
5. A candidate whose loss turns NaN, or exceeds
   `train.divergence_threshold` (infinite, so off, by default), stops at
   once. Candidates with a NaN criterion are dropped at selection.

The fit of a family ends when every candidate has stopped, when more than
`train.early_stop_completion_percentage` percent of them have (100, so
never before all, by default), or when the step count passes
`train.steps`. These are checked only at evaluations, so a fit
runs to the first evaluation past `train.steps`; with the default
`eval_step_interval` of 1000, any `steps` below 1000 has no effect. The
names read the other way round from their roles: `early_stop_patience` is
the minimum improvement, and `early_stop_tolerance_steps` the patience,
counted in evaluations.

With `early_stop_source: train`, the fits stop on the training loss, and
the dev split is used once per layer, by selection: least squares on train, ranking on
dev, the classical GMDH arrangement. On the California housing Legendre
configuration, both settings stop the fits at 81 steps and give test errors
within 0.0012 of each other over three runs each (changelog).

## The growth rule

After every layer, the growth rule decides whether to build another. It
keeps two numbers: the lowest layer error so far, and the error of the last
layer that counted as an improvement. A new layer counts as an improvement
when it lowers that last improving error by at least the margin

$$
\text{margin} = \max\big(\delta,\ \varepsilon \cdot e_{\text{best}}\big)
$$

where $\delta$ is `train.stop_train_min_delta` (0 by default), $\varepsilon$
is `train.stop_train_epsilon_condition` (0.001) and $e_{\text{best}}$ the
lowest error so far. Several small gains add up, because each is measured
from the last improving layer rather than from the layer just before. The
search stops after `train.criterion_minimum_width` (5) layers in a row
without an improvement, or at `train.max_layer_count` layers, and the model
keeps the layers up to the one with the lowest error, whether or not that
layer cleared the margin.

### A worked example

The [regression quickstart](../getting-started/quickstart-regression.md)
with `max_layer_count: 30` instead of 10, and nothing else changed, logs
these growth-rule lines (a selection):

```text
Layer #10: error 0.3110, best 0.3110 at layer 10; improved by +0.0033 (margin 0.0003); 0 of 5 layers without improvement
Layer #11: error 0.3114, best 0.3110 at layer 10; improved by -0.0003 (margin 0.0003); 1 of 5 layers without improvement
Layer #12: error 0.3116, best 0.3110 at layer 10; improved by -0.0006 (margin 0.0003); 2 of 5 layers without improvement
Layer #13: error 0.3082, best 0.3082 at layer 13; improved by +0.0028 (margin 0.0003); 0 of 5 layers without improvement
...
Layer #25: error 0.2982, best 0.2982 at layer 25; improved by +0.0009 (margin 0.0003); 0 of 5 layers without improvement
Layer #26: error 0.2988, best 0.2982 at layer 25; improved by -0.0006 (margin 0.0003); 1 of 5 layers without improvement
Layer #27: error 0.2991, best 0.2982 at layer 25; improved by -0.0009 (margin 0.0003); 2 of 5 layers without improvement
Layer #28: error 0.2986, best 0.2982 at layer 25; improved by -0.0004 (margin 0.0003); 3 of 5 layers without improvement
Layer #29: error 0.2983, best 0.2982 at layer 25; improved by -0.0000 (margin 0.0003); 4 of 5 layers without improvement
```

- Layers 11 and 12 are worse than layer 10, so the count rises to 2.
  Layer 13 beats layer 10 by 0.0028, more than the margin, and the count
  starts again.
- Each gain is measured from the last improving layer: layer 12's −0.0006
  compares 0.3116 with layer 10's 0.3110, not with layer 11's 0.3114.
- After layer 25, four layers in a row bring nothing. A fifth would have
  ended the search; here `max_layer_count` ends it first, after layer 29,
  and the model keeps layers 0 to 25.

The 26-layer model reaches a test MSE of 0.3994, against 0.4199 for the
quickstart's 10 layers.

### Choosing the margin

With $\delta = 0$ the margin is relative: 0.1% of the best error, 0.00017
at an error of 0.17. That is below what a dev evaluation on a few thousand
rows can resolve, so rounding noise decides whether a layer counts. On the
California housing tutorial, the order of float32 summation alone moved the
search between 9 and 11 layers and the test MSE by 0.005 (comment on
`train.stop_train_min_delta`). With `stop_train_min_delta: 0.002` and
`criterion_minimum_width: 3`, the same configuration repeats over three
runs to within 0.0002 at a fixed depth (changelog). Set $\delta$ near the
noise level of the dev error.

## The validation split

`Trainer.train` and `Trainer.train_finetune` accept an optional `val_dl`, a
split that selects nothing. After every layer the log then compares the
best survivor's error on both splits:

```text
Layer #<k>: dev <dev error> | val <validation error> (best neuron on each split; gap <val - dev>)
```

The validation error is measured with the training loss, the squared error
divided by the training targets' variance. The dev error divides by the dev
targets' spread instead, so the two sit on nearly the same scale.
A gap that keeps growing with depth means the search is fitting the dev
split. With `train.stop_source: val`, the growth rule and the end-to-end
pass's early stop read the validation split instead of dev; `stop_source:
val` without a `val_dl` raises an error. Validation errors are not computed
for multi-class models.

The extra split has a cost: its rows fit nothing. On California
housing, a 10% validation split cost the fits 0.006 in test MSE, and
reading the stop rules from its 1,239 rows cost another 0.003 to 0.010, while
the dev-validation gap narrowed with depth (changelog). The tutorial
therefore leaves it off (`tutorial.val_split: 0`).

## The knobs

`train.eval_step_interval`, `train.eval_smoothing_factor`,
`train.early_stop_patience`, `train.early_stop_tolerance_steps`,
`train.early_stop_source`, `train.steps`, `train.divergence_threshold`,
`train.early_stop_completion_percentage`, `train.optimizer.optimizer_params`
(`min_lr`, `gamma`), `train.stop_train_min_delta`,
`train.stop_train_epsilon_condition`, `train.criterion_minimum_width`,
`train.max_layer_count`, `train.stop_source`. See
[Configuration keys](../reference/config.md).

<small>Checked against TorchSONN 0.1.5.</small>
