# Criteria

A criterion gives every fitted candidate one number; lower is better.
[Selection](selection.md) keeps the candidates with the best numbers, and
the [growth rule](splits-and-stopping.md#the-growth-rule) compares layers by
them. `train.criterion_type` chooses the criterion:

| Value | Candidates are ranked by |
|---|---|
| `validate` (default) | the regularity error on the dev split |
| `bias` | the minimum-bias error between two half-fits |
| `validate_bias` | $(1 - \alpha)\cdot\text{bias} + \alpha\cdot\text{regularity}$, with $\alpha$ = `train.error_alpha` (0.5) |
| `bias_retrain` | accepted by the configuration but not implemented: training raises `NotImplementedError` when the first layer selects its survivors |

[GMDH](gmdh.md) explains the classical criteria these follow.

## Regression

### Regularity error

The candidate is fitted on the train split and scored on the dev split:

$$
\Delta^2 = \frac{\sum_{i \in \text{dev}} \big(y_i - \hat{y}_i\big)^2}{\sum_{i \in \text{dev}} \big(y_i - \bar{y}\big)^2}
$$

where $\bar{y}$ is the mean of the dev targets. This is one minus the
coefficient of determination: 0 for a perfect fit, 1 for a candidate no
better than predicting the mean, above 1 for one that is worse.

### Minimum-bias error

The candidate is fitted twice, on the even rows (A) and on the odd rows (B)
of the train split, each fit early-stopping on the matching half of the dev
split. The criterion measures how much the two fits disagree over all the
training rows:

$$
\eta^2 = \frac{\sum_{i \in \text{train}} \big(\hat{y}^{A}_i - \hat{y}^{B}_i\big)^2}{\sum_{i \in \text{train}} \big(y_i - \bar{y}\big)^2}
$$

A candidate that captures real structure gives nearly the same predictions
whichever half it was fitted on.

!!! warning "`bias` on its own leaves the survivors untrained"
    With `criterion_type: bias`, only the two half-fits are trained. The
    survivors that selection keeps are the layer's own neurons, which keep
    their random initial coefficients, so the network predicts like an
    untrained one. `validate_bias` trains the layer's neurons on the full
    train split and is not affected.

### Normalization

`train.error_normalization` sets the denominator of both regression
criteria and of the training loss:

- **`variance`** (default): $\sum (y - \bar{y})^2$, as above. The error is a
  fraction of the target's variance, so it means the same on every dataset,
  and the absolute thresholds that compare errors (the early-stop and
  growth margins, `train.divergence_threshold`) do too.
- **`energy`**: $\sum y^2$, Ivakhnenko's original and GmdhPy's. Its baseline
  is the prediction 0, which suits targets centred on 0. For a target with
  mean 454 and standard deviation 17, $\sum y^2$ is about 700 times the
  variance, and every error collapses into a thin band near 0.

Within one layer the denominator is the same for every candidate, so the
choice does not change which candidates survive; it changes the reported
errors and what the absolute thresholds mean.

The training loss of a regression candidate uses the same scale: its
squared error divided by the variance of the training targets (their mean
square under `energy`), measured once before the first layer. `train.ridge_alpha` adds an L2 penalty on the
neuron's coefficients to the training loss only; criteria are computed
without it.

### Censored targets

Some targets are recorded at a cap: California housing prices stop at 5.0
for every house worth 5.0 or more, 4.8% of the rows. With
`train.censor_target_at` set to the cap, the training loss clips the
prediction to the cap on rows whose target sits at or above it, so
predicting above the cap costs nothing there. The criteria are unchanged.
Clip the predictions to the cap at inference as well.

## Classification

### Multi-class regularity error

The candidate's class probabilities are scored on the dev split by their
cross-entropy, divided by the entropy of the dev class frequencies:

$$
\text{NCE} = \frac{-\frac{1}{N}\sum_{i \in \text{dev}} \log p(y_i \mid x_i)}{H(Y)},
\qquad H(Y) = -\sum_{k} \pi_k \log \pi_k
$$

where $\pi_k$ is the share of class $k$ on the dev split. Predicting the
class frequencies for every row gives 1, and a perfect classifier 0.

### Multi-class bias error

`train.bias_ce_type` chooses how the two half-fits' disagreement is
measured:

- **`js`** (default): the Jensen-Shannon divergence between their predicted
  class distributions, summed over the rows and divided by $N \ln 2$, its
  largest possible value. The result lies in $[0, 1]$.
- **`l2`**: the squared difference between their class scores (logits),
  summed over rows and classes and divided by $N$.

### Binary models

`type: binary` uses the regression formulas above, applied to the neuron's
raw output and the 0/1 labels. The raw output is a logit, because a binary
model trains with the binary cross-entropy on logits. With `variance`
normalization the denominator is $N\,p(1-p)$, where $p$ is the share of
positive labels.

## The layer's error

After selection, each layer gets one error, which the growth rule
compares across layers:

- **`train.layer_err_criterion: top`** (default): the best survivor's
  criterion value. **`avg`**: the mean over the survivors.
- **`train.layer_err_source: neuron`** (default): the survivors' own
  criterion values, as above. **`readout`**: the dev loss of a linear head
  fitted over all the layer's survivors, which is what a model with an
  output head is scored on. With `train.layer_finetune` on, that is the
  fine-tune's own head; otherwise a temporary head is fitted over the
  frozen survivors only to measure the layer. `readout` requires
  `model.use_output_projection: true` and works for regression and binary
  models only.

On California housing, `readout` together with `layer_finetune` stops the
search at 3 layers with a test MSE of 0.2099, against 0.1972 for the
15-layer search with the per-layer fine-tune off (changelog).

## The knobs

`train.criterion_type`, `train.error_alpha`, `train.bias_ce_type`,
`train.error_normalization`, `train.censor_target_at`, `train.ridge_alpha`,
`train.layer_err_criterion`, `train.layer_err_source`. See
[Configuration keys](../reference/config.md).

<small>Checked against TorchSONN 0.1.5.</small>
