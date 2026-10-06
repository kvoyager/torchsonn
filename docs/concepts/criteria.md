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

Any other value raises `ValueError` when the model is built.

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

The two half-fits only rank the candidates. Under every criterion the
candidates are also fitted on the whole train split, with the early stop
on dev, and those fits are the survivors the layer keeps; `bias` and
`validate_bias` therefore run three fits per family instead of one.

Ranking by agreement is not the same as ranking by accuracy: a candidate
can be stable across the two half-fits without being the most accurate one,
so `bias` on its own can select differently from `validate`. `validate_bias`
blends the two, and `validate` is the default.

### Normalization

`train.error_normalization` sets the denominator of both regression
criteria and of the training loss:

- **`variance`** (default): $\sum (y - \bar{y})^2$, as above. The error is a
  fraction of the target's variance, so it means the same on every dataset,
  and the absolute thresholds that compare errors (the early-stop and
  growth margins, `train.divergence_threshold`) do too.
- **`energy`**: $\sum y^2$, as in the original GMDH (and GmdhPy). Its
  baseline is the prediction 0, which suits targets centred on 0. For a
  target far from 0, $\sum y^2$ can be many times the variance, and the
  errors collapse into a thin band near 0.

Within one layer the denominator is the same for every candidate, so the
choice does not change which candidates survive; it changes the reported
errors and what the absolute thresholds mean.

The training loss of a regression candidate uses the same scale: its
squared error divided by the variance of the training targets (their mean
square under `energy`), measured once before the first layer. `train.ridge_alpha` adds an L2 penalty on the
neuron's coefficients to the training loss only; criteria are computed
without it.

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
  fine-tune's own head, and the error is the dev loss of the weights the
  layer keeps: its last evaluation's, or its best under
  `train.out_proj_train.keep_best_weights`. Otherwise a temporary head is
  fitted over the frozen survivors only to measure the layer, and the
  error is its lowest dev loss. `readout` requires
  `model.use_output_projection: true` and works for regression models
  only: multi-class models reject it, and binary models, which have no
  head, fail with `ValueError` at the first layer.

## The knobs

`train.criterion_type`, `train.error_alpha`, `train.bias_ce_type`,
`train.error_normalization`, `train.ridge_alpha`,
`train.layer_err_criterion`, `train.layer_err_source`. See
[Configuration keys](../reference/config.md).

<small>Checked against TorchSONN 0.1.5.</small>
