# Classification

Every neuron outputs one number, and a classifier needs a score for every
class. This page covers how a `multi-class` model turns one into the other,
how `binary` models work, the losses, and class weights. The
[classification quickstart](../getting-started/quickstart-classification.md)
trains a multi-class model on iris.

## The two classifier types

| `model.type` | Labels | Training loss | `infer` returns |
|---|---|---|---|
| `binary` | 0 or 1; `num_classes: 2` | binary cross-entropy, reading the neuron's output as a logit | one logit per row |
| `multi-class` | the class index 0, 1, ..., C−1 as integers; `num_classes` above 2 | negative log-likelihood of the log-softmax class scores | one row of C log-probabilities per sample |

The default `model.type` is `multi-class` with `num_classes: 3`, so a
regression model has to say `type: regressor`.

## Binary models

A binary model's neuron predicts the logit of the positive class. Its
probability is the sigmoid of the prediction:

```python
logits, targets = trainer.infer(model, test_dl)
probs = torch.sigmoid(logits)
```

The criterion uses the regression formulas on the logit and the 0/1 labels
(see [Criteria](criteria.md#binary-models)).

!!! warning "Binary selection picks poor neurons"
    On a synthetic two-class task, a `binary` model reached a test accuracy
    of 0.467, chance level, and a `regressor` on the same 0/1 labels 0.943.
    Train two-class problems as a regressor on the 0/1 labels, and read a
    prediction above 0.5 as the positive class. `multi-class` does not
    accept two classes.

A binary model has no output head: `model.use_output_projection` has no
effect on it, and the best neuron of the last layer always makes the
prediction. The per-layer fine-tune and `train.layer_err_source: readout`
need a head and fail on a binary model with `ValueError` at the first
layer.

## Multi-class: from one number to class scores

Two keys choose how a neuron's output becomes C class scores:

| `soft_binner` | `use_neuron_proj` | Class map |
|---|---|---|
| `true` (default) | `false` | the soft binner: fixed, nothing to learn |
| `false` | `false` | one shared projection, learned |
| `false` | `true` | a projection per candidate, learned |
| `true` | `true` | rejected: building the model fails with `AssertionError` |

The class map is part of every candidate's fit, so it shapes what the
neurons learn and how the criterion scores them.

### The soft binner

The soft binner places the C classes at fixed points $c_k$, evenly spaced
between 0.05 and 0.95 in label order, and scores class $k$ by how close the
neuron's output $x$ lies to its point:

$$
z_k = -s\,(x - c_k)^2, \qquad c_k = 0.05 + 0.9\,\frac{k}{C - 1}
$$

where $s$ is `model.soft_binner_scale` (100). The softmax of the scores
gives the class probabilities. Each neuron therefore learns a regression
onto the point of the right class. With three classes the points are 0.45
apart, so an output that lands on a class's point gives it a lead of about
20 in score over its neighbours, a probability close to 1.

The binner puts the classes on a line in label order: class 1 lies between
classes 0 and 2. A neuron has to separate them by one number in that
order, which suits labels with a natural order best. Iris is not ordered,
and the quickstart still classifies every test flower correctly.

### The shared projection

With `soft_binner: false`, one learned linear map turns a neuron's output
into the class scores, $z_k = w_k x + b_k$. It starts as the soft binner:
$w_k = 2 s c_k$ and $b_k = -s c_k^2$, which are the binner's scores without
the $-s x^2$ term that is the same for every class and cancels in the
softmax. It then trains together with the candidates, at their learning
rate times `train.shared_proj_lr_multiplier` (0.1). All the candidates of a
family share it. The model keeps one map: the one trained with the
survivors of the best layer.

The scores are still linear in one number, so the classes still take turns
along a line, but the map learns their order and spacing instead of fixing
them by label.

### A projection per candidate

With `use_neuron_proj: true` (which needs `soft_binner: false`), every
candidate gets its own map from its output to the class scores, initialized
with Xavier-uniform weights and zero biases, and trained with the
candidate's coefficients. To predict, the model combines every survivor of
the last layer: it multiplies each survivor's output by that survivor's
weights, adds the products up, adds the mean of the survivors' biases and
takes the log-softmax. The prediction therefore reads the whole last layer,
not one neuron.

!!! warning "Pruning changes this model's predictions"
    `Trainer.prune` keeps one neuron in the last layer of a model without an
    output head, but a per-candidate-projection model predicts from all of
    them. On iris, pruning such a model moved its test log loss from 0.0001
    to 0.0168. Do not prune a `use_neuron_proj` model that has no head.

### A head over the survivors

`model.use_output_projection: true` adds a learned linear layer from the
last layer's survivors to the C class scores, followed by the log-softmax
(see [Heads and fine-tuning](heads-and-finetune.md)). The class map still
drives the search; the head replaces it at prediction, once
`Trainer.train_out_proj` has fitted it. The
[Otto tutorial](../tutorials/otto.md) combines a projection per candidate
during the search with a head over all 93 survivors.

### On iris

Each class map, on the classification quickstart's data and settings:

| Class map | Layers kept | Test log loss |
|---|---|---|
| soft binner (the quickstart) | 4 | 0.0029 |
| shared projection | 5 | 0.0071 |
| projection per candidate | 3 | 0.0001 |
| soft binner, plus a head fitted with `lbfgs` | 4 | 0.0089 |
| soft binner, plus a head fitted with the default `adam` | 4 | 0.2907 |

All five classify the 23 test flowers correctly. With so few rows, the log
losses show that each variant works, not which one is better. The last row
shows that the head fit needs its own settings: see
[Fitting the head](heads-and-finetune.md#fitting-the-head).

## The criterion

The multi-class criterion is the cross-entropy divided by the entropy of
the class frequencies: 1 for a model that predicts only the class
frequencies, 0 for a perfect one (see [Criteria](criteria.md)). The binary
criterion is the regression formula on the logit.

## Class weights

For imbalanced classes, pass a weight per class to the `Trainer`:

```python
import numpy as np
import torch
from torchsonn import Trainer

counts = np.bincount(y_train)
class_weights = torch.tensor(counts.sum() / (len(counts) * counts),
                             dtype=torch.float32)
trainer = Trainer(config, class_weights=class_weights)
```

This is the Otto tutorial's choice: each class weighted by the inverse of
its frequency, so that every class carries the same total weight. For
`multi-class`, the tensor has one weight per class. For `binary`, it has one
element, the weight of the positive class.

The weights change the training loss of the candidates, the head fit and
the fine-tune passes. The criterion that selects the survivors and decides
the depth stays unweighted.

## The knobs

`model.type`, `model.num_classes`, `model.soft_binner`,
`model.soft_binner_scale`, `model.use_neuron_proj`,
`train.shared_proj_lr_multiplier`, `model.use_output_projection`. See
[Configuration keys](../reference/config.md#multi-class-heads).

<small>Checked against TorchSONN 0.1.5.</small>
