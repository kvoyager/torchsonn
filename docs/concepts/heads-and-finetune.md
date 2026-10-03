# Heads and fine-tuning

The search fits every neuron on its own and freezes it once its layer is
done; nothing in it trains two neurons together. This page covers what
reads the prediction off the last layer, the three optional passes that do
train neurons together, and pruning:

| Step | Call or key | Trains |
|---|---|---|
| head fit | `Trainer.train_out_proj` | a linear head over the last layer's survivors |
| per-layer fine-tune | `train.layer_finetune` | each layer's survivors together, as the search goes |
| end-to-end pass | `Trainer.train_finetune` | every parameter of the network at once |
| pruning | `Trainer.prune` | nothing: it deletes what the prediction does not use |

## Without a head

By default the model predicts with one neuron: the survivor of the last
layer with the lowest criterion value. For regression `infer` returns that
neuron's output, for `binary` its output is the logit, and a `multi-class`
model passes it through its class map (see
[Classification](classification.md)). The result is the classical GMDH
model, a single polynomial of polynomials of the inputs. The one exception
is a multi-class model with a projection per candidate, which combines
every survivor of the last layer.

## The output head

`model.use_output_projection: true` adds `out_proj`, a linear layer over
the last layer's survivors: `Linear(num_out_neurons, 1)` for regression,
and `Linear(num_out_neurons, C)` followed by the log-softmax for
multi-class. Binary models have no head and ignore the key.

The head reads the `model.num_out_neurons` survivors with the lowest
criterion values. Set it to `model.nbest_neurons`. Left at null, it takes
the value of `model.max_neuron_models` (the candidate cap, usually larger),
and the head gets zeros for the inputs the layer cannot fill; with both at
null, building the model fails with `TypeError`.

### Fitting the head

The search does not train the head. Until `Trainer.train_out_proj` fits it,
the head has random weights and the model's predictions are meaningless:

```python
trainer.train(model, train_dl, dev_dl, test_dl)
trainer.train_out_proj(model, train_dl, dev_dl)
```

`train_out_proj` freezes the rest of the network, computes its outputs once
per split, fits the head on the train split and early-stops on dev. It
then saves the model to `model_last.ckpt`. `train.out_proj_train` sets the
optimizer (`adam` by default, `sgd` or `lbfgs`), the learning rate, the step
limit, the evaluation interval and the early stop.

Fitting the head is least squares for regression and logistic regression
for multi-class: small convex problems that `lbfgs` with its line search
solves in a few dozen steps. The default `adam` at a learning rate of 0.001
can stop long before the optimum: on iris it left a test log loss of 0.2907,
against 0.0089 with `lbfgs`. The California housing configs fit the head
with:

```yaml
train:
  out_proj_train:
    optimizer: lbfgs
    lr: 0.1
    max_steps: 100
    eval_interval: 5
    early_stop_patience: 6
```

A head on its own does not always help. On the regression quickstart, a
head fitted this way over the 8 survivors lowered the dev error from 0.3143
to 0.3040 but raised the test MSE from 0.4199 to 0.4316. The head pays off
as the starting point of the end-to-end pass, below.

## The per-layer fine-tune

`train.layer_finetune: true` adds a step to every layer, after selection:
the survivors train together through a temporary linear head, against the
model's training loss, on the train split, with an early stop on dev. Every
parameter the survivors own trains, except a per-candidate class map: their
coefficients and, for the RBF family, their centres and widths. The temporary head is then discarded.
The pass takes its settings from `train.out_proj_train`, like the head
fit.

The pass turns the survivors from individual predictors into a basis for a
linear combination: together they fit better, each alone fits worse. That
has two consequences.

- **It needs an output head.** A model without one predicts with a single
  survivor, which the pass has made worse: on CCPP that readout collapses
  to a mean absolute error of about 13 MW, against 3.3 MW for the plain
  model (CCPP README). Set `model.use_output_projection` and keep
  `num_out_neurons` equal to `nbest_neurons`. A head over 6 of 8 survivors
  cost 0.05 MW and doubled the spread across folds.
- **It changes where the search stops.** The growth rule compares the
  survivors' own criterion values by default, and those get worse, so the
  search stops early. `train.layer_err_source: readout` scores each layer
  by the temporary head's dev loss instead (see
  [Criteria](criteria.md#the-layers-error)).

On California housing the per-layer fine-tune made the model worse
(California housing fine-tune config):

| Survivors | Per-layer fine-tune | Layers | Test MSE |
|---|---|---|---|
| 8 | on | 5 trained | 0.2160 |
| 8 | off | 14 | 0.2048 |
| 16 | on | 4 trained, 3 kept | 0.2102 |
| 16 | on, with `layer_err_source: readout` | 5 trained, 3 kept | 0.2099 |
| 16 | off | 15 | 0.1986 |

On CCPP it is part of the stack that helps most (see below). The pass does
not work on binary models: it fails with `ValueError` at the first layer.

## The end-to-end pass

`Trainer.train_finetune(model, train_dl, dev_dl, cfg=None, val_dl=None)`
trains every parameter of the network at once: all layers' coefficients,
RBF centres and widths, the class map and the head. Its loss is the
model's training loss on exactly what `infer` returns, so it optimizes the
readout the model is scored on.

```python
trainer.train(model, train_dl, dev_dl, test_dl)
trainer.train_out_proj(model, train_dl, dev_dl)
trainer.train_finetune(model, train_dl, dev_dl)
trainer.save_model_checkpoint(model)
preds, targets = trainer.infer(model, test_dl)
```

`train.finetune_train` (or a block of the same shape passed as `cfg`) sets
the optimizer (`adam`, `adamw` or `sgd`; there is no LBFGS here), the
learning rate, the plateau schedule, the step limit and the early stop.
The early stop reads the dev split, or the validation split when
`train.stop_source` is `val`.

On the regression quickstart, with a head and this block, the pass stopped
on its own after 1,375 to 2,925 steps, 1.5 to 3.5 minutes on a CPU:

```yaml
train:
  finetune_train:
    optimizer: adamw
    lr: 1.0e-4
    weight_decay: 1.0e-4
    max_steps: 20000
    eval_interval: 25
    early_stop_patience: 20
    early_stop_min_delta: 1.0e-6
    lr_min: 1.0e-8
```

| Regression quickstart readout | Test MSE |
|---|---|
| best neuron, no head | 0.4199 |
| head fitted with `lbfgs` | 0.4316 |
| head, then the end-to-end pass, three runs | 0.3839 to 0.3987 |

Things to know before using it:

- **Use a small learning rate.** Every weight moves at once, so the pass
  wants a far gentler step than the head fit. The California housing
  configs use AdamW at `1e-4`, the CCPP configs at `1e-5`.
- **Check that it converged.** The log prints `finetune early stop at step
  ...` when the pass stops on its own. Without that line, it ran to
  `max_steps` and its result is cut short. On CCPP,
  `ccpp_legendre_finetune.yaml` needed `max_steps: 20000`: at 5000 only 6
  of 10 folds converged, and the mean absolute error was 3.1916 MW instead
  of 3.1748.
- **Keep the head.** On CCPP, removing the head and fine-tuning the bare
  network against its best neuron scored 3.4637 ± 0.2730 MW, worse than no
  fine-tuning at all (3.3084 ± 0.0147). With the head, every fold's dev loss
  improved or held.
- **It does not repeat exactly.** The search and the head fit give the
  same numbers run after run, but the pass does not: in three runs of the
  quickstart example with the same seed, it stopped after 1,375, 2,925 and
  1,375 steps, at a test MSE of 0.3978, 0.3839 and 0.3987.
- **It widens the spread.** The fine-tuned CCPP models vary more from fold
  to fold: ±0.05 to 0.07 MW, against ±0.013 to 0.016 without the pass.
- **It keeps the last step's weights,** not those of its best evaluation,
  and it does not save the model: call `trainer.save_model_checkpoint`.
- **Run it before `Trainer.infer`.** On a model with a head, a call to
  `Trainer.infer` before the pass makes the pass fail (see
  [Troubleshooting](../guides/troubleshooting.md#known-problems)).

On a CUDA device with `adam` or `adamw`, the pass runs as a captured CUDA
graph, which is much faster (see [Performance](../guides/performance.md#the-end-to-end-pass-on-a-gpu)).

### What the stack is worth on CCPP

The CCPP tutorial's `*_finetune` configs add all three passes, the
per-layer fine-tune, the head and the end-to-end pass, to four of its
configurations and change nothing else. Mean absolute error over 10 folds
(CCPP README):

| Config | Without | With the three passes |
|---|---|---|
| `ccpp_legendre.yaml` | 3.3084 ± 0.0147 | 3.1748 ± 0.0520 |
| `ccpp_legendre_poly.yaml` | 3.2782 ± 0.0135 | 3.1495 ± 0.0696 |
| `ccpp_legendre_heavy.yaml` | 3.3035 ± 0.0162 | 3.1408 ± 0.0620 |
| `ccpp_legendre_poly_heavy.yaml` | 3.2777 ± 0.0200 | 3.1284 ± 0.0485 |

The gain, 0.13 to 0.16 MW, is larger than the difference between the
cheapest and the most expensive search.

### In the tutorial scripts

The California housing and CCPP scripts read three top-level flags:
`finetune_end_to_end` runs the pass after the head fit,
`finetune_drop_head` removes the head first, and `finetune_prune_first`
prunes the network first. Pruning first leaves the accuracy unchanged,
because the neurons it removes get no gradient anyway.

## Pruning

`Trainer.prune(model)` deletes every neuron the prediction does not reach.
It keeps the neurons the readout reads in the last layer, one without a
head and all `out_proj.in_features` with one, then works down the layers,
keeping whatever those neurons read. It follows any input layout,
including inputs from older layers under `model.shortcut.prev_layers`, and
deletes a layer that nothing reads any more. The predictions stay the
same, with the one exception in the warning below.

| Regression quickstart | Neurons per layer after `prune` |
|---|---|
| no head | 2, 1, 1, 1, 1, 1, 1, 1, 1, 1 |
| head over 8 survivors | 3, 2, 2, 2, 2, 3, 5, 5, 4, 8 |

Without a head, 11 of the 80 trained neurons remain and the model uses 7 of
the 8 features; with a head, 36 neurons remain and use all 8.

!!! warning
    A multi-class model with `model.use_neuron_proj` and no head predicts
    from every survivor of its last layer, but `prune` keeps one, which
    changes its predictions (see
    [Classification](classification.md#a-projection-per-candidate)).

## The knobs

`model.use_output_projection`, `model.num_out_neurons`,
`train.layer_finetune`, `train.layer_err_source`, `train.out_proj_train`,
`train.finetune_train`, `train.stop_source`. See
[Head fit and per-layer fine-tune](../reference/config.md#head-fit-and-per-layer-fine-tune)
and [End-to-end pass](../reference/config.md#end-to-end-pass).

<small>Checked against TorchSONN 0.1.5.</small>
