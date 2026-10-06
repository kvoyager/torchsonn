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
criterion values. Left at null, it reads `model.nbest_neurons` of them,
every survivor of the last layer.

### Fitting the head

The search does not train the head. Until `Trainer.train_out_proj` fits it,
the head has random weights and the model's predictions are meaningless:

```python
trainer.train(model, train_dl, dev_dl, test_dl)
trainer.train_out_proj(model, train_dl, dev_dl)
```

`train_out_proj` freezes the rest of the network, computes its outputs once
per split, fits the head on the train split and early-stops on dev. It
ends on its last step (see
[Last step or best evaluation](#last-step-or-best-evaluation)) and saves
the model to `model_last.ckpt`. `train.out_proj_train` sets the
optimizer (`adam` by default, `sgd` or `lbfgs`), the learning rate, the step
limit, the evaluation interval and the early stop.

Fitting the head is least squares for regression and logistic regression
for multi-class: small convex problems that `lbfgs` with its line search
solves in a few dozen steps. The default `adam` at a learning rate of 0.001
can stop well short of what `lbfgs` reaches, so an `lbfgs` block is usually
the better choice for the head:

```yaml
train:
  out_proj_train:
    optimizer: lbfgs
    lr: 0.1
    max_steps: 100
    eval_interval: 5
    early_stop_patience: 6
```

A head on its own does not always help: fitting it can lower the dev error
yet raise the test error, since it only optimizes the readout on the splits
it sees. The head pays off mainly as the starting point of the end-to-end
pass, below.

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
  survivor, which the pass has made worse, so that readout collapses. Set
  `model.use_output_projection` and let the head read every survivor
  (`num_out_neurons` null); reading only some of them tends to cost
  accuracy and widen the spread across folds.
- **It changes where the search stops.** The growth rule compares the
  survivors' own criterion values by default, and those get worse, so the
  search stops early. `train.layer_err_source: readout` scores each layer
  by the temporary head's dev loss instead (see
  [Criteria](criteria.md#the-layers-error)).

The per-layer fine-tune does not always pay off. On some data it stops the
search earlier and ends up a little worse than leaving it off; on other data
it is part of the stack that helps most (see
[the end-to-end pass](#the-end-to-end-pass)). It is worth measuring both
ways on your own data. The pass does not work on binary models: it fails
with `ValueError` at the first layer.

## The end-to-end pass

`Trainer.train_finetune(model, train_dl, dev_dl, cfg=None, val_dl=None)`
trains every parameter of the network at once: all layers' coefficients,
RBF centres and widths, a learned class map (the shared projection or one
per candidate; the soft binner's points stay fixed) and the head. Its loss
is the model's training loss on exactly what `infer` returns, so it
optimizes the readout the model is scored on.

```python
trainer.train(model, train_dl, dev_dl, test_dl)
trainer.train_out_proj(model, train_dl, dev_dl)
trainer.train_finetune(model, train_dl, dev_dl)
preds, targets = trainer.infer(model, test_dl)
```

`train.finetune_train` (or a block of the same shape passed as `cfg`) sets
the optimizer (`adam`, `adamw` or `sgd`; there is no LBFGS here), the
learning rate, the plateau schedule, the step limit and the early stop.
The early stop reads the dev split, or the validation split when
`train.stop_source` is `val`.

With a head and a block like the one below, the pass typically stops on its
own within a couple of thousand steps, a few minutes on a CPU:

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

The usual progression: the best neuron alone gives one test error, a fitted
head on its own can be slightly worse, and the head followed by the
end-to-end pass improves on both.

Things to know before using it:

- **Use a small learning rate.** Every weight moves at once, so the pass
  wants a far gentler step than the head fit: AdamW at around `1e-4` to
  `1e-5` is a good starting range.
- **Check that it converged.** The log prints `finetune early stop at step
  ...` when the pass stops on its own. Without that line, it ran to
  `max_steps` and its result is cut short, which is worse and more variable;
  raise `max_steps` until the early-stop line appears on every run.
- **Keep the head.** Removing the head and fine-tuning the bare network
  against its best neuron tends to do worse than no fine-tuning at all,
  because a single readout neuron is a poor target. With the head,
  fine-tuning reliably helps.
- **It repeats only with the same calls.** Seeded, the pass gives the same
  result on a CPU run after run. Every pass over a loader draws from the
  random generator, so extra loops over a loader, to score the head along
  the way for instance, change the pass's shuffling and its result. On a
  GPU, set `train.use_deterministic_algorithms` (see
  [Training](../guides/training.md#seeds)).
- **It widens the spread.** Fine-tuned models vary more from run to run than
  the plain search.
- **It saves the model** to the run folder's `model_last.ckpt` when it
  ends, on its last step unless `finetune_train.keep_best_weights` is on
  (see [Last step or best evaluation](#last-step-or-best-evaluation)).
- **Run it before `Trainer.infer`.** On a model with a head, a call to
  `Trainer.infer` before the pass makes the pass fail (see
  [Troubleshooting](../guides/troubleshooting.md#known-problems)).

On a CUDA device with `adam` or `adamw`, the pass runs as a captured CUDA
graph, which is much faster (see [Performance](../guides/performance.md#the-end-to-end-pass-on-a-gpu)).

The three passes together, the per-layer fine-tune, the head and the
end-to-end pass, are what the tutorials' `*_finetune` configs add. On some
tasks the gain from the full stack is larger than the difference between the
cheapest and the most expensive search.

### In the tutorial scripts

The tutorial scripts read three top-level flags: `finetune_end_to_end`
runs the pass after the head fit, `finetune_drop_head` removes the head
first, and `finetune_prune_first` prunes the network first. Pruning first
leaves the accuracy unchanged, because the neurons it removes get no
gradient anyway.

## Last step or best evaluation

The head fit, the per-layer fine-tune and the end-to-end pass evaluate
every `eval_interval` steps, and a last step that falls between two
evaluations is evaluated too. By default each pass ends on its last step.
`keep_best_weights: true` in `train.out_proj_train` (the head fit and the
per-layer fine-tune) or `train.finetune_train` (the end-to-end pass) makes
it end on the weights of its lowest evaluated loss instead: the pass copies
the weights it trains on every new best and copies them back when it
stops. `best_weights_copy` says where the copy is kept: `device`, next to
the parameters, in GPU memory on CUDA; `cpu`, in host memory; or `disk`, in
a file in the run folder that is deleted when the pass ends. The copy is as
large as what the pass trains: the head, a layer's survivors, or the whole
model.

Neither setting is better everywhere. The last step's weights sometimes win
on the test split even when the best evaluation's are lower on dev, and for
the per-layer fine-tune the best weights can leave lower layer errors and
let the search train more layers. Measure both on your data.

With `keep_best_weights` on, the log says which step each pass kept:

```text
finetune: kept the weights of step 875 (dev loss 0.2704); the last evaluation, step 1375, gave 0.2715
```

## Pruning

`Trainer.prune(model)` deletes every neuron the prediction does not reach.
It keeps the neurons the readout reads in the last layer: the
`out_proj.in_features` the head reads, every survivor of a multi-class
model with a projection per candidate and no head, and one neuron
otherwise. Then it works down the layers, keeping whatever those neurons
read. It follows any input layout, including inputs from older layers
under `model.shortcut.prev_layers`, and deletes a layer that nothing reads
any more. The predictions stay the same.

A model built with `model.use_layer_norm` cannot be pruned: each of its
LayerNorms normalizes a row over every column of a layer's input, so
removing columns would change the predictions. `prune` raises `ValueError`
and leaves such a model as it is.

Without a head, pruning typically leaves only a handful of neurons: the
model collapses toward the single chain the best neuron depends on, and
unused features drop out. With a head that reads every survivor, far more
neurons survive, since the prediction reaches back into every layer.

## The knobs

`model.use_output_projection`, `model.num_out_neurons`,
`train.layer_finetune`, `train.layer_err_source`, `train.out_proj_train`,
`train.finetune_train`, `train.stop_source`. See
[Head fit and per-layer fine-tune](../reference/config.md#head-fit-and-per-layer-fine-tune)
and [End-to-end pass](../reference/config.md#end-to-end-pass).

<small>Checked against TorchSONN 0.1.5.</small>
