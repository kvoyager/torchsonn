# Troubleshooting

Errors first, by the message they print, then results that look wrong,
then the known problems of the library as it stands.

## Errors

### `Error creating layer. No functions were created`

```text
Layer #0 has 4 input(s), fewer than the 5 that Legendre3x5 requires; skipping this neuron family for the layer.
torchsonn.types.LayerCreationError: Error creating layer. No functions were created
```

Every family of the layer needs more inputs than the layer has. A family
that is too wide sits the layer out with the warning above; when no family
is left, the layer cannot be built. Add a family that fits, such as a pair
family, or lower the wide family's `dim`. A family that sits out the first
layer joins later ones once the shortcut gives them enough inputs.

A `polyquad` family with a `dim` above 2 and no `model.max_neuron_models`
fails earlier, with a bare `NotImplementedError`: set the cap.

### Readout and validation preconditions

```text
ValueError: train.layer_err_source='readout' requires model.use_output_projection=true: ...
NotImplementedError: train.layer_err_source='readout' is implemented for regressor / binary models only.
ValueError: train.stop_source='val' needs a validation loader (val_dl)
```

The first two come from the `Trainer` constructor. `readout` scores a layer
by a head over its survivors, so it needs `model.use_output_projection:
true`, and it works for regression models only (binary models fail later,
see [Known problems](#known-problems)). The third comes from `train` and
`train_finetune`: pass `val_dl=` or set `train.stop_source: dev`.

### Building the model fails

```text
AssertionError: soft_binner and use_neuron_proj are mutually exclusive
```

`model.use_neuron_proj: true` needs `model.soft_binner: false` next to it.

### Building the trainer fails

```text
ValueError: train.checkpoint_dir is empty; set the folder that holds the run folders (the default is 'checkpoints', in the working directory).
```

Set `train.checkpoint_dir` to a folder, or leave the key out of the
configuration for the default, `checkpoints` in the working directory.

### `gather(): Expected dtype int32/int64 for index`

A multi-class model was given float labels. Pass the class indices as
integers (see [Data](data.md#sonndataset)).

### `Graphviz 'dot' executable not found in PATH`

`PlotModel.plot()` needs the Graphviz programs, not only the `graphviz`
Python package (see [Installation](../getting-started/install.md)).

### The end-to-end pass does not use the CUDA graph

```text
end-to-end pass: CUDA graph capture failed (...); running eager steps
```

Before capturing, the pass runs one training step in a mode that flags any
point where the GPU waits for the CPU, because such a step cannot be
captured. If it finds one, the pass runs uncaptured and logs the reason
above. The result is the same, only slower.

```text
RuntimeError: end-to-end pass: CUDA graph capture failed after a clean rehearsal (...); the CUDA context is now unusable for this process. Re-run with finetune_train.cuda_graph=false.
```

A capture that fails after a clean rehearsal leaves the GPU unusable for
the rest of the process. Run again with
`train.finetune_train.cuda_graph: false`.

## Results that look wrong

### The test error is far above the dev error

A few test rows far outside the range of the training rows can dominate the
test error, because a polynomial extrapolates wildly there. Without
clipping, a handful of extreme standardized rows can carry essentially all
of the test error, so the figure swings wildly from run to run while the dev
error stays low. Clip the standardized features, as the quickstarts do at
±5, or log-transform skewed ones (see [Data](data.md#preparing-features)).

### Layer outputs hit the clamp

Every layer's outputs are cut to ±`model.output_clamp_value` (1000) before
the next layer reads them, and the fine-tune passes get no gradient through
a cut output. On heavy-tailed features that are not clipped, a pair
neuron's outputs can pass 1000 on the outliers, since the product of two
large standardized values grows fast. Clip the features, or raise the clamp
(some configs use `1.0e+6`).

### Candidates diverge or turn NaN

A candidate whose loss turns NaN stops at once, and selection drops
candidates whose criterion is NaN. A candidate whose evaluated loss exceeds
`train.divergence_threshold` also stops; the default, infinity, turns that
check off, because on heavy-tailed data a sound candidate can start with a
loss in the thousands. Candidates that diverge usually point to unscaled
features or a learning rate too high for `adam` or `sgd`: standardize and
clip the features, and keep gradient clipping on (see
[Optimizers](optimizers.md#gradient-clipping)).

### RBF centres leave the data

With `center_radius: null` and the LBFGS safeguards off, many of the
survivors' centres can drift far from the data, well beyond the range the
training rows cover. Keep the default `center_radius` and the LBFGS
`max_step` and `curvature_eps` (see [Gaussian RBF](../concepts/neurons/rbf.md)
and [Optimizers](optimizers.md#lbfgs)).

### A model with the per-layer fine-tune predicts badly

Without an output head, a model trained with `train.layer_finetune`
predicts with one survivor, which the fine-tune has made worse on its own:
on CCPP, a mean absolute error of about 13 MW instead of about 3.3. Add the
head (see [Heads and fine-tuning](../concepts/heads-and-finetune.md#the-per-layer-fine-tune)).

### The end-to-end pass made the model worse

- Its log has no `finetune early stop` line: the pass ran to `max_steps`
  and stopped short. Raise `max_steps` or the learning rate.
- The head was removed before the pass: keep it.
- The learning rate is too high: every weight moves at once, so the pass
  needs a much smaller rate than the head fit.

### The search stops after a few layers

Look at the growth-rule lines in the log (see
[Splits and stopping](../concepts/splits-and-stopping.md)). With
`train.layer_finetune`, the survivors' own errors get worse and the search
stops early by design. A small dev split makes the layer errors noisy, so a
chance rise can end the search: `train.criterion_minimum_width` allows
more layers without improvement.

### A resumed run continued the wrong run

`train(..., resume=True)` continues the newest run folder in
`train.checkpoint_dir` that holds a step checkpoint, whichever
configuration wrote it. Give every configuration its own
`train.checkpoint_dir`, or name the run: `resume="2026-10-04-10-15-30"`.

## Known problems

These are problems of the library as it stands, with a way around each.

| Problem | Way around |
|---|---|
| A `binary` model's criterion compares logits with the 0/1 labels, so selection picks poor neurons; on a synthetic task the model scored chance-level accuracy | Train a `regressor` on the 0/1 labels and threshold its prediction at 0.5 (see [Classification](../concepts/classification.md#binary-models)). |
| A binary model with `train.layer_finetune` or `train.layer_err_source: readout` fails with `ValueError: Target size ... must be the same as input size` | Use neither on binary models; they have no head. |
| On a model with a head, a `Trainer.infer` call before `Trainer.train_finetune` makes the pass fail with `RuntimeError: Inference tensors cannot be saved for backward` | Run the pass first. To measure the model before it, call `model.infer(x)` inside `torch.no_grad()`. |
| `Trainer.infer` does not apply the trainer's `batch_callback` | Give `infer` a loader that yields `(x, y)` pairs. |
| Resuming a run that finished trains one more layer, even past `train.max_layer_count`; resuming it again can fail to create a layer | Resume only interrupted runs (see [Training](training.md#resuming)). |

<small>Checked against TorchSONN 0.1.5.</small>
