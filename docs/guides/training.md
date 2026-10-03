# Training

`Trainer` runs everything that learns: the layer-by-layer search, the
optional head fit and end-to-end pass, pruning and prediction. This page
covers its calls in the order a run makes them, then checkpoints, devices
and distributed runs.

## The trainer

```python
from torchsonn import SONN, Trainer

model = SONN(config, d_model=n_features, feature_names=names)
trainer = Trainer(config, feature_names=names)
```

`Trainer(config, batch_callback=None, feature_names=None, class_weights=None)`
takes the same configuration as the model. `batch_callback` and
`class_weights` are described under [Data](data.md); `feature_names` names
the inputs in logs and model descriptions.

## Seeds

```python
trainer.set_seed(config.train.seed)
```

`Trainer.set_seed` seeds Python's `random`, NumPy and PyTorch on the CPU and
every GPU, and makes cuDNN deterministic. It fixes the sampled candidate
tuples and the initial weights, so a run repeats exactly on the same
machine; the quickstarts give identical numbers run after run. Runs on CUDA
are not bit-for-bit reproducible, and small differences can move where the
stop rules fire (see [Splits and stopping](../concepts/splits-and-stopping.md#choosing-the-margin)).
`train.seed` also seeds the input pass's sampling on its own.

## The search

```python
trainer.train(model, train_dl, dev_dl, test_dl)
```

`train(model, train_dl, dev_dl, test_dl, verbose=True, resume=False,
val_dl=None)` grows the network until the growth rule stops it, keeps the
layers up to the best one, saves the model and returns it (the same
object). It discards any layers the model already had. `test_dl` is not
used during training. `verbose` logs the time each layer takes; `val_dl`
adds a [validation split](../concepts/splits-and-stopping.md#the-validation-split).
[The algorithm](../concepts/algorithm.md) describes what happens inside.

## After the search

| Call | Does |
|---|---|
| `trainer.train_out_proj(model, train_dl, dev_dl)` | Fits the linear head of a model built with `model.use_output_projection`, then saves the model. Needed before `infer` for such a model. |
| `trainer.train_finetune(model, train_dl, dev_dl, cfg=None, val_dl=None)` | Trains every parameter at once, with the settings of `train.finetune_train` (or `cfg`). Does not save the model. |
| `trainer.prune(model)` | Deletes every neuron that does not reach the output; the predictions stay the same. |
| `trainer.infer(model, dl, verbose=True, use_compile=False)` | Predicts over a loader and returns `(predictions, targets)`, both in loader order. |

`infer` returns one prediction per row for regression, a logit per row for
`binary`, and a row of log-probabilities per sample for `multi-class`.
`use_compile=True` compiles the forward pass with `torch.compile` first,
which pays off when `infer` runs over many batches. To predict a tensor of
features without a loader, call `model.infer(x)`.
[Heads and fine-tuning](../concepts/heads-and-finetune.md) explains the
head and the end-to-end pass.

## Checkpoints

Training writes into `train.checkpoint_dir`. Set it: left empty, it points
at a folder three levels above the library's `trainer.py`, which for an
installed package is inside the Python environment.

| File | Written |
|---|---|
| `model_last.ckpt` | by `train` at the end, and by `train_out_proj`: the finished model |
| `model_layer_<L>_neuron_<N>_step_<S>.ckpt` | during each family's fit, every `train.save_interval` steps and when the fit ends: the training state, for resuming |
| `model_layer_<L>_neuron_<N>_step_<S>_last.ckpt` | when a family's fit ends, with `train.save_last_layer` |

`L` is the layer, `N` the family within it and `S` the step. Only the
`train.keep_last_n` most recent step checkpoints are kept; that cleanup
leaves the `_last` copies and `model_last.ckpt` alone. When the search ends,
the step checkpoints of the layers trained past the best one, `_last`
copies included, are deleted.

To use a trained model later, build a model from the same configuration
and load the saved one into it:

```python
model = SONN(config, d_model=n_features, feature_names=names)
trainer.load_model_checkpoint(model, "cpu")
```

`load_model_checkpoint` rebuilds the saved layers and moves the model to
the given device. `train_finetune` leaves its result in memory only; call
`trainer.save_model_checkpoint(model)` to keep it, which overwrites
`model_last.ckpt`.

## Resuming

`train(..., resume=True)` continues an interrupted search from the latest
step checkpoint in the folder: it restores the layers trained so far and
picks up the family and step where training stopped. Without a step
checkpoint it starts from scratch. Use a separate `train.checkpoint_dir` for
every configuration, so that a resumed run never picks up another
configuration's checkpoints.

## Devices and precision

`train.device` decides where the neurons live and compute: `cpu`, `cuda`,
`cuda:1`. Batches are moved to it as they are read, so the loaders can stay
on the CPU. The candidate fits and the end-to-end pass gain most from a GPU;
on CUDA the end-to-end pass runs as a captured graph (see
[Performance](performance.md)).

Neuron parameters use PyTorch's default type, float32. `train.dtype` only
sets the type of class weights and of the input statistics the squash and
RBF neurons use.

## Several GPUs

The candidate fit can be split across processes with `torch.distributed`:
every process fits a slice of each family's candidates. Call
`Trainer.init_distributed()` (backend `nccl` by default) once before
`train`, and launch the script with `torchrun`:

```bash
torchrun --nproc_per_node=4 my_script.py
```

Each process uses the GPU of its `LOCAL_RANK`. A family is split only when
its number of candidates divides by the number of processes; otherwise
that family is fitted whole on every process. Only the candidate fit is
split; selection and the other stages run on every process.

!!! note
    The split candidate fit is not exercised by the test suite or the
    tutorials; only its helper functions are tested, against a simulated
    process group.

<small>Checked against TorchSONN 0.1.5.</small>
