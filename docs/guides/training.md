# Training

`Trainer` runs everything that learns: the layer-by-layer search, the
optional head fit and end-to-end pass, pruning and prediction. This page
covers its calls in the order a run makes them, then the run folders and
their checkpoints, resuming, devices and distributed runs.

## The trainer

```python
from torchsonn import SONN, Trainer

Trainer.set_seed(config.train.seed)
model = SONN(config, d_model=n_features, feature_names=names)
trainer = Trainer(config, feature_names=names)
```

`Trainer(config, batch_callback=None, feature_names=None, class_weights=None)`
takes the same configuration as the model. `batch_callback` and
`class_weights` are described under [Data](data.md); `feature_names` names
the inputs in logs and model descriptions.

## Seeds

`Trainer.set_seed(seed)` seeds Python's `random`, NumPy and PyTorch on the
CPU and every GPU, and makes cuDNN deterministic. Call it before building
the model, as above: it fixes the sampled candidate tuples, the candidates'
initial weights and the shuffling. The output head takes its starting
weights from `train.seed` whenever the model is built, and `train.seed`
also seeds the input pass's sampling on its own.

On a CPU a seeded run then repeats exactly, the end-to-end pass included;
the quickstarts give identical numbers run after run. It takes the same
calls in the same order: every pass over a `DataLoader` draws a number from
the random generator, even when the loader does not shuffle, so an extra
loop over a loader changes the shuffling of everything after it.

On CUDA some kernels give slightly different results from run to run, and
small differences can move where the stop rules fire (see
[Splits and stopping](../concepts/splits-and-stopping.md#choosing-the-margin)).
`train.use_deterministic_algorithms: true` switches PyTorch to deterministic
kernels for the whole process when the `Trainer` is built. Two runs of the
California housing fine-tune config on a GPU then saved identical models
(without it, two different ones), at about 20% more time. An operation
that has no deterministic kernel raises an error naming it.

## The search

```python
trainer.train(model, train_dl, dev_dl, test_dl)
```

`train(model, train_dl, dev_dl, test_dl, verbose=True, resume=False,
val_dl=None)` creates the run's folder, grows the network until the growth
rule stops it, keeps the layers up to the best one, saves the model there
and returns it (the same object). It discards any layers the model already had. `test_dl` is not
used during training. `verbose` logs the time each layer takes; `val_dl`
adds a [validation split](../concepts/splits-and-stopping.md#the-validation-split).
[The algorithm](../concepts/algorithm.md) describes what happens inside.

## After the search

| Call | Does |
|---|---|
| `trainer.train_out_proj(model, train_dl, dev_dl)` | Fits the linear head of a model built with `model.use_output_projection`, then saves the model to the run folder. Needed before `infer` for such a model. |
| `trainer.train_finetune(model, train_dl, dev_dl, cfg=None, val_dl=None)` | Trains every parameter at once, with the settings of `train.finetune_train` (or `cfg`), then saves the model to the run folder. |
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

Every `train` call is a run with a folder of its own, inside
`train.checkpoint_dir`:

```text
checkpoints/california/            train.checkpoint_dir
    2026-10-04-10-15-30/           one run
        train.log
        model_last.ckpt
        model_layer_0_neuron_0_step_80.ckpt
        ...
    2026-10-04-11-02-07/           the next run
```

The folder is named after the time the run started,
`YYYY-MM-DD-HH-MM-SS`, with `-2`, `-3`, ... when two runs start in the same
second. `trainer.run_dir` is the current run's folder and
`trainer.checkpoint_root` the folder above it. `train.checkpoint_dir` is
`checkpoints` by default. A relative path is taken from the working
directory, so the runs go to `checkpoints/` in the folder the script is
launched from.

| File | Written |
|---|---|
| `train.log` | from the start of the run: its log (see below) |
| `model_last.ckpt` | by `train` at the end, and by `train_out_proj`, `train_finetune` and `save_model_checkpoint`: the finished model |
| `model_layer_<L>_neuron_<N>_step_<S>.ckpt` | during each family's fit, every `train.save_interval` steps and when the fit ends: the training state, for resuming |
| `model_layer_<L>_neuron_<N>_step_<S>_last.ckpt` | when a family's fit ends, with `train.save_last_layer` |
| `best_<pass>.ckpt` | during a head fit or fine-tune pass with `keep_best_weights` and `best_weights_copy: disk`: the best weights so far; deleted when the pass ends |

`L` is the layer, `N` the family within it and `S` the step. Only the
`train.keep_last_n` most recent step checkpoints are kept. That cleanup
counts and deletes nothing else, so `train.log`, the `_last` copies,
`model_last.ckpt` and any file of your own stay. When the search ends, the
step checkpoints of the layers trained past the best one, `_last` copies
included, are deleted.

### The run's log

When a run starts, the trainer adds a handler for the run folder's
`train.log` to Python's root logger, so the file gets every INFO record
logged during the run, your script's as well as the library's. It stays
attached after `train` returns, so the head fit, the end-to-end pass and
your own report land in the same file, until another run starts; only one
run's log is attached at a time. If the root logger's level is above INFO,
the trainer lowers it to INFO.

`torchsonn.logger.setup_logger()` shows the log on the console.
`setup_logger(path)` also writes a file of your own, for the lines outside
the runs.

### Loading a saved model

To use a trained model later, build a model from the same configuration
and load the saved one into it:

```python
model = SONN(config, d_model=n_features, feature_names=names)
trainer.load_model_checkpoint(model, "cpu")
```

`load_model_checkpoint(model, device="cpu", run=None)` reads
`model_last.ckpt` from the trainer's current run. A trainer with no run
yet, such as one created in a new script, reads the newest run folder that
holds one, and `run="2026-10-04-10-15-30"` names a run, by its folder name
or its path. It rebuilds the saved layers, moves the model to the device
and makes that run the trainer's current run. `train_out_proj` and
`train_finetune` save to the same file, so it holds the model as the last
of them left it.

## Resuming

`train(..., resume=True)` continues an interrupted run in its own folder:
the newest run folder in `train.checkpoint_dir` that holds a step
checkpoint. It restores the layers trained so far, picks up the family and
step where training stopped, and appends to the run's `train.log`.
`resume="2026-10-04-10-15-30"` continues that run. When there is nothing
to resume, the trainer logs so and starts a new run. Use a separate
`train.checkpoint_dir` for every configuration, so that a resumed run never
continues another configuration's run.

!!! warning "Resume only an interrupted run"
    Resuming a run that finished trains one more layer, even past
    `train.max_layer_count`, and resuming it a second time can fail to
    create a layer.

## Devices and precision

`train.device` decides where the neurons live and compute: `cpu`, `cuda`,
`cuda:1`. Batches are moved to it as they are read, so the loaders can stay
on the CPU. The candidate fits and the end-to-end pass gain most from a GPU;
on CUDA the end-to-end pass runs as a captured graph (see
[Performance](performance.md)).

The neurons, the head, the class weights and the input statistics the
squash and RBF neurons use are all in PyTorch's default type, float32.

## Several GPUs

The candidate fit can be split across processes with `torch.distributed`:
every process fits a slice of each family's candidates. Call
`Trainer.init_distributed()` (backend `nccl` by default) once before
`train`, and launch the script with `torchrun`:

```bash
torchrun --nproc_per_node=4 my_script.py
```

Each process uses the GPU of its `LOCAL_RANK`. Rank 0 creates the run
folder and writes `train.log` and the step checkpoints into it; every
process uses that same folder. A family is split only when
its number of candidates divides by the number of processes; otherwise
that family is fitted whole on every process. Only the candidate fit is
split; selection and the other stages run on every process.

!!! note
    The split candidate fit is not exercised by the test suite or the
    tutorials; only its helper functions are tested, against a simulated
    process group.

<small>Checked against TorchSONN 0.1.5.</small>
