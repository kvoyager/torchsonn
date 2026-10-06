# Performance

Almost all of a run's time goes into the candidate fits: every layer fits
every candidate of every family, evaluates them on the dev split as it
goes, and runs the frozen layers below to compute their inputs. This page
covers the settings that decide that cost, the GPU path of the end-to-end
pass, and a few smaller settings.

## What a run costs

| Run | Device | Wall-clock |
|---|---|---|
| [Regression quickstart](../getting-started/quickstart-regression.md): 28 candidates per layer, 10 layers | CPU | about 2 min |
| The regression quickstart's end-to-end pass: 1,400 steps without the CUDA graph | CPU | about 2.5 min |
| California housing, Legendre with head and end-to-end pass | GPU | about 40 to 45 s |
| CCPP `ccpp_legendre.yaml`: 8 survivors, 60 candidates per family, 10 folds | CPU | about 12 min |
| CCPP `ccpp.yaml`: three families, same pools | CPU | about 19 min |
| CCPP `ccpp_legendre_heavy.yaml`: 60 survivors, 600 candidates | GPU | about 1.4 h |
| CCPP `ccpp_heavy.yaml`: six families, 120 survivors, 1,200 candidates | GPU | about 11.3 h |

The CCPP times cover all ten folds of its cross-validation, on a Core
i9-12900 CPU or an RTX 5080 GPU (CCPP README). The heavy CCPP search costs
about 36 times as much as `ccpp.yaml` and lowers the mean absolute error by
only 0.08 MW.

## The number of candidates

`model.max_neuron_models` is the main lever. Each family makes one
candidate per pair or tuple of the layer's inputs, up to that cap, and a
layer's time grows roughly with its number of candidates. The inputs are
the previous layer's survivors plus the features
(`model.shortcut.raw_features`), plus the survivors of older layers under
`model.shortcut.prev_layers`. The quickstart's layers after the first have
8 survivors and 8 features, 16 inputs and 120 pairs, of which the cap of 28
tries a random subset.

What the cap costs on California housing (California housing configs):

| Change | Layer time | Test MSE |
|---|---|---|
| pair cap 140 instead of 70 | about 1.6 times | 0.2605 against 0.2630, within the seed noise |
| a second family of 3-input Legendre neurons, cap 140 | about 1.7 times | 0.1945, against 0.1972 to 0.1986 for pairs alone |
| the same family with all 680 triplets, cap 700 | about 7 times | 0.1944 |
| 32 survivors with a pair cap of 300 | 104 s per layer on a GPU | 0.1904 |

More survivors cost twice: the next layer has more inputs, so more tuples
to sample from, and `prev_layers` adds the survivors of every older layer
it feeds forward. A family with a larger `dim` has far more tuples: 16
inputs make 120 pairs but 4,368 tuples of 5.

## The candidate fit

- **Batch size.** LBFGS wants one batch per split (see
  [Data](data.md#batch-size)), which also makes each step one large,
  efficient operation.
- **Evaluation interval.** Every `train.eval_step_interval` steps, every
  candidate is evaluated on the whole dev split. The quickstarts evaluate
  every 5 steps, because their LBFGS fits stop after 80 to 420 steps; the
  default of 1000 suits long first-order fits. The early stop can only fire
  at an evaluation.
- **`train.early_stop_completion_percentage`** ends a family's fit once
  that percentage of its candidates has stopped, instead of waiting for the
  slowest.
- **`train.precompute_features: true`** runs the frozen layers over each
  split once per layer and keeps the result in memory, instead of running
  them on every step. That saves a forward pass through the frozen layers
  per step, at the cost of a copy of every split's layer inputs: rows times
  input width numbers per split, which grows with depth under `shortcut`.
  No measurements of either are recorded.
- **The input pass.** A layer with an orthogonal or RBF family first runs
  the frozen layers once over the train split to collect its input
  statistics and, for RBF, a sample for k-means; streaming k-means adds
  `train.rbf_kmeans_passes` passes when the split is larger than
  `train.input_sample_rows`. Layers without such a family skip the pass.

The candidate fits suit a GPU: every family's candidates train as one
batched tensor operation. The fit can also be split across several GPUs
(see [Training](training.md#several-gpus)).

## The end-to-end pass on a GPU

`Trainer.train_finetune` trains a network of many small layers, so each
step launches hundreds of tiny kernels and spends most of its time
launching them. On a CUDA device with `adam` or `adamw` and
`train.finetune_train.cuda_graph: true` (the default), the pass records
the whole training step once as a CUDA graph and replays it for every
batch. A batch of another shape, such as the last batch of an epoch, runs
the same step without the graph, so no row is lost. On California housing
the captured pass takes 8 to 11 s and gives the same test numbers as the
uncaptured pass to the fourth decimal (changelog).

`train.finetune_train.data_on_device` decides where the training split
lives during the pass: on the GPU (`true`), streamed from the loader
(`false`), or on the GPU when it fits in a quarter of the free memory
(`auto`, the default). Streamed batches should come from a loader built
with `pin_memory=True` (see [Data](data.md#loaders-for-the-end-to-end-pass)).

The log says which path the pass took:

```text
end-to-end pass: CUDA graph (captured on the first batch); batches ...; batch size ...
end-to-end pass: CUDA graph captured (... parameters, batch ...)
end-to-end pass: ... steps replayed, ... eager (warm-up and tail batches)
```

On a CPU, with `sgd` or with `cuda_graph: false`, the pass runs its steps
one by one and logs `end-to-end pass: eager steps`. A step that cannot be
captured also falls back to that path (see
[Troubleshooting](troubleshooting.md#the-end-to-end-pass-does-not-use-the-cuda-graph)).

## Smaller settings

- **Progress bars.** Each refresh of a candidate fit's progress bar reads
  several numbers back from the GPU, which makes the GPU wait.
  `train.eval_display_interval` refreshes it every that many evaluations
  instead of every one, and `train.optimizer.verbose: false` hides it.
  Neither changes the results.
- **Prediction.** `Trainer.infer(model, dl, use_compile=True)` compiles the
  forward pass with `torch.compile` first. The first batch takes longer;
  the rest run faster, which pays off when the loader has many batches.

## The knobs

`model.max_neuron_models`, `model.nbest_neurons`, `model.shortcut`,
`train.eval_step_interval`, `train.early_stop_completion_percentage`,
`train.precompute_features`, `train.input_sample_rows`,
`train.rbf_kmeans_passes`, `train.finetune_train.cuda_graph`,
`train.finetune_train.data_on_device`, `train.eval_display_interval`,
`train.optimizer.verbose`, `train.device`. See
[Configuration keys](../reference/config.md).

<small>Checked against TorchSONN 0.1.5.</small>
