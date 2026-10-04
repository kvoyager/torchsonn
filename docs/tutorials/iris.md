# Iris

Multi-class classification of the 150 iris flowers into their three
species, from four measurements. The tutorial is the smallest of the five:
it trains in seconds on a CPU and ends with a network of one neuron. The
[classification quickstart](../getting-started/quickstart-classification.md)
builds a similar model step by step.

```bash
python -m tutorials.iris.iris_recognition
```

The script is `tutorials/iris/iris_recognition.py`, its configuration
`tutorials/iris/iris.yaml`. A notebook version,
`tutorials/iris/iris_recognition.ipynb`, draws the plots inline. The
script needs the `viz` extra: it plots the layer errors and the confusion
matrix with matplotlib and draws the network with Graphviz.

## Data

The script loads iris from scikit-learn and interleaves the three species,
so that each run of three rows holds one flower of each. The first 70% of
the rows train the neurons (105 flowers), the next 15% are the dev split
(22) and the last 15% the test split (23). The features stay in
centimetres, unscaled.

## The configuration

| Setting | Value | Why |
|---|---|---|
| `model.type` | `multi-class` | three species |
| `model.soft_binner` | false | the class scores come from a learned projection shared by the candidates (see [Classification](../concepts/classification.md#the-shared-projection)) |
| `model.ref_functions` | `linear_cov` | pair neurons $w_0 + w_1 x_i + w_2 x_j + w_3 x_i x_j$ |
| `model.shortcut` | false | each layer reads only the survivors of the layer before |
| `model.nbest_neurons` | 4 | survivors per layer |
| `model.max_neuron_models` | 8 | above the 6 pairs that 4 inputs make, so every pair is tried; the log warns that the cap is clamped to 6 |
| `train.optimizer` | `sgd`, `lr` 1e-4, `min_lr` 1e-5 | with batches of 30 rows |
| `train.steps`, `train.eval_step_interval` | 900, 5 | |
| `train.max_layer_count` | 5 | |
| `train.criterion_minimum_width` | 5 | the search runs all five layers and keeps the best |

`train.checkpoint_dir` is empty in this file, so the checkpoints go to the
default folder (see [Training](../guides/training.md#checkpoints)). Set it
on the command line to keep them elsewhere:

```bash
python -m tutorials.iris.iris_recognition train.checkpoint_dir=checkpoints/iris
```

The commented-out entries of `model.ref_functions` in the file show the
other power-basis families; uncommenting one adds it to the search.

## Results

Two runs on a CPU gave the same numbers, about 11 seconds each. The dev
error of each layer:

| Layer | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Dev error | 0.0789 | 0.2466 | 0.4239 | 0.2853 | 9.7661 |

The first layer is the best, so the model keeps one layer. It classifies
all 23 test flowers correctly:

```text
Confusion matrix, without normalization
[[7 0 0]
 [0 8 0]
 [0 0 8]]
```

With 23 test rows, that shows the model works on this data; it is too few
rows to measure its accuracy precisely.

## The network

The script draws the network before and after pruning, into
`iris_model.svg` and `iris_pruned_model.svg` next to the script,
overwriting the copies in the repository. The four survivors of the one
layer read different pairs of measurements; after pruning, the prediction
comes from a single neuron that reads sepal length and petal length:

![The pruned iris network: one LinearCov neuron reading sepal length and petal length](../assets/tutorials/iris/iris_pruned_model.svg)

The script prints the selected features before it prunes, so it lists all
four measurements. [Inspecting a model](../guides/inspecting.md) shows how
to read the network after pruning.

<small>Checked against TorchSONN 0.1.5.</small>
