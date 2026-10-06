# Data

TorchSONN trains from PyTorch data loaders. This page covers the dataset
wrapper, the splits a run needs, how to prepare features, and the options
for unusual batch formats and imbalanced classes.

## `SONNDataset`

`SONNDataset(x, target)` pairs a feature matrix with its targets; any
PyTorch `DataLoader` works on top of it.

```python
from torch.utils.data import DataLoader
from torchsonn import SONNDataset

train_dl = DataLoader(SONNDataset(x_train, y_train), batch_size=16384, shuffle=True)
```

- `x` is an array or tensor with one row per sample and one column per
  feature. Float32 and float64 both work.
- `target` holds one value per row: a number for regression, the class
  index 0, 1, ... for `multi-class` (integers; float labels fail inside the
  loss), and 0 or 1 for `binary`. A loader cannot batch missing targets;
  to predict rows that have none, call `model.infer(x)` on a tensor of
  features, or give the dataset placeholder targets.
- `split=0` or `split=1` keeps only the even or the odd rows. The trainer
  uses this for the minimum-bias criterion; you rarely need it.

## The splits

A run reads three loaders, plus an optional fourth:

| Loader | Passed to | Used for |
|---|---|---|
| train | `train(train_dl, ...)` | fitting every coefficient |
| dev | `train(..., dev_dl, ...)` | scoring and selecting candidates, the depth, the early stops |
| test | `train(..., test_dl)` | nothing during training; your final measurement with `infer` |
| validation | `train(..., val_dl=...)` | reported per layer; optional (see [Splits and stopping](../concepts/splits-and-stopping.md#the-validation-split)) |

The dev split decides a lot, so its rows must not appear in the test split.
The quickstarts use 50% / 25% / 25%, and holding out around 20% for test and
splitting the rest 3:1 into train and dev is a reasonable default. Shuffle
the rows before cutting when the data is stored in a meaningful order, such
as by location or by time.

## Preparing features

- **Standardize** each feature with the mean and standard deviation of the
  training rows only, then apply the same transform to every split.
- **Tame heavy tails.** A few rows far outside the bulk get extreme
  predictions from a polynomial network and can dominate the test error.
  Clip standardized features (the quickstarts clip at ±5), or log-transform
  skewed ones. Between layers, `model.output_clamp_value` (1000 by default)
  bounds every layer's outputs.
- The orthogonal families squash their inputs themselves (see
  [Orthogonal polynomials](../concepts/neurons/orthogonal.md#the-squash)),
  and the RBF family standardizes its own; the power-basis families use the
  inputs as they come.

## Batch size

The trainer takes batches from the loaders as they come; `train.batch_size`
is read only by the tutorial scripts, which build their loaders from it.
For the LBFGS optimizer, use one batch per split: a batch size larger than
the number of training rows. LBFGS builds its curvature estimate from
successive gradients, which must come from the same rows. Shuffle the
training loader; the order of the dev and test loaders does not matter.

## Deterministic splits

`torchsonn.data.preprocessing` has helpers for splitting data by row
position, as GMDH implementations traditionally do:

```python
from torchsonn.data.preprocessing import SequenceTypeSet, split_dataset, train_preprocessing

x, y = train_preprocessing(x, y, feature_names)   # arrays, shape checks
x_train, y_train, x_dev, y_dev = split_dataset(x, y, SequenceTypeSet.sqMode4_1)
```

`train_preprocessing` turns pandas objects into arrays, checks that `x` is
2-D with at least two features and two rows, transposes it if its rows and
columns are swapped, and checks `feature_names` against the number of
columns. `split_dataset` assigns each row to train or to dev (called
"validate" there):

| `SequenceTypeSet` | Rows sent to dev |
|---|---|
| `sqRandom` | each row at random, with probability 1/2 |
| `sqMode1`, `sqMode3_1`, `sqMode4_1` | every 2nd, 3rd or 4th row |
| `sqMode2`, `sqMode3_2`, `sqMode4_2` | all but every 2nd, 3rd or 4th row |

`sqMode4_1`, for example, gives a 3:1 train/dev split.

## Custom batch formats

If your loader yields something other than `(x, y)` pairs, pass a function
that turns a batch into one:

```python
trainer = Trainer(config, batch_callback=lambda batch: (batch["features"], batch["label"]))
```

The trainer applies it wherever it reads a batch during training. `infer`
does not: give it a loader that yields `(x, y)` pairs.

## Class weights

For imbalanced classes, weight the loss per class:

```python
import torch
trainer = Trainer(config, class_weights=torch.tensor([1.0, 4.0, 2.0]))
```

For `multi-class`, the tensor has one weight per class. For `binary`, it
has one element, the weight of the positive class.

## Loaders for the end-to-end pass

On a CUDA device the end-to-end pass copies each batch into a fixed buffer
(see [Performance](performance.md)). With `train.finetune_train.data_on_device:
false` the batches stream from the loader; build that loader with
`pin_memory=True`, because pinning a batch inside each step costs more than
the step itself (about 35 ms per 12,000 rows, comment on
`OutProjTrainConfig.data_on_device`).

<small>Checked against TorchSONN 0.1.5.</small>
