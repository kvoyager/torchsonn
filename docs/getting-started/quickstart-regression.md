# Quickstart: regression

This page fits a regression model to the California housing data, one block
at a time. Run the blocks in order, in one script or in a notebook. On a CPU
the whole run takes about two minutes. The dataset downloads through
scikit-learn on first use.

## 1. Load the data and split it

```python
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
x = housing.data.astype(np.float32)
y = housing.target.astype(np.float32)
feature_names = list(housing.feature_names)

# The rows come grouped by location: shuffle, then cut 50% / 25% / 25%.
order = np.random.default_rng(0).permutation(len(x))
x, y = x[order], y[order]
i_train, i_dev = int(0.50 * len(x)), int(0.75 * len(x))
```

California housing has 20,640 rows, one per census block group, with 8
numeric features: median income, house age, average rooms and bedrooms,
population, average occupancy, latitude and longitude. The target is the
median house value in units of $100,000.

The data is cut into three splits, each with its own job:

- **train** fits the coefficients of every candidate neuron;
- **dev** scores the candidates: it picks the neurons that survive each
  layer and decides how many layers to keep;
- **test** stays untouched until the end and measures the finished model.

The dev split is what lets the network choose its own structure. An error
measured on the rows a neuron was fitted to keeps falling as neurons get
more complex, so structure is chosen on rows the fit never saw (see
[GMDH](../concepts/gmdh.md)). The rows are stored grouped by location, so
the code shuffles them before cutting; an ordered cut would test on a
different region than it trains on.

## 2. Standardize and clip the features

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x[:i_train])
x = np.clip(scaler.transform(x), -5.0, 5.0).astype(np.float32)
```

The scaler is fitted on the training rows only, so nothing about the dev
and test rows leaks into the inputs. The clip matters on this dataset: a few
block groups have an extreme population or average occupancy, more than 200
standard deviations from the mean. A polynomial network extrapolates far
outside the range it was fitted on, so without the clip those few rows
dominate the test error.

## 3. Wrap the splits in loaders

```python
from torch.utils.data import DataLoader
from torchsonn import SONNDataset

batch = 16384  # larger than the 10,320 training rows: one batch per split
train_dl = DataLoader(SONNDataset(x[:i_train], y[:i_train]),
                      batch_size=batch, shuffle=True)
dev_dl = DataLoader(SONNDataset(x[i_train:i_dev], y[i_train:i_dev]),
                    batch_size=batch)
test_dl = DataLoader(SONNDataset(x[i_dev:], y[i_dev:]), batch_size=batch)
```

`SONNDataset` pairs the features with the targets, and any PyTorch
`DataLoader` works on top of it. With a batch larger than the training
split, every optimizer step sees all 10,320 training rows. LBFGS, the
optimizer configured below, builds its curvature estimate from successive
gradients and works best when they come from the same rows.

## 4. Configure the model

```python
from omegaconf import OmegaConf
from torchsonn import SONN

config = OmegaConf.merge(SONN.default_config(), {
    "model": {
        "type": "regressor",
        "ref_functions": ["linear_cov"],
        "nbest_neurons": 8,
        "max_neuron_models": 28,
    },
    "train": {
        "max_layer_count": 10,
        "steps": 500,
        "eval_step_interval": 5,
        "early_stop_patience": 1e-4,
        "early_stop_tolerance_steps": 3,
        "checkpoint_dir": "checkpoints/california",
        "optimizer": {
            "name": "lbfgs",
            "optimizer_params": {
                "lr": 0.1, "min_lr": 0.01, "history_size": 10,
            },
        },
    },
})
```

`SONN.default_config()` is the full configuration schema with every
default filled in. The dictionary holds only what this model changes:

- **`type: regressor`**: a model with one numeric output.
- **`ref_functions: [linear_cov]`**: the neuron family. Every candidate
  neuron reads a pair of inputs $x_i, x_j$ and computes
  $w_0 + w_1 x_i + w_2 x_j + w_3 x_i x_j$. See
  [Neuron families](../concepts/neurons/index.md) for the others.
- **`nbest_neurons: 8`**: how many neurons survive each layer.
- **`max_neuron_models: 28`**: the most candidates a layer tries per family.
  Layer 0 reads the 8 features, which make exactly 28 pairs, so it tries
  them all. Every later layer reads the 8 survivors of the layer before
  plus the 8 original features, 16 inputs that make 120 pairs, and tries a
  seeded random 28 of them.
- **`max_layer_count: 10`**: the deepest the network may grow.
- **`steps`, `eval_step_interval`, `early_stop_patience`,
  `early_stop_tolerance_steps`**: how long each candidate's fit runs. Every
  5 steps, the fit evaluates each candidate on the dev split. A drop in the
  smoothed dev loss of less than 1e-4 does not count as an improvement.
  After 3 evaluations without one, the candidate's learning rate halves.
  Once the learning rate is down to `min_lr`, the candidate stops. The step
  limit is checked at the same evaluations, so a fit ends at the first
  evaluation after step 500 at the latest.
- **`checkpoint_dir`**: where training saves its checkpoints and the
  final model, relative to the working directory.
- **`optimizer`**: LBFGS with a step size of 0.1, a floor of 0.01 and the
  last 10 steps kept for its curvature estimate. Each candidate is a small
  least-squares problem, 4 coefficients here, which a quasi-Newton method
  solves in tens to a few hundred steps.

## 5. Train

```python
from torchsonn import Trainer
from torchsonn.logger import setup_logger

setup_logger("train.log")
model = SONN(config, d_model=x.shape[1], feature_names=feature_names)
trainer = Trainer(config, feature_names=feature_names)
trainer.set_seed(config.train.seed)
trainer.train(model, train_dl, dev_dl, test_dl)
```

`setup_logger` sends the training log to the console and to `train.log`.
`set_seed` fixes the random draws (the sampled candidate pairs and the
initial weights), so a rerun on the same machine gives the same model.
`train` builds layers until the growth rule stops it. It takes the test
loader as an argument but does not use it.

For each layer, the log shows the candidate fit and then the verdict of the
growth rule. Without the timestamps, layer 1 reads:

```text
Creating layer #1
Training layer #1
All models of LinearCovPolynomNeuron early stopped at step 80
LinearCovPolynomNeuron fit: 81 optimizer steps; 0.9% of updates capped at max_step, 13.5% of curvature pairs rejected
Current layer error: 0.376
Layer errors: [0.488, 0.376]
Executed train layer #1 in 5.47 sec
Layer #1: error 0.3758, best 0.3758 at layer 1; improved by +0.1121 (margin 0.0005); 0 of 5 layers without improvement
```

- **error** is the layer's dev error: the best survivor's squared error on
  the dev split, divided by the spread of the dev targets around their
  mean. 1 is no better than predicting the mean, and 0 is a perfect fit
  (see [Criteria](../concepts/criteria.md)).
- **improved by** compares the layer with the last layer that counted as an
  improvement. A gain counts when it reaches the margin, here 0.1% of the
  best error so far.
- **0 of 5 layers without improvement**: the search stops after 5 layers in
  a row without an improvement (`criterion_minimum_width`), and the model
  keeps the layers up to the one with the lowest error (see
  [Splits and stopping](../concepts/splits-and-stopping.md)).
- **capped** and **curvature pairs rejected** count two safeguards of the
  LBFGS optimizer (see [Optimizers](../guides/optimizers.md)).

In this run, layer 7 brings no gain and counts 1 of 5; layer 8 improves
again:

```text
Layer #7: error 0.3195, best 0.3195 at layer 6; improved by -0.0000 (margin 0.0003); 1 of 5 layers without improvement
Layer #8: error 0.3172, best 0.3172 at layer 8; improved by +0.0022 (margin 0.0003); 0 of 5 layers without improvement
Layer #9: error 0.3143, best 0.3143 at layer 9; improved by +0.0029 (margin 0.0003); 0 of 5 layers without improvement
```

The search ends at `max_layer_count` with the last layer still improving.

## 6. Restore the trained model

```python
model = SONN(config, d_model=x.shape[1], feature_names=feature_names)
trainer.load_model_checkpoint(model, "cpu")
print("layers kept:", len(model.layers))
print("dev error per layer:", [round(e, 4) for e in model.layer_err])
```

```text
layers kept: 10
dev error per layer: [0.4879, 0.3758, 0.3496, 0.3444, 0.3391, 0.3281, 0.3195, 0.3195, 0.3172, 0.3143]
```

Training ends by saving the model, the layers up to the best one, as
`model_last.ckpt` in the checkpoint folder. `load_model_checkpoint`
rebuilds that model into a fresh `SONN` built from the same config, which is
how a trained model is used in a later session. `layer_err` holds the dev
error of every layer trained, including any trained past the best one.

## 7. Evaluate on the test split

```python
preds, targets = trainer.infer(model, test_dl)
mse = ((preds - targets) ** 2).mean().item()
mae = (preds - targets).abs().mean().item()
print(f"test MSE {mse:.4f}, MAE {mae:.4f}")
```

```text
test MSE 0.4199, MAE 0.4700
```

`infer` runs the model over a loader and returns the predictions and the
targets in loader order. A model without an output head predicts with the
best neuron of its last layer. The
[California housing tutorial](../tutorials/california-housing.md) reaches a
test MSE of 0.185 on its own split, with engineered features, the
`legendre` family, a wider search, an output head and a fine-tuning pass.

## 8. Prune and inspect

```python
trainer.prune(model)
print("neurons per layer:", [len(layer) for layer in model.layers])
print("features used:", model.get_selected_features())
```

```text
neurons per layer: [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
features used: MedInc, HouseAge, AveRooms, AveBedrms, AveOccup, Latitude, Longitude
```

The prediction reads one neuron of the last layer, which reads a few
neurons of the layer before, and so on down to the features. `prune`
deletes every neuron that does not reach the output, and the predictions
stay the same. Here 11 of the 80 trained neurons remain, and the model
uses 7 of the 8 features: it never reads `Population`. To draw the network,
see [Inspecting a model](../guides/inspecting.md).

## What to change first

- **`max_layer_count`**: this run still improves at its last layer, so a
  higher limit lets it grow deeper. At 30, the search keeps 26 layers and
  reaches a test MSE of 0.3994 (see
  [Splits and stopping](../concepts/splits-and-stopping.md#a-worked-example)).
- **`ref_functions`**: add or swap neuron families. The California housing
  tutorial's best configs use `legendre`.
- **`nbest_neurons` and `max_neuron_models`**: more survivors and more
  candidates per layer widen the search. Training time grows with the
  number of candidates.
- **An output head**: `model.use_output_projection` reads the prediction
  from a linear combination of the last layer's survivors instead of one
  neuron (see [Heads and fine-tuning](../concepts/heads-and-finetune.md)).
