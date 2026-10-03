# TorchSONN

TorchSONN is a Python library implementing a self-organizing neural
network built on PyTorch. Layers of small neurons are grown one at a time;
each layer tries every pair (or tuple) of the previous layer's outputs
against a set of reference functions and keeps the top-k that minimize a
validation criterion. Training stops automatically when adding a layer no
longer reduces the criterion error. Three families of reference functions
are available:

- **Power-basis polynomials**: `linear`, `linear_cov`, `quadratic`,
  `cubic` and multi-input `polyquad`;
- **Orthogonal polynomials**: `legendre` and `chebyshev`, with a
  configurable degree and number of inputs. They stay well conditioned at
  higher degrees, where the raw power basis breaks down;
- **Gaussian radial basis functions**: `rbf`, local bumps whose centres and
  widths are initialized by k-means and then learned.

It is a GPU-accelerated extension of
[GmdhPy](https://github.com/kvoyager/GmdhPy), an earlier scikit-learn-style
library implementing the iterative Group Method of Data Handling (GMDH).
TorchSONN reimplements the same self-organizing algorithm on PyTorch,
bringing GPU acceleration to model training and inference.

## Plotting a model

A trained network can be drawn as a diagram with `torchsonn.plot_model`.
Each box is a neuron, and its incoming edges are the inputs it reads:

```python
from torchsonn.plot_model import PlotModel

PlotModel(model, filename="model", plot_neuron_name=True).plot()  # writes model.svg
```

This example is the pruned network from the
[CCPP tutorial](tutorials/ccpp/README.md), cut down to the neurons that
reach the output:

<details open>
<summary>Pruned CCPP network (click to collapse; click the image for full size)</summary>

<br>

<a href="img/ccpp_pruned_model.svg"><img src="img/ccpp_pruned_model.svg" alt="Pruned network from the CCPP tutorial" width="80%"></a>

</details>

Plotting needs the `viz` extra and the system Graphviz binaries (see
below).

## Install

TorchSONN needs Python 3.12 or later. This README and the documentation
describe the `main` branch on GitHub; install it with:

```bash
pip install "git+https://github.com/kvoyager/torchsonn.git"
```

The latest release on PyPI is 0.1.1, which does not have everything
described here (the `rbf` neuron family, for example):

```bash
pip install torchsonn
```

Plotting is opt-in via the `viz` extra, which adds `graphviz` (for the
`torchsonn.plot_model` network diagrams) and `matplotlib` (for
`SONN.plot_layer_error`):

```bash
pip install "torchsonn[viz] @ git+https://github.com/kvoyager/torchsonn.git"
```

`graphviz` additionally requires the system Graphviz binaries — `dot` must be
on your `PATH` (see [graphviz.org/download](https://graphviz.org/download/)).
Training and inference work without any of this.

For development, clone the repo and install editably:

```bash
git clone https://github.com/kvoyager/torchsonn.git
cd torchsonn
pip install -e ".[test]"
```

## Quickstart

California housing regression on the CPU, in about two minutes:

```python
import numpy as np
from omegaconf import OmegaConf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchsonn import SONN, SONNDataset, Trainer

# The rows come grouped by location, so shuffle before splitting:
# 50% train, 25% dev, 25% test; standardize on the train rows only.
x, y = fetch_california_housing(return_X_y=True)
order = np.random.default_rng(0).permutation(len(x))
x, y = x[order], y[order]
i_train, i_dev = int(0.50 * len(x)), int(0.75 * len(x))
x = StandardScaler().fit(x[:i_train]).transform(x).astype(np.float32)
# A few rows sit hundreds of standard deviations out (population, average
# occupancy); clip them so the polynomials do not extrapolate there.
x = np.clip(x, -5.0, 5.0)
y = y.astype(np.float32)

def loader(start, stop, shuffle=False):
    dataset = SONNDataset(x[start:stop], y[start:stop])
    # One batch holds a whole split (10,320 training rows).
    return DataLoader(dataset, batch_size=16384, shuffle=shuffle)

train_dl = loader(0, i_train, shuffle=True)
dev_dl = loader(i_train, i_dev)
test_dl = loader(i_dev, None)

# Only what differs from the defaults in torchsonn.config.SONNConfig.
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

model, trainer = SONN(config, d_model=x.shape[1]), Trainer(config)
trainer.set_seed(config.train.seed)
trainer.train(model, train_dl, dev_dl, test_dl)
trainer.load_model_checkpoint(model, "cpu")
preds, targets = trainer.infer(model, test_dl)
print(f"test MSE: {((preds - targets) ** 2).mean().item():.4f}")
```

The [regression](docs/getting-started/quickstart-regression.md) and
[classification](docs/getting-started/quickstart-classification.md)
quickstarts walk through a model like this one step by step.

## Tutorials

Five end-to-end examples live under `tutorials/`, each with every config
key overridable on the command line:

```bash
# Iris classification — multi-class, linear_cov neurons.
python -m tutorials.iris.iris_recognition

# California housing regression — gmdhpy-equivalent setup with LBFGS.
python -m tutorials.california_housing.california_housing

# UCI Concrete compressive-strength regression — reports MSE / MAE / R².
python -m tutorials.concrete.concrete

# UCI Combined Cycle Power Plant regression — 5×2 cross-validation, reports
# mean ± std of RMSE / MAE / R² over the 10 folds (Tüfekci & Kaya benchmark).
python -m tutorials.ccpp.ccpp

# Otto Group product classification — multi-class log loss, out_proj fine-tune.
python tutorials/otto/otto_classification.py
```

Hydra overrides work on any field in the schema. A few useful ones:

```bash
# Resume from the last checkpoint
python -m tutorials.iris.iris_recognition resume=true

# Swap optimizer + tune its kwargs
python -m tutorials.california_housing.california_housing \
    train.optimizer.name=adam \
    train.optimizer.optimizer_params.lr=5e-3

# Redirect Hydra's per-run output directory
python -m tutorials.iris.iris_recognition hydra.run.dir=/tmp/sonn_run
```

The iris and otto tutorials also ship notebook versions
(`tutorials/iris/iris_recognition.ipynb` and `tutorials/otto/otto.ipynb`)
that render the layer-error curve, the confusion matrix, and the graphviz
network diagrams inline.

Every tutorial except `tutorials/otto/otto_classification.py` plots its
results, so they need the `[viz]` extra — iris, california_housing,
concrete, and ccpp render network diagrams (graphviz), and iris plus both
notebooks call `SONN.plot_layer_error` (matplotlib). The otto script runs on
a base install.

## Configuration

Every setting has a typed default in `torchsonn.config.SONNConfig`, and a
configuration lists only what it changes, from Python or from a Hydra YAML
file with command-line overrides. The
[configuration guide](docs/guides/configuration.md) shows both, and the
[configuration reference](docs/reference/config.md) lists every key.

## License

Released under the MIT License — see [LICENSE](LICENSE) for the full text.

## Citation

If you use TorchSONN in your research, please cite it as:

```bibtex
@software{kolokolov_torchsonn_2026,
  author    = {Kolokolov, Konstantin},
  title     = {{TorchSONN}: A {PyTorch} Implementation of the self-organizing polynomial neural network},
  year      = {2026},
  version   = {0.1.5},
  url       = {https://github.com/kvoyager/torchsonn},
  note      = {GitHub repository}
}
```

