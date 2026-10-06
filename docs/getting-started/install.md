# Install

TorchSONN needs Python 3.12 or later. These pages describe version 0.1.6.
The latest release on PyPI is 0.1.1, which does not have everything
described here (the `rbf` neuron family, for example), so install 0.1.6
from GitHub:

```bash
pip install "git+https://github.com/kvoyager/torchsonn.git@v0.1.6"
```

The PyPI release installs with:

```bash
pip install torchsonn
```

pip also installs the dependencies: PyTorch, NumPy, scikit-learn, Hydra,
OmegaConf and tqdm.

## Plotting

Network diagrams and the layer-error plot need the `viz` extra, which adds
`graphviz` and `matplotlib`:

```bash
pip install "torchsonn[viz] @ git+https://github.com/kvoyager/torchsonn.git@v0.1.6"
```

The `graphviz` Python package only drives the Graphviz programs, which are
installed separately: the `dot` command must be on your `PATH`. Get them
from [graphviz.org/download](https://graphviz.org/download/). Training and
prediction work without any of this.

## GPU

pip installs the PyTorch build that PyPI serves by default for your
platform, which on some platforms runs on the CPU only. For a GPU, install
the PyTorch build for your CUDA version first, using the selector on
[pytorch.org](https://pytorch.org/get-started/locally/), then install
TorchSONN. A model trains on the GPU when its config sets
`train.device: cuda`.

The end-to-end fine-tuning pass is fastest on a CUDA device with the `adam`
or `adamw` optimizer, where its training step runs as a captured CUDA graph
(see [Performance](../guides/performance.md)).

## Development install

To work on the code, clone the repository and install it in editable mode
with the test dependencies:

```bash
git clone https://github.com/kvoyager/torchsonn.git
cd torchsonn
pip install -e ".[test]"
pytest
```

The `test` extra includes `viz`, so the plotting tests need the Graphviz
programs as well. To build this documentation, add the `docs` extra and
start the preview server:

```bash
pip install -e ".[docs]"
mkdocs serve
```

`mkdocs build --strict` builds the site into `site/` and fails on any
broken link.
