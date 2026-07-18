"""TorchSONN — self-organizing polynomial neural network on PyTorch.

Everything needed for a full train / infer loop is re-exported here:

    from torchsonn import SONN, SONNConfig, SONNDataset, Trainer

`PlotModel` is deliberately *not* re-exported: it pulls in the optional
`graphviz` dependency (``pip install "torchsonn[viz]"``), which would make a
bare `import torchsonn` fail on a base install. It stays reachable as
`from torchsonn.plot_model import PlotModel`.
"""
from torchsonn.config import SONNConfig
from torchsonn.data.dataset import SONNDataset
from torchsonn.model import SONN
from torchsonn.trainer import Trainer


__all__ = [
    "SONN",
    "SONNConfig",
    "SONNDataset",
    "Trainer",
]