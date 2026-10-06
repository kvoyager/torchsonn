# API reference

These pages are generated from the docstrings in the source. They list the
public classes, functions and methods; names that start with an
underscore are left out. The [guides](../../guides/configuration.md) show
how the pieces fit together, and [Configuration keys](../config.md) lists
every configuration key.

The package re-exports the four names a training script needs:

```python
from torchsonn import SONN, SONNConfig, SONNDataset, Trainer
from torchsonn.plot_model import PlotModel  # needs the viz extra
```

`PlotModel` is not re-exported, because it imports `graphviz`, which a
base install does not have.

| Page | Contents |
|---|---|
| [SONN](model.md) | the model, its layers and the layer-creation error |
| [Trainer](trainer.md) | training, the head and fine-tune passes, pruning, prediction, checkpoints, the growth rule, logging |
| [Data](data.md) | `SONNDataset` and the deterministic split helpers |
| [Neurons](neurons.md) | the neuron families and their base classes |
| [Losses and criteria](loss.md) | the training loss and the criterion functions |
| [Modules](modules.md) | the squash and the soft binner |
| [Optimizers](optimizers.md) | the batched optimizers of the candidate fit |
| [Plotting](plot_model.md) | the network diagram |
| [Configuration](config.md) | the configuration schema's classes |

<small>Checked against TorchSONN 0.1.5.</small>
