"""Cutting a trained layer down to a subset of its output columns."""
import torch

from torchsonn.layer import NeuronModuleList, SONNLayer


def keep_columns(layer: SONNLayer, used: torch.Tensor) -> None:
    """Cut a trained layer down to the output columns `used` (sorted, ascending).

    Each neuron module keeps its rows among them, and the layer's
    `module_idxs` is rebuilt with renumbered (new_module_idx, arange(K))
    rows, the same shape contract as train_layer's output, so a later
    prune() sees consistent (module_idxs <-> modules) state. train_layer
    lays module_idxs out module by module, so kept column used[r] becomes
    column r and `err_values[used]` is already in the new order. The
    layer's `d_model` is not touched.
    """
    dev = layer.module_idxs.device
    kept_idxs = layer.module_idxs[used.to(dev)]
    new_modules = []
    renumbered_rows = []
    for new_idx, module_idx in enumerate(torch.unique(kept_idxs[:, 0])):
        rows = kept_idxs[kept_idxs[:, 0] == module_idx]
        module = layer.neuron_models[module_idx]
        module.prune(rows[:, 1])
        new_modules.append(module)
        k = rows.shape[0]
        renumbered_rows.append(torch.stack([
            torch.full((k,), new_idx, dtype=torch.long, device=dev),
            torch.arange(k, dtype=torch.long, device=dev),
        ], dim=1))
    layer.neuron_models = NeuronModuleList(new_modules)
    # Keep module_idxs / err_values in sync with the pruned neuron list so
    # a second prune() call (e.g. the iris tutorial that prunes for the
    # `_pruned_model` plot after already pruning for the regular one)
    # doesn't index the modules with stale rows.
    if layer.err_values is not None:
        layer.err_values = layer.err_values[used.to(layer.err_values.device)]
    layer.module_idxs = torch.cat(renumbered_rows)
