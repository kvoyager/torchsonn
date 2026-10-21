"""The vmapped candidate ensemble of one neuron module: its loss functions
and the write-back of the trained parameters."""
from functools import partial
from typing import Callable

import torch
from torch import nn
from torch.func import grad, vmap

from torchsonn.model import SONN
from torchsonn.neurons import BasePolynomNeuron


def write_back_shared_params(model: SONN, params_batch: dict) -> None:
    """Copy shared_proj params back into model.shared_proj after each optimizer step."""
    if model.shared_proj is not None and "shared_proj_weight" in params_batch:
        with torch.no_grad():
            model.shared_proj.weight.copy_(params_batch["shared_proj_weight"])
            model.shared_proj.bias.copy_(params_batch["shared_proj_bias"])


def write_back_params(
    model: nn.Module,
    new_params: dict[str, torch.Tensor],
    new_buffers: dict[str, torch.Tensor] | None = None,
) -> None:
    """
    Copy updated batched parameters & buffers into the model.

    Parameters
    ----------
    model : nn.Module
        The module to update in place.
    new_params : dict of str -> torch.Tensor
        Updated batched parameters.
    new_buffers : dict of str -> torch.Tensor, optional
        Updated buffers.
    """
    # Write back parameters
    for name, param in model.named_parameters():
        if name in new_params:
            with torch.no_grad():
                param.copy_(new_params[name])

    # Write back buffers
    if new_buffers is not None:
        for name, buf in model.named_buffers():
            if name in new_buffers:
                buf.copy_(new_buffers[name])


def create_loss_functions(
    model: SONN, module: BasePolynomNeuron,
) -> tuple[Callable, Callable, Callable, dict[str, torch.Tensor], dict[str, torch.Tensor], list[str]]:
    """
    Build the vmapped loss functions for one neuron module's candidates.

    Parameters
    ----------
    model : SONN
        The model being grown.
    module : BasePolynomNeuron
        The neuron module whose candidates are trained.

    Returns
    -------
    loss_fn_vmapped : callable
        Per-candidate gradient of the training loss, including the
        `train.ridge_alpha` penalty.
    eval_loss_fn_vmapped : callable
        Per-candidate loss without the penalty.
    pred_fn_vmapped : callable
        Per-candidate predictions.
    params_batch : dict of str to tensor
        The module's parameters, plus the shared projection for
        multi-class models that use one.
    buffers_batch : dict of str to tensor
        The module's buffers and `src_idxs`.
    shared_param_names : list of str
        Names of the parameters shared by all candidates (not vmapped).
    """
    # prepare params & buffers as dicts
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    buffers["src_idxs"] = module.src_idxs

    params_batch = params
    buffers_batch = buffers

    # set in_dims for dicts
    params_in_dims = {k: 0 for k in params_batch}
    buffers_in_dims = {k: 0 for k in buffers_batch}

    if isinstance(model.loss_fn, nn.NLLLoss) and model.soft_binner is None and model.shared_proj is not None:
        # shared_proj is trained jointly by all ensemble members (in_dims=None).
        # The optimizer's shared path averages their gradients into one update.
        # use_neuron_proj: proj_weight/proj_bias live on the neuron module and
        # are already in params via named_parameters() with in_dims=0.
        params["shared_proj_weight"] = model.shared_proj.weight
        params["shared_proj_bias"] = model.shared_proj.bias
        params_in_dims["shared_proj_weight"] = None
        params_in_dims["shared_proj_bias"] = None
        shared_param_names = ["shared_proj_weight", "shared_proj_bias"]
    else:
        shared_param_names = []

    # Ridge α applies to the training-loss partial only. The eval and
    # pred partials are pinned to 0.0 so the dev loss used for
    # neuron_selection, the layer-error reported in the log, and the
    # final R² / RMSE in report() all reflect pure data fit, not the
    # penalised training objective. This is the conventional ridge
    # tuning regime: "train regularized, evaluate raw".
    ridge_alpha = float(model.param.train.ridge_alpha)

    # create loss function that is vmapped & differentiated
    loss_fn_vmapped = vmap(
        grad(partial(model.compute_loss, module, model.loss_fn, model.param.model.type, model.soft_binner, ridge_alpha)),
        in_dims=(params_in_dims, buffers_in_dims, None, None)
    )

    # Create vmapped evaluation loss function (no gradient)
    eval_loss_fn_vmapped = vmap(
        partial(model.compute_loss, module, model.loss_fn, model.param.model.type, model.soft_binner, 0.0),
        in_dims=(params_in_dims, buffers_in_dims, None, None)
    )

    pred_fn_vmapped = vmap(
        partial(model.compute_loss, module, None, model.param.model.type, model.soft_binner, 0.0),
        in_dims=(params_in_dims, buffers_in_dims, None, None)
    )

    return loss_fn_vmapped, eval_loss_fn_vmapped, pred_fn_vmapped, params_batch, buffers_batch, shared_param_names
