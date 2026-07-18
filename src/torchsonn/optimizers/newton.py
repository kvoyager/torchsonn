import math

import torch

from collections.abc import Iterable

from torchsonn.optimizers.base import LRLike


class BatchedNewton:
    def __init__(
        self,
        params: dict[str, torch.Tensor],
        shared_param_names: Iterable[str],
        lr: LRLike,
        damping: float = 1e-3,
        shared_param_lr_multiplier: float = 1.0,
    ) -> None:
        """
        Newton-like optimizer with damping (Levenberg-style regularization).

        params: dict of parameter tensors with leading batch dimension
        shared_param_names: list of keys for shared parameters
        lr: base learning rate (can be scalar tensor or per-model tensor)
        damping: small constant added to Hessian diagonal
        """
        self.lr = lr
        self.damping = damping
        self.shared_param_names = set(shared_param_names)
        self.shared_param_lr_multiplier = shared_param_lr_multiplier

        # Per-parameter state for Hessian approximations
        self.prev_params = {k: v.clone() for k, v in params.items()}
        self.prev_grads = {k: torch.zeros_like(v) for k, v in params.items()}
        # H is stored as (leading_dim, flat_param_size, flat_param_size). For
        # params with a leading batch dim (>=2 D), one Hessian per batch element.
        # For 1-D params (e.g. shared_proj.bias) there is no batch dim, so we
        # keep a single Hessian with leading dim 1.
        self.H = {}
        for k, v in params.items():
            if v.dim() >= 2:
                b = v.shape[0]
                d = math.prod(v.shape[1:])
            else:
                b = 1
                d = v.numel()
            self.H[k] = torch.eye(d, device=v.device).unsqueeze(0).repeat(b, 1, 1)

    def step(
        self,
        params: dict[str, torch.Tensor],
        grads: dict[str, torch.Tensor],
        active_mask: torch.Tensor | None = None,
        b: object = None,
    ) -> dict[str, torch.Tensor]:
        """
        params, grads: dicts of tensors with leading batch dimension
        active_mask: tensor of shape (batch,) bool, True if model active
        """
        batch_size = next(iter(params.values())).shape[0]
        device = next(iter(params.values())).device

        if active_mask is None:
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        new_params = {}

        for k in params.keys():
            p = params[k]
            g = grads[k]
            H = self.H[k]
            prev_p = self.prev_params[k]
            prev_g = self.prev_grads[k]

            update_mask = active_mask.view(-1, *([1] * (p.dim() - 1)))

            lr = self.lr.unsqueeze(dim=1)
            if k in self.shared_param_names:
                idx = torch.where(active_mask)[0]
                lr = lr.index_select(0, idx).mean() * self.shared_param_lr_multiplier
                g = g.index_select(0, idx).mean(dim=0, keepdim=True)
                p = p.index_select(0, idx).mean(dim=0, keepdim=True)
                prev_p = prev_p.index_select(0, idx).mean(dim=0, keepdim=True)
                prev_g = prev_g.index_select(0, idx).mean(dim=0, keepdim=True)
                H = H.index_select(0, idx).mean(dim=0, keepdim=True)

            p_flat = p.reshape(p.shape[0], -1)
            g_flat = g.reshape(p.shape[0], -1)
            prev_p_flat = prev_p.reshape(prev_p.shape[0], -1)
            prev_g_flat = prev_g.reshape(prev_g.shape[0], -1)

            s = p_flat - prev_p_flat
            y = g_flat - prev_g_flat

            for i in range(p_flat.shape[0]):
                if not active_mask[i]:
                    continue

                s_i = s[i].unsqueeze(1)
                y_i = y[i].unsqueeze(1)
                H_i = H[i]
                rho = 1.0 / (y_i.T @ s_i + 1e-12)
                I = torch.eye(H_i.shape[0], device=device)

                # BFGS-style Hessian update
                H_i = (I - rho * s_i @ y_i.T) @ H_i @ (I - rho * y_i @ s_i.T) + rho * (s_i @ s_i.T)
                H_i += self.damping * I  # regularization for stability

                # Newton step: Δθ = H⁻¹ g
                try:
                    delta = torch.linalg.solve(H_i, g_flat[i].unsqueeze(1))
                except RuntimeError:
                    delta = torch.pinverse(H_i) @ g_flat[i].unsqueeze(1)

                p_flat[i] -= lr * delta.squeeze(1)
                H[i] = H_i

            new_params[k] = p_flat.reshape_as(p)
            self.H[k] = H
            self.prev_params[k] = p.detach().clone()
            self.prev_grads[k] = g.detach().clone()

        return new_params