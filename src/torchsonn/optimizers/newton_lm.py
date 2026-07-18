import torch

from collections.abc import Iterable

from torchsonn.optimizers.base import LRLike


class BatchedNewtonLM:
    def __init__(
        self,
        params: dict[str, torch.Tensor],
        shared_param_names: Iterable[str],
        lr: LRLike,
        damping: float = 1e-2,
        max_damping: float = 1e3,
        shared_param_lr_multiplier: float = 1.0,
    ) -> None:
        """
        Batched Newton / Levenberg–Marquardt optimizer.

        params: dict of batched tensors (same as BatchedSGD)
        shared_param_names: list of parameter names shared across models
        lr: tensor of learning rates (shape [batch])
        damping: initial damping term for LM
        max_damping: upper limit for damping
        """
        self.lr = lr
        self.damping = damping
        self.max_damping = max_damping
        self.shared_param_names = set(shared_param_names)
        self.shared_param_lr_multiplier = shared_param_lr_multiplier

    def step(
        self,
        params: dict[str, torch.Tensor],
        grads: dict[str, torch.Tensor],
        hessians: dict[str, torch.Tensor] | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        params, grads: dict of tensors with leading batch dimension
        hessians: optional dict of Hessian matrices per param, or None (approximates as diag(g^2))
        active_mask: tensor of shape (batch,) bool, True if model is active
        """
        batch_size = next(iter(params.values())).shape[0]
        device = next(iter(params.values())).device

        if active_mask is None:
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        new_params = {}
        for k in params.keys():
            p = params[k]
            g = grads[k]

            # Estimate Hessian if not provided
            if hessians is not None and k in hessians:
                H = hessians[k]
            else:
                # Diagonal approximation: H ≈ diag(g^2)
                flat_g = g.view(batch_size, -1)
                H = torch.diag_embed(flat_g.pow(2)) + self.damping * torch.eye(flat_g.shape[-1], device=device)

            update_mask = active_mask.view(-1, *([1] * (p.dim() - 1)))
            lr = self.lr.unsqueeze(dim=1)

            if k in self.shared_param_names:
                idx = torch.where(active_mask)[0]
                lr = lr.index_select(0, idx).mean() * self.shared_param_lr_multiplier
                g_mean = g.index_select(0, idx).mean(dim=0)
                H_mean = H.index_select(0, idx).mean(dim=0)

                delta = torch.linalg.solve(H_mean, g_mean.view(-1, 1)).view_as(g_mean)
                update = lr * delta
                new_params[k] = p - update
            else:
                flat_g = g.view(batch_size, -1, 1)
                delta = torch.linalg.solve(H, flat_g).squeeze(-1).view_as(p)
                update = lr.view(-1, *([1] * (p.dim() - 1))) * delta
                new_params[k] = torch.where(update_mask, p - update, p)

        return new_params