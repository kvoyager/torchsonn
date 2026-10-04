import torch

from collections.abc import Iterable

from torchsonn.optimizers.base import LRLike


class BatchedNewtonLM:
    """Damped Newton step (Levenberg-Marquardt style), batched over the
    candidate ensemble.

    Per member and parameter tensor, solves H delta = g and moves

        theta = theta - lr*delta

    H comes from the `hessians` argument of `step` when given; otherwise it
    is the diagonal estimate diag(g^2) + `damping`*I. Members with
    `active_mask` False keep their parameters. Shared parameters use H and g
    averaged over the active members. There is no gradient clipping, and the
    damping is fixed: `max_damping` is stored but not used by `step`.

    Parameters
    ----------
    params : dict of str -> (B, ...) tensor
        Initial parameters (not stored; the optimizer keeps no state).
    shared_param_names : iterable of str
        Names of the parameters shared by every ensemble member.
    lr : (B,) tensor
        Learning rate per member.
    damping : float
        Diagonal term of the estimated H. Default 1e-2.
    max_damping : float
        Upper limit for the damping; not used. Default 1e3.
    shared_param_lr_multiplier : float
        Factor on the learning rate of the shared parameters. Default 1.0.
    """

    def __init__(
        self,
        params: dict[str, torch.Tensor],
        shared_param_names: Iterable[str],
        lr: LRLike,
        damping: float = 1e-2,
        max_damping: float = 1e3,
        shared_param_lr_multiplier: float = 1.0,
    ) -> None:
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
        """Take one optimizer step for every active member.

        Parameters
        ----------
        params : dict of str -> (B, ...) tensor
            Current parameters, with the members on the leading dimension.
        grads : dict of str -> (B, ...) tensor
            Their gradients, with the same keys and shapes.
        hessians : dict of str -> tensor, optional
            Hessian matrices per parameter; None approximates each as
            diag(g^2).
        active_mask : (B,) bool tensor, optional
            True for the members that step; None steps every member.

        Returns
        -------
        dict of str -> (B, ...) tensor
            The new parameters, with the same keys and shapes.
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
            # Per-member lr broadcast over every trailing dim of the parameter
            # (2-D weights, 3-D RBF centers alike).
            lr = self.lr.view(-1, *([1] * (p.dim() - 1)))

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