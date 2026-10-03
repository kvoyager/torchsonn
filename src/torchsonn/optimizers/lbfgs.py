import torch
from collections import deque
from collections.abc import Iterable
from typing import Any

from torchsonn.optimizers.base import BaseOptimizer, LRLike


class BatchedLBFGS(BaseOptimizer):
    """Limited-memory BFGS, batched over the candidate ensemble.

    Each member keeps its own history of the last `history_size` correction
    pairs s = delta theta, y = delta g per parameter tensor. The two-loop
    recursion turns the clipped gradient into a quasi-Newton direction
    d ~ H^-1 g, and the step is

        theta = theta - lr*d

    with no line search. A pair is stored only if it passes the curvature
    condition (`curvature_eps`), and each member's update of a parameter
    tensor is capped at norm `max_step`. Members with `active_mask` False in
    `step` keep their parameters and store no pair. Shared parameters keep
    one history, driven by the gradient averaged over the active members.
    `stats` counts steps, pairs, rejected pairs, updates and capped updates
    for the trainer's log.

    Parameters
    ----------
    params : dict of str -> (B, ...) tensor
        Initial parameters.
    shared_param_names, lr, clip_value, clip_norm, shared_param_lr_multiplier
        See `BaseOptimizer`.
    history_size : int
        Correction pairs kept per member. Default 10.
    max_step : float or None
        Cap on the norm of one member's update of one parameter tensor.
        Default 1.0.
    curvature_eps : float or None
        Store a pair only if y.s > curvature_eps*|s|*|y|. Default 1e-8.

    `__init__` explains why the two guards exist and how the history is
    stored.
    """

    def __init__(
        self,
        params: dict[str, torch.Tensor],
        shared_param_names: Iterable[str],
        lr: LRLike,
        history_size: int = 10,
        clip_value: float | None = None,
        clip_norm: float | None = None,
        shared_param_lr_multiplier: float = 1.0,
        max_step: float | None = 1.0,
        curvature_eps: float | None = 1e-8,
    ) -> None:
        """
        params: dict of batched tensors (shape [B, ...])
        shared_param_names: iterable of keys that are shared parameters
        lr: torch.Tensor of shape (B,) or scalar
        history_size: number of (s,y) correction pairs to keep
        clip_value / clip_norm: forwarded to BaseOptimizer.gradient_clipping
        max_step: cap on the norm of one member's update of one parameter
            tensor, in parameter units (a crude trust region); None = no cap.
        curvature_eps: a correction pair (s, y) is stored only if
            y.s > curvature_eps * |s| |y| (the BFGS curvature condition, with
            the standard skipping rule); None = store every pair with s != 0,
            the historical behaviour.

        Why the two guards. The two-loop recursion scales its direction by
        s.y / y.y and takes a fixed step. On a direction whose gradient and
        curvature vanish together (an RBF bump losing its mass) that scale
        is a ratio of two vanishing numbers and the step does not shrink
        with the gradient: one step threw centres tens of standard units
        off the data. A pair with y.s <= 0 makes the inverse-Hessian
        estimate indefinite and the direction can point uphill. The cap
        bounds the damage of any single step; the curvature test keeps the
        estimate positive definite. Neither changes a well-conditioned
        convex fit (the polynomial families) beyond its first steps.

        Storage. For the batched (per-member) parameters the correction
        history is a pair of tensors `s_hist[k]`, `y_hist[k]` of shape
        (B, history_size, P) plus a per-member fill count `hist_count[k]`;
        the newest pair sits in the last slot and the valid slots are the
        last `hist_count` ones. The two-loop recursion then runs as
        `history_size` batched operations over the whole ensemble instead
        of a Python loop over its members, which was most of a layer's time.
        Shared parameters keep a single deque history and the single-vector
        recursion.
        """
        super().__init__(shared_param_names, lr, clip_value, clip_norm, shared_param_lr_multiplier)
        self.history_size = int(history_size)
        if self.history_size < 1:
            raise ValueError(f"history_size must be >= 1, got {history_size}")
        if max_step is not None and float(max_step) <= 0.0:
            raise ValueError(f"max_step must be > 0 or None, got {max_step}")
        if curvature_eps is not None and float(curvature_eps) < 0.0:
            raise ValueError(f"curvature_eps must be >= 0 or None, got {curvature_eps}")
        self.max_step = None if max_step is None else float(max_step)
        self.curvature_eps = None if curvature_eps is None else float(curvature_eps)
        # Diagnostics for the trainer's log: how often the guards acted.
        self.stats = {"steps": 0, "pairs": 0, "rejected": 0, "updates": 0, "capped": 0}

        first = next(iter(params.values()))
        self.batch_size = first.shape[0]

        # per-key storage:
        # - for non-shared keys: (B, m, P) tensors + a (B,) fill count
        # - for shared keys: a single deque used for the shared parameter history
        self.s_hist: dict[str, Any] = {}
        self.y_hist: dict[str, Any] = {}
        self.hist_count: dict[str, torch.Tensor] = {}
        for k, v in params.items():
            if k in self.shared_param_names:
                self.s_hist[k] = deque(maxlen=self.history_size)
                self.y_hist[k] = deque(maxlen=self.history_size)
            else:
                self._init_batched_history(k, v)

        # prev params/grads for computing s = p - p_prev, y = g - g_prev
        self.prev_params = {k: v.detach().clone() for k, v in params.items()}
        self.prev_grads = {k: torch.zeros_like(v) for k, v in params.items()}

    def _init_batched_history(self, k: str, v: torch.Tensor) -> None:
        b = v.shape[0]
        p = v[0].numel() if b > 0 else 0
        self.s_hist[k] = torch.zeros((b, self.history_size, p), dtype=v.dtype, device=v.device)
        self.y_hist[k] = torch.zeros((b, self.history_size, p), dtype=v.dtype, device=v.device)
        self.hist_count[k] = torch.zeros(b, dtype=torch.long, device=v.device)

    def _flatten(self, t: torch.Tensor) -> torch.Tensor:
        """Flatten a parameter tensor preserving device/dtype: input shape [B, ...] -> [B, P]."""
        B = t.shape[0]
        return t.reshape(B, -1)

    def _accept_pair(self, s: torch.Tensor, y: torch.Tensor) -> bool:
        """Curvature condition for one (s, y) pair (1-D tensors)."""
        s_norm = s.norm()
        if s_norm <= 1e-12:
            return False
        if self.curvature_eps is None:
            return True
        return bool(y.dot(s) > self.curvature_eps * s_norm * y.norm())

    def _cap_updates(self, updates: torch.Tensor, active: torch.Tensor | None) -> torch.Tensor:
        """Scale each row of `updates` (n, P) down to `max_step` in norm; counts
        the rows that were capped among the active ones."""
        if self.max_step is None:
            return updates
        norms = updates.norm(dim=-1, keepdim=True)
        over = norms > self.max_step
        if active is not None:
            over = over & active.view(-1, 1)
        self.stats["capped"] += int(over.sum().item())
        scale = torch.clamp(self.max_step / norms.clamp(min=1e-12), max=1.0)
        return updates * scale

    def _two_loop_recursion_single(
        self,
        s_list: Iterable[torch.Tensor],
        y_list: Iterable[torch.Tensor],
        g: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classical two-loop recursion for a single parameter-vector (non-batched).
        s_list, y_list: lists/iterables of tensors shape (P,)
        g: tensor shape (P,)
        returns: r tensor shape (P,) ~ H_k * g
        """
        if len(s_list) == 0:
            return g.clone()

        s_list = list(s_list)
        y_list = list(y_list)

        q = g.clone()
        alphas = []
        rhos = []
        eps = 1e-12

        for s, y in zip(reversed(s_list), reversed(y_list)):
            rho = 1.0 / (y.dot(s) + eps)
            alpha = rho * s.dot(q)
            q = q - alpha * y
            alphas.append(alpha)
            rhos.append(rho)

        s_last = s_list[-1]
        y_last = y_list[-1]
        denom = y_last.dot(y_last) + eps
        gamma = (s_last.dot(y_last)) / denom
        r = gamma * q

        for s, y, rho, alpha in zip(s_list, y_list, reversed(rhos), reversed(alphas)):
            beta = rho * y.dot(r)
            r = r + s * (alpha - beta)

        return r

    @staticmethod
    def _two_loop_recursion_batched(
        S: torch.Tensor, Y: torch.Tensor, count: torch.Tensor, g: torch.Tensor,
    ) -> torch.Tensor:
        """
        Two-loop recursion for every member at once.
        S, Y: (B, m, P) histories, newest pair in the last slot, the last
              `count[b]` slots valid for member b
        g: (B, P) gradients
        returns: (B, P) ~ H_k g per member; the plain gradient for a member
        with an empty history. Same arithmetic as the single-vector version,
        with the per-slot dot products batched over B.
        """
        b, m, _ = S.shape
        eps = 1e-12
        slots = torch.arange(m, device=S.device)
        valid = slots.unsqueeze(0) >= (m - count).unsqueeze(1)                 # (B, m)
        ys = (S * Y).sum(dim=-1)                                                # (B, m)
        rho = 1.0 / (ys + eps)
        zero = torch.zeros((), dtype=g.dtype, device=g.device)

        q = g.clone()
        alphas = [None] * m
        for j in range(m - 1, -1, -1):                                          # newest -> oldest
            a = rho[:, j] * (S[:, j] * q).sum(dim=-1)
            a = torch.where(valid[:, j], a, zero)
            alphas[j] = a
            q = q - a.unsqueeze(1) * Y[:, j]

        has = count > 0
        denom = (Y[:, m - 1] * Y[:, m - 1]).sum(dim=-1) + eps
        gamma = torch.where(has, ys[:, m - 1] / denom, torch.ones_like(denom))
        r = gamma.unsqueeze(1) * q

        for j in range(m):                                                      # oldest -> newest
            beta = rho[:, j] * (Y[:, j] * r).sum(dim=-1)
            upd = (alphas[j] - beta).unsqueeze(1) * S[:, j]
            r = torch.where(valid[:, j].unsqueeze(1), r + upd, r)
        return r

    def step(
        self,
        params: dict[str, torch.Tensor],
        grads: dict[str, torch.Tensor],
        active_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        params, grads: dicts mapping param name -> tensor of shape [B, ...]
        active_mask: optional boolean tensor shape (B,) where True indicates active models
        returns new_params: dict with same keys and shapes
        """
        device = next(iter(params.values())).device
        if active_mask is None:
            active_mask = torch.ones(self.batch_size, dtype=torch.bool, device=device)

        new_params = {}
        self.stats["steps"] += 1

        for k in params.keys():
            p = params[k]
            g = grads[k]

            # Apply gradient clipping (no-op when clip_value / clip_norm are None).
            g = self.gradient_clipping(g)

            if k in self.shared_param_names:
                # Shared params are not batched — shape is the raw param shape,
                # not (ensemble_size, ...). _flatten would misuse p.shape[0] as B
                # and produce a tensor with only `first_dim` rows, making
                # index_select with ensemble indices go out of range. Work on
                # flat 1-D vectors directly instead.
                idx = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    new_params[k] = p
                    continue

                # p: (*param_shape,) — single shared tensor, not batched.
                # g: (ensemble_size, *param_shape) — vmap produces one grad per
                # ensemble member even for in_dims=None params. Average over
                # active members to get a single update direction.
                p_1d = p.detach().reshape(-1)
                g_1d = g.index_select(0, idx).mean(dim=0).detach().reshape(-1)
                prev_p_1d = self.prev_params[k].reshape(-1)
                prev_g_1d = self.prev_grads[k].reshape(-1)

                s = p_1d - prev_p_1d
                y = g_1d - prev_g_1d

                if s.abs().sum() > 1e-12:
                    self.stats["pairs"] += 1
                    if self._accept_pair(s, y):
                        self.s_hist[k].append(s.clone())
                        self.y_hist[k].append(y.clone())
                    else:
                        self.stats["rejected"] += 1

                d = self._two_loop_recursion_single(self.s_hist[k], self.y_hist[k], g_1d)

                if isinstance(self.lr, torch.Tensor):
                    lr_mean = self.lr.index_select(0, idx).mean().item() * self.shared_param_lr_multiplier
                else:
                    lr_mean = float(self.lr) * self.shared_param_lr_multiplier

                self.stats["updates"] += 1
                update = self._cap_updates((lr_mean * d).view(1, -1), None).view(-1)
                new_params[k] = (p_1d - update).reshape_as(p)

                self.prev_params[k] = p.detach().clone()
                # Save the mean gradient (not the full batched tensor) so the
                # shape stays (*param_shape,) and matches p on the next step.
                self.prev_grads[k] = g_1d.reshape_as(p).detach().clone()

            else:
                p_flat = self._flatten(p)
                g_flat = self._flatten(g)
                prev_p_flat = self._flatten(self.prev_params[k])
                prev_g_flat = self._flatten(self.prev_grads[k])

                s = p_flat - prev_p_flat
                y = g_flat - prev_g_flat

                # Curvature condition, batched: which members' pairs to store.
                s_norm = s.norm(dim=1)
                nonzero = (s_norm > 1e-12) & active_mask
                if self.curvature_eps is None:
                    accept = nonzero
                else:
                    ys = (s * y).sum(dim=1)
                    accept = nonzero & (ys > self.curvature_eps * s_norm * y.norm(dim=1))
                n_nonzero = int(nonzero.sum().item())
                if n_nonzero:
                    self.stats["pairs"] += n_nonzero
                    self.stats["rejected"] += n_nonzero - int(accept.sum().item())
                    if accept.any():
                        acc = torch.nonzero(accept, as_tuple=False).squeeze(1)
                        S, Y = self.s_hist[k], self.y_hist[k]
                        # Push: shift the member's history left, newest in the last slot.
                        S[acc] = torch.cat([S[acc, 1:], s[acc].detach().unsqueeze(1)], dim=1)
                        Y[acc] = torch.cat([Y[acc, 1:], y[acc].detach().unsqueeze(1)], dim=1)
                        self.hist_count[k][acc] = (self.hist_count[k][acc] + 1).clamp(max=self.history_size)

                d = self._two_loop_recursion_batched(self.s_hist[k], self.y_hist[k], self.hist_count[k], g_flat)
                if isinstance(self.lr, torch.Tensor):
                    updates = self.lr.view(-1, 1).to(dtype=d.dtype) * d
                else:
                    updates = float(self.lr) * d

                self.stats["updates"] += int(active_mask.sum().item())
                updates = self._cap_updates(updates, active_mask)

                p_new_flat = torch.where(active_mask.view(self.batch_size, 1), p_flat - updates, p_flat)

                new_params[k] = p_new_flat.reshape_as(p)

                self.prev_params[k] = p.detach().clone()
                self.prev_grads[k] = g.detach().clone()

        return new_params

    def state_dict(self) -> dict[str, Any]:
        """Snapshot of curvature history + prev params/grads + lr so a resumed run
        keeps the L-BFGS approximation rather than starting from identity."""
        def dump_hist(h):
            return {
                k: [t.clone() for t in v] if isinstance(v, deque) else v.clone()
                for k, v in h.items()
            }

        return {
            "lr": self.lr,
            "history_size": self.history_size,
            "batch_size": self.batch_size,
            "shared_param_names": list(self.shared_param_names),
            "clip_value": self.clip_value,
            "clip_norm": self.clip_norm,
            "max_step": self.max_step,
            "curvature_eps": self.curvature_eps,
            "s_hist": dump_hist(self.s_hist),
            "y_hist": dump_hist(self.y_hist),
            "hist_count": {k: v.clone() for k, v in self.hist_count.items()},
            "prev_params": {k: v.clone() for k, v in self.prev_params.items()},
            "prev_grads": {k: v.clone() for k, v in self.prev_grads.items()},
            "shared_param_lr_multiplier": self.shared_param_lr_multiplier,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the state written by `state_dict`, including checkpoints from the older per-member deque layout."""
        self.lr = state_dict["lr"]
        self.history_size = state_dict["history_size"]
        self.batch_size = state_dict["batch_size"]
        self.shared_param_names = set(state_dict["shared_param_names"])
        self.clip_value = state_dict.get("clip_value")
        self.clip_norm = state_dict.get("clip_norm")
        self.max_step = state_dict.get("max_step", self.max_step)
        self.curvature_eps = state_dict.get("curvature_eps", self.curvature_eps)
        self.shared_param_lr_multiplier = state_dict.get("shared_param_lr_multiplier", 1.0)
        self.prev_params = {k: v.clone() for k, v in state_dict["prev_params"].items()}
        self.prev_grads = {k: v.clone() for k, v in state_dict["prev_grads"].items()}

        counts = state_dict.get("hist_count", {})
        self.s_hist, self.y_hist, self.hist_count = {}, {}, {}
        for k, v in state_dict["s_hist"].items():
            yv = state_dict["y_hist"][k]
            if k in self.shared_param_names:
                self.s_hist[k] = deque((t.clone() for t in v), maxlen=self.history_size)
                self.y_hist[k] = deque((t.clone() for t in yv), maxlen=self.history_size)
            elif isinstance(v, torch.Tensor):
                self.s_hist[k] = v.clone()
                self.y_hist[k] = yv.clone()
                self.hist_count[k] = counts[k].clone()
            else:
                # A checkpoint written by the per-member deque version: rebuild
                # the right-aligned tensors from the lists.
                self._init_batched_history(k, self.prev_params[k])
                for i, lst in enumerate(v):
                    n = min(len(lst), self.history_size)
                    for j, (st, yt) in enumerate(zip(lst[-n:], yv[i][-n:])):
                        self.s_hist[k][i, self.history_size - n + j] = st
                        self.y_hist[k][i, self.history_size - n + j] = yt
                    self.hist_count[k][i] = n
