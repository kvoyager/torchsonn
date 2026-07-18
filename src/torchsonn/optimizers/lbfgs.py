import torch
from collections import deque
from collections.abc import Iterable
from typing import Any

from torchsonn.optimizers.base import BaseOptimizer, LRLike


class BatchedLBFGS(BaseOptimizer):
    def __init__(
        self,
        params: dict[str, torch.Tensor],
        shared_param_names: Iterable[str],
        lr: LRLike,
        history_size: int = 10,
        clip_value: float | None = None,
        clip_norm: float | None = None,
        shared_param_lr_multiplier: float = 1.0,
    ) -> None:
        """
        params: dict of batched tensors (shape [B, ...])
        shared_param_names: iterable of keys that are shared parameters
        lr: torch.Tensor of shape (B,) or scalar
        history_size: number of (s,y) correction pairs to keep
        clip_value / clip_norm: forwarded to BaseOptimizer.gradient_clipping
        """
        super().__init__(shared_param_names, lr, clip_value, clip_norm, shared_param_lr_multiplier)
        self.history_size = history_size

        first = next(iter(params.values()))
        self.batch_size = first.shape[0]

        # per-key storage:
        # - for non-shared keys: a list of deques, one deque per model index
        # - for shared keys: a single deque used for the shared parameter history
        self.s_hist = {}
        self.y_hist = {}
        for k in params.keys():
            if k in self.shared_param_names:
                self.s_hist[k] = deque(maxlen=history_size)
                self.y_hist[k] = deque(maxlen=history_size)
            else:
                self.s_hist[k] = [deque(maxlen=history_size) for _ in range(self.batch_size)]
                self.y_hist[k] = [deque(maxlen=history_size) for _ in range(self.batch_size)]

        # prev params/grads for computing s = p - p_prev, y = g - g_prev
        self.prev_params = {k: v.detach().clone() for k, v in params.items()}
        self.prev_grads = {k: torch.zeros_like(v) for k, v in params.items()}

    def _flatten(self, t: torch.Tensor) -> torch.Tensor:
        """Flatten a parameter tensor preserving device/dtype: input shape [B, ...] -> [B, P]."""
        B = t.shape[0]
        return t.reshape(B, -1)

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

        for k in params.keys():
            p = params[k]
            g = grads[k]

            # Apply gradient clipping (no-op when clip_value / clip_norm are None).
            g = self.gradient_clipping(g)

            p_flat = self._flatten(p)
            g_flat = self._flatten(g)
            prev_p_flat = self._flatten(self.prev_params[k])
            prev_g_flat = self._flatten(self.prev_grads[k])

            Pk = p_flat.shape[1]

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
                    self.s_hist[k].append(s.clone())
                    self.y_hist[k].append(y.clone())

                d = self._two_loop_recursion_single(self.s_hist[k], self.y_hist[k], g_1d)

                if isinstance(self.lr, torch.Tensor):
                    lr_mean = self.lr.index_select(0, idx).mean().item() * self.shared_param_lr_multiplier
                else:
                    lr_mean = float(self.lr) * self.shared_param_lr_multiplier

                new_params[k] = (p_1d - lr_mean * d).reshape_as(p)

                self.prev_params[k] = p.detach().clone()
                # Save the mean gradient (not the full batched tensor) so the
                # shape stays (*param_shape,) and matches p on the next step.
                self.prev_grads[k] = g_1d.reshape_as(p).detach().clone()

            else:
                s = p_flat - prev_p_flat
                y = g_flat - prev_g_flat

                for i in range(self.batch_size):
                    if not active_mask[i]:
                        continue
                    if s[i].abs().sum() > 1e-12:
                        self.s_hist[k][i].append(s[i].detach().clone())
                        self.y_hist[k][i].append(y[i].detach().clone())

                updates = torch.zeros_like(p_flat, device=device)
                if isinstance(self.lr, torch.Tensor):
                    lr_per = self.lr.view(-1, 1)
                else:
                    lr_per = None

                for i in range(self.batch_size):
                    if not active_mask[i]:
                        continue
                    g_i = g_flat[i]
                    d_i = self._two_loop_recursion_single(self.s_hist[k][i], self.y_hist[k][i], g_i)
                    if lr_per is None:
                        lr_i = float(self.lr)
                    else:
                        lr_i = float(lr_per[i].item())
                    updates[i] = lr_i * d_i

                update_mask = active_mask.view(-1, *([1] * (p.dim() - 1)))
                p_new_flat = torch.where(update_mask.view(self.batch_size, 1), p_flat - updates, p_flat)

                new_params[k] = p_new_flat.reshape_as(p)

                self.prev_params[k] = p.detach().clone()
                self.prev_grads[k] = g.detach().clone()

        return new_params

    def state_dict(self) -> dict[str, Any]:
        """Snapshot of curvature history + prev params/grads + lr so a resumed run
        keeps the L-BFGS approximation rather than starting from identity."""
        def dump_deque(d):
            return [t.clone() for t in d]

        def dump_hist(h):
            return {
                k: dump_deque(v) if isinstance(v, deque) else [dump_deque(dq) for dq in v]
                for k, v in h.items()
            }

        return {
            "lr": self.lr,
            "history_size": self.history_size,
            "batch_size": self.batch_size,
            "shared_param_names": list(self.shared_param_names),
            "clip_value": self.clip_value,
            "clip_norm": self.clip_norm,
            "s_hist": dump_hist(self.s_hist),
            "y_hist": dump_hist(self.y_hist),
            "prev_params": {k: v.clone() for k, v in self.prev_params.items()},
            "prev_grads": {k: v.clone() for k, v in self.prev_grads.items()},
            "shared_param_lr_multiplier": self.shared_param_lr_multiplier,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.lr = state_dict["lr"]
        self.history_size = state_dict["history_size"]
        self.batch_size = state_dict["batch_size"]
        self.shared_param_names = set(state_dict["shared_param_names"])
        self.clip_value = state_dict.get("clip_value")
        self.clip_norm = state_dict.get("clip_norm")
        self.shared_param_lr_multiplier = state_dict.get("shared_param_lr_multiplier", 1.0)

        def load_hist(saved):
            out = {}
            for k, v in saved.items():
                if k in self.shared_param_names:
                    dq = deque(maxlen=self.history_size)
                    for t in v:
                        dq.append(t.clone())
                    out[k] = dq
                else:
                    out[k] = [
                        deque((t.clone() for t in lst), maxlen=self.history_size)
                        for lst in v
                    ]
            return out

        self.s_hist = load_hist(state_dict["s_hist"])
        self.y_hist = load_hist(state_dict["y_hist"])
        self.prev_params = {k: v.clone() for k, v in state_dict["prev_params"].items()}
        self.prev_grads = {k: v.clone() for k, v in state_dict["prev_grads"].items()}