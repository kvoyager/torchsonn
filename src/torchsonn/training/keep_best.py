"""The best evaluated weights of a head fit, per-layer fine-tune or
end-to-end pass (`keep_best_weights`, `best_weights_copy`)."""
import logging
import os
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from torchsonn.training import config_checks, distributed

logger = logging.getLogger(__name__)


class BestWeights:
    """A pass's evaluations, and the weights it ends with.

    A head fit, per-layer fine-tune or end-to-end pass calls `update` after
    every evaluation, its last step included. With `keep`
    (`OutProjTrainConfig.keep_best_weights`) the parameters are copied
    whenever the loss is the lowest so far (strictly lower, without the
    early stop's `early_stop_min_delta`; NaN never is), and `restore`
    copies the best ones back in place, so the tensors a captured CUDA
    graph holds stay valid, and logs which step's weights the pass ends
    with. Without it nothing is copied and the pass ends on its last step.
    Where the copy is kept (`OutProjTrainConfig.best_weights_copy`):

    - 'device': tensors next to the parameters (GPU memory on CUDA);
    - 'cpu': host tensors, pinned when the parameters are on CUDA;
    - 'disk': the file `path`, rewritten through a temporary name on every
      new best and deleted by `restore`.

    `final_loss` is the evaluated loss of the weights the pass ends with.
    With no parameters (a head fitted over frozen survivors, which is then
    discarded) only the losses are tracked, and `final_loss` is the lowest.
    """

    def __init__(self, params: list[torch.Tensor], keep: bool, where: str,
                 path: Path | None, label: str, split: str = "dev") -> None:
        if where not in config_checks.BEST_WEIGHTS_COPIES:
            raise ValueError(
                f"best_weights_copy={where!r}; expected 'device', 'cpu' or 'disk'.")
        if keep and where == "disk" and params and path is None:
            raise ValueError("best_weights_copy='disk' needs a file path.")
        self.params = list(params)
        self.keep = bool(keep)
        self.where, self.path, self.label, self.split = where, path, label, split
        self.best_loss = float("inf")
        self.best_step: int | None = None
        self.last_loss = float("nan")
        self.last_step: int | None = None
        self._copies: list[torch.Tensor] | None = None

    def update(self, step: int, loss: float) -> None:
        """Record the evaluation at `step`; copy the parameters on a new best."""
        self.last_step, self.last_loss = step, loss
        if not loss < self.best_loss:
            return
        self.best_step, self.best_loss = step, loss
        if not self.keep or not self.params:
            return
        with torch.no_grad():
            if self.where == "disk":
                tmp = self.path.with_name(self.path.name + ".tmp")
                torch.save([p.detach().cpu() for p in self.params], tmp)
                os.replace(tmp, self.path)
                return
            if self._copies is None:
                if self.where == "cpu":
                    pin = any(p.is_cuda for p in self.params)
                    self._copies = [torch.empty(p.shape, dtype=p.dtype, pin_memory=pin)
                                    for p in self.params]
                else:
                    self._copies = [torch.empty_like(p) for p in self.params]
            for c, p in zip(self._copies, self.params):
                c.copy_(p.detach())

    @property
    def final_loss(self) -> float:
        """The evaluated loss of the weights the pass ends with (inf if there is none)."""
        if self.keep or not self.params:
            return self.best_loss
        return self.last_loss if self.last_step is not None else float("inf")

    def restore(self) -> None:
        """Copy the best evaluation's parameters back, unless the last one is the best."""
        if not self.keep or not self.params or self.best_step is None:
            return
        if self.best_step == self.last_step:
            logger.info(f"{self.label}: kept the last step's weights (step {self.last_step}, "
                        f"the lowest {self.split} loss, {self.best_loss:.4f})")
        else:
            if self.where == "disk":
                saved = torch.load(self.path, map_location="cpu", weights_only=True)
            else:
                saved = self._copies
            with torch.no_grad():
                for p, c in zip(self.params, saved):
                    p.copy_(c)
            logger.info(f"{self.label}: kept the weights of step {self.best_step} "
                        f"({self.split} loss {self.best_loss:.4f}); the last evaluation, "
                        f"step {self.last_step}, gave {self.last_loss:.4f}")
        self._copies = None
        if self.where == "disk":
            self.path.unlink(missing_ok=True)


def record(params: list[torch.Tensor], cfg: Any, name: str, label: str, split: str = "dev", *,
           run_dir: Callable[[], Path]) -> BestWeights:
    """A pass's evaluation record; it copies the best weights under `cfg.keep_best_weights`.

    The copy is kept where `cfg.best_weights_copy` says. Under 'disk' the
    file is `best_<name>.ckpt` in the run folder (with `_rank<N>` in a
    distributed run, where every rank keeps its own); `run_dir` is called
only then, so no run folder is made for the other places.
    """
    keep = bool(getattr(cfg, "keep_best_weights", False))
    where = str(getattr(cfg, "best_weights_copy", "device"))
    path = None
    if keep and where == "disk" and params:
        rank = f"_rank{dist.get_rank()}" if distributed.is_dist() else ""
        path = run_dir() / f"best_{name}{rank}.ckpt"
    return BestWeights(params, keep, where, path, label, split)
