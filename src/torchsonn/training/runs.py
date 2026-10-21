"""Run folders and checkpoint files.

A run folder is `<train.checkpoint_dir>/<YYYY-MM-DD-HH-MM-SS>[-<n>]` and
holds the step checkpoints, `model_last.ckpt` and `train.log` of one
`Trainer.train` call.
"""
import datetime
import logging
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from torchsonn.logger import attach_run_log
from torchsonn.model import SONN
from torchsonn.training import distributed
from torchsonn.training.state import StepCheckpoint

logger = logging.getLogger(__name__)

# A run folder: `YYYY-MM-DD-HH-MM-SS`, with `-<n>` for runs started in the
# same second.
RUN_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(?:-(\d+))?")
# A step checkpoint the trainer writes, with its optional `_last` copy.
STEP_CKPT_RE = re.compile(r"model_layer_(\d+)_neuron_(\d+)_step_(\d+)(_last)?\.ckpt")


def run_folders(root: str | Path) -> list[Path]:
    """The run folders in `root`, oldest first: by start time, then by the
    `-<n>` suffix as a number. Other entries are left out; empty when `root`
    does not exist."""
    root = Path(root)
    if not root.is_dir():
        return []
    runs = []
    for path in root.iterdir():
        m = RUN_NAME_RE.fullmatch(path.name)
        if m and path.is_dir():
            runs.append(((m.group(1), int(m.group(2) or 1)), path))
    return [path for _, path in sorted(runs)]


def new_run_dir(root: Path) -> Path:
    """Create `root/<YYYY-MM-DD-HH-MM-SS>`, adding `-2`, `-3`, ... when that name is taken."""
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    n = 1
    while True:
        run_dir = root / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            n += 1


def resolve_run(root: Path, run: str | Path) -> Path:
    """A run folder given by its name in `root`, or by its path."""
    for candidate in (root / str(run), Path(run)):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"no run folder {str(run)!r} in {root}")


def start_run(root: Path, resume: bool | str) -> tuple[Path, bool]:
    """Open the run folder of a `train` call and attach its `train.log`.

    Returns the folder and whether an existing run is resumed. With several
    processes, rank 0 picks or creates the folder and sends its path to the
    others, and only rank 0 writes `train.log`.
    """
    rank0 = not distributed.is_dist() or dist.get_rank() == 0
    choice: list[Any] = [None, False]
    if rank0:
        run_dir, resumed = None, False
        if isinstance(resume, str):
            run_dir, resumed = resolve_run(root, resume), True
        elif resume:
            runs = [r for r in run_folders(root) if step_checkpoints(r, include_last=True)]
            if runs:
                run_dir, resumed = runs[-1], True
        if run_dir is None:
            run_dir = new_run_dir(root)
        choice = [str(run_dir), resumed]
    if distributed.is_dist():
        dist.broadcast_object_list(choice, src=0)
    run_dir = Path(choice[0])
    resumed = bool(choice[1])
    if rank0:
        attach_run_log(run_dir / "train.log", append=resumed)
    if resumed:
        logger.info("Resuming the run in %s", run_dir)
    else:
        if resume:
            logger.info("No run to resume in %s; starting a new run", root)
        logger.info("Run folder: %s", run_dir)
    return run_dir, resumed


def step_checkpoints(folder: Path, include_last: bool = False,
                     layer_idx: int | None = None) -> list[Path]:
    """The step-checkpoint files directly in `folder`.

    Only regular files named `model_layer_<L>_neuron_<N>_step_<S>.ckpt`
    count, and their `_last` copies when `include_last` is set;
    `layer_idx` keeps one layer's. Everything else in the folder, such as
    `train.log`, `model_last.ckpt` or a subfolder, is left out.
    """
    if not folder.is_dir():
        return []
    files = []
    for path in folder.iterdir():
        m = STEP_CKPT_RE.fullmatch(path.name)
        if not m or not path.is_file():
            continue
        if m.group(4) and not include_last:
            continue
        if layer_idx is not None and int(m.group(1)) != layer_idx:
            continue
        files.append(path)
    return files


def parse_checkpoint_step(fname: str) -> tuple[int, int, int]:
    """`(layer, neuron_model, step)` of a step checkpoint file name, or
    `(-1, -1, -1)` when the name does not match."""
    m = re.search(r"model_layer_(\d+)_neuron_(\d+)_step_(\d+)(?:_last)?\.ckpt$", fname)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    # Same arity as the match path so max()/sorted() compare apples to apples;
    # the (-1, -1, -1) sentinel sorts strictly below any real (layer, neuron,
    # step) tuple, so stray non-matching files never get picked as "latest".
    return -1, -1, -1


def latest_step_checkpoint(folder: Path) -> dict[str, Any] | None:
    """The contents of the latest step checkpoint in `folder`, `_last` copies
    included, or None when it holds none."""
    checkpoints = step_checkpoints(folder, include_last=True)
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda path: parse_checkpoint_step(path.name))
    return torch.load(latest, weights_only=False)


def save_step_checkpoint(run_dir: Path, ckpt: StepCheckpoint, suffix: str = "") -> None:
    """Write `model_layer_<L>_neuron_<N>_step_<S>[_<suffix>].ckpt` with the
    snapshot and the model config, then keep the `train.keep_last_n` most
    recent step checkpoints of `run_dir`."""
    if suffix:
        suffix = "_" + suffix
    checkpoint_filename = f"model_layer_{ckpt.layer_idx}_neuron_{ckpt.neuron_model_idx}_step_{ckpt.global_step}{suffix}.ckpt"
    d = ckpt.to_dict()
    d["config"] = ckpt.model.param
    torch.save(d, run_dir / checkpoint_filename)
    cleanup_checkpoints(run_dir, keep_last_n=ckpt.model.param.train.keep_last_n)


def model_run_dir(root: Path, run: str | Path | None, current: Path | None) -> Path:
    """The run folder to load `model_last.ckpt` from: `run` (a name in `root`
    or a path), else `current`, else the newest run folder in `root` that
    holds one."""
    if run is not None:
        return resolve_run(root, run)
    if current is not None:
        return current
    saved = [r for r in run_folders(root) if (r / "model_last.ckpt").is_file()]
    if not saved:
        raise FileNotFoundError(f"no run folder with a model_last.ckpt in {root}")
    return saved[-1]


def load_model(model: SONN, run_dir: Path, device: torch.device | str) -> None:
    """Restore `model` from `run_dir/model_last.ckpt` and move it to `device`."""
    checkpoint_data = torch.load(run_dir / "model_last.ckpt", weights_only=False)
    # restore model architecture
    model.restore_from_checkpoint_metadata(checkpoint_data["model"])
    # load weights
    model.load_state_dict(checkpoint_data["model"], strict=False)
    model.to(device=device)


def save_model(model: SONN, run_dir: Path) -> None:
    """Save the model's state dict as `run_dir/model_last.ckpt`."""
    torch.save({
        'model': model.state_dict(),
    }, run_dir / "model_last.ckpt")


def cleanup_checkpoints(folder: Path, keep_last_n: int = 10) -> None:
    """Delete the oldest step checkpoints in `folder`, keeping the
    `keep_last_n` most recent; `_last` copies and other files stay."""
    checkpoints = sorted(step_checkpoints(folder),
                         key=lambda path: parse_checkpoint_step(path.name))
    for path in checkpoints[:max(0, len(checkpoints) - keep_last_n)]:
        path.unlink()


def cleanup_layer_checkpoints(folder: Path, layer_idx: int) -> None:
    """Delete every step checkpoint of one layer in `folder`, `_last` copies included."""
    for path in step_checkpoints(folder, include_last=True, layer_idx=layer_idx):
        path.unlink()
