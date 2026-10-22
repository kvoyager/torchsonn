"""Otto Group Product Classification with SONN.

Downloads the Otto Group dataset from OpenML (ARFF format), performs a
stratified 70/15/15 split, trains a SONN multi-class classifier followed by
an out_proj fine-tuning pass, and reports multi-class log loss on all splits.

Each run writes its checkpoints and its train.log to a time-stamped folder,
tutorials/otto/checkpoints/YYYY-MM-DD-HH-MM-SS/. With `resume: true` in the
config the script continues the newest run folder there instead. The log
holds the trainer's records, each progress bar's final line and the results.

Run from any directory:
    python tutorials/otto/otto_classification.py
    python tutorials/otto/otto_classification.py --config-name otto_tuned
    python tutorials/otto/otto_classification.py train.seed=43 train.max_layer_count=1

`--config-name` (or `-cn`) picks another YAML file in this folder, and
`key=value` arguments override any schema field, as with `@hydra.main`. The
script reads its config with Hydra's compose API, so it parses those
arguments itself.
"""

import logging
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm as _TqdmBase

# ---------------------------------------------------------------------------
# Path bootstrap — makes `torchsonn` importable regardless of cwd
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir
while not (_project_root / "src").is_dir() and _project_root != _project_root.parent:
    _project_root = _project_root.parent
_src = _project_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import torchsonn.config  # noqa: F401 — registers SONNConfig with Hydra ConfigStore
import torchsonn.trainer as _trainer_module
import torchsonn.training.evaluation as _evaluation_module
import torchsonn.training.finetune as _finetune_module
import torchsonn.training.head_fit as _head_fit_module
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torchsonn.data.dataset import SONNDataset
from torchsonn.data.preprocessing import train_preprocessing
from torchsonn.logger import RunLogHandler, setup_logger
from torchsonn.model import SONN
from torchsonn.trainer import Trainer

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
TUTORIAL_DIR = _script_dir
DATA_DIR = TUTORIAL_DIR / "data"
CKPT_DIR = TUTORIAL_DIR / "checkpoints"

DATA_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

# Console handler; each run's log is the train.log in its run folder.
setup_logger()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress bars into the run log
# ---------------------------------------------------------------------------
# The trainer draws its progress bars on the console only, so their final
# state (100%, timings, the loss postfix) would never reach train.log. The
# bars below also write their final line to the run log when they close,
# verbatim, between the trainer's records.
def _log_bar_line(line: str) -> None:
    """Append one finished progress bar line to the open run log.

    Written under the handler's lock, so it cannot interleave with a log
    record. Does nothing while no run log is open.
    """
    fh = next((h for h in logging.getLogger().handlers if isinstance(h, RunLogHandler)), None)
    if fh is None or not line:
        return
    fh.acquire()
    try:
        fh.stream.write(line + "\n")
        fh.flush()
    finally:
        fh.release()


class _LoggingTqdm(_TqdmBase):
    """tqdm that also writes its final line to the run log on close().

    `format_meter(**format_dict)` is the line tqdm draws on the console, so
    the log shows the bar as it was last seen. Disabled bars log nothing.
    """

    def close(self) -> None:
        if not self.disable:
            try:
                _log_bar_line(self.format_meter(**self.format_dict))
            except Exception:
                # Never let the log mirror break a training run.
                pass
        super().close()


# The trainer's modules each did `from tqdm import tqdm`; rebinding that name
# makes their bars _LoggingTqdm.
for _module in (_trainer_module, _evaluation_module, _finetune_module, _head_fit_module):
    _module.tqdm = _LoggingTqdm


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
USAGE = "otto_classification.py [--config-name NAME] [key=value ...]"


def parse_args(argv: list[str]) -> tuple[str, list[str]]:
    """Split the arguments into a config name and a list of Hydra overrides.

    Accepts the spellings `@hydra.main` does: `--config-name X`,
    `--config-name=X`, `-cn X` and `-cn=X`. An argument with `=` in it, or
    one starting with `+` or `~`, is an override and goes to `compose` as it
    is. A trailing `.yaml` on the name is dropped.
    """
    name, overrides, i = "otto", [], 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--config-name", "-cn"):
            if i + 1 >= len(argv):
                raise SystemExit(f"{arg} needs a value. Usage: {USAGE}")
            name, i = argv[i + 1], i + 2
            continue
        if arg.startswith(("--config-name=", "-cn=")):
            name = arg.split("=", 1)[1]
        elif "=" in arg or arg.startswith(("+", "~")):
            overrides.append(arg)
        else:
            raise SystemExit(f"unknown argument {arg!r}. Usage: {USAGE}")
        i += 1
    for ext in (".yaml", ".yml"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name, overrides


# ---------------------------------------------------------------------------
# Download & parse dataset
# ---------------------------------------------------------------------------
DATASET_URL = "https://api.openml.org/data/download/22116516/dataset"
CACHE_FILE = DATA_DIR / "otto_dataset.arff"


def _progress(block_num, block_size, total_size):
    # Console only: the line is redrawn with \r, which a log file would keep
    # as one record per chunk.
    if total_size > 0:
        pct = min(100, block_num * block_size * 100 // total_size)
        print(f"\rDownloading... {pct}%", end="", flush=True)


def load_dataset():
    if not CACHE_FILE.exists():
        logger.info("Downloading from %s", DATASET_URL)
        urllib.request.urlretrieve(DATASET_URL, CACHE_FILE, reporthook=_progress)
        print()  # end the \r progress line before the next record
        logger.info("Saved to %s (%.1f MB)", CACHE_FILE, CACHE_FILE.stat().st_size / 1e6)
    else:
        logger.info("Using cached file: %s (%.1f MB)", CACHE_FILE, CACHE_FILE.stat().st_size / 1e6)

    with open(CACHE_FILE) as f:
        lines = f.readlines()

    attr_names, data_start = [], None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.upper() == "@DATA":
            data_start = i + 1
            break
        if s.upper().startswith("@ATTRIBUTE"):
            attr_names.append(s.split()[1])

    assert data_start is not None, "@DATA marker not found"

    rows, labels = [], []
    for line in lines[data_start:]:
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        parts = s.split(",")
        rows.append(list(map(float, parts[1:-1])))
        labels.append(int(parts[-1].strip().split("_")[1]) - 1)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    feature_names = attr_names[1:-1]

    logger.info("X: %s   classes: %s", X.shape, sorted(set(labels)))
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None):
    config_name, overrides = parse_args(sys.argv[1:] if argv is None else argv)
    X, y, feature_names = load_dataset()

    # -- Config --------------------------------------------------------------
    with initialize_config_dir(version_base="1.3", config_dir=str(TUTORIAL_DIR)):
        config = compose(config_name=config_name, overrides=overrides)
    if config.get("finetune_drop_head", False):
        raise SystemExit("finetune_drop_head is not supported by this script: the end-to-end pass keeps the head.")
    config.train.checkpoint_dir = str(CKPT_DIR)
    logger.info("Config %r%s", config_name, f" with overrides {overrides}" if overrides else "")
    logger.info(OmegaConf.to_yaml(config))

    # -- Split ---------------------------------------------------------------
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, test_size=0.30, random_state=config.train.seed, stratify=y
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_rem, y_rem, test_size=0.50, random_state=config.train.seed, stratify=y_rem
    )
    logger.info("train: %d   dev: %d   test: %d", len(X_train), len(X_dev), len(X_test))

    # -- Preprocessing -------------------------------------------------------
    x_train, y_train = train_preprocessing(X_train, y_train, feature_names)
    x_dev,   y_dev   = train_preprocessing(X_dev,   y_dev,   feature_names)
    x_test,  y_test  = train_preprocessing(X_test,  y_test,  feature_names)

    x_train = np.log1p(x_train)
    x_dev = np.log1p(x_dev)
    x_test = np.log1p(x_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_dev   = scaler.transform(x_dev)
    x_test  = scaler.transform(x_test)

    bs = config.train.batch_size
    train_dl = DataLoader(SONNDataset(x_train, y_train), batch_size=bs, shuffle=True)
    dev_dl   = DataLoader(SONNDataset(x_dev,   y_dev),   batch_size=bs)
    test_dl  = DataLoader(SONNDataset(x_test,  y_test),  batch_size=bs)

    # -- No class weights ----------------------------------------------------
    # The score is the unweighted log loss. Inverse-frequency class weights
    # would make the model predict a flat class prior, over-predicting the
    # small classes, and raise the log loss.

    preprocessing = torch.nn.Sequential(torch.nn.LayerNorm(x_train.shape[1], elementwise_affine=False))
    # -- Model ---------------------------------------------------------------
    # Seed first: everything random from here on, the model included,
    # follows train.seed.
    Trainer.set_seed(config.train.seed)
    model = SONN(config, d_model=x_train.shape[1], feature_names=feature_names, preprocessing=preprocessing)
    model = model.to(config.train.device)
    logger.info("Model:\n%s", model)

    # -- Train SONN layers ---------------------------------------------------
    trainer = Trainer(config, feature_names=feature_names)
    trainer.train(model, train_dl, dev_dl, test_dl, resume=config.resume)

    # -- Train out_proj ------------------------------------------------------
    if model.out_proj is not None:
        logger.info("Training out_proj...")
        trainer.train_out_proj(model, train_dl, dev_dl)

    # -- Load best checkpoint ------------------------------------------------
    trainer.load_model_checkpoint(model, config.train.device)

    # -- Optional end-to-end pass ----------------------------------------------
    # Every pass so far fits one part of the network with the rest frozen;
    # this one trains every parameter at once, the head included, with
    # train.finetune_train. It starts from the best checkpointed weights and
    # saves the model it ends with as the run's model_last.ckpt.
    if config.get("finetune_end_to_end", False):
        if config.get("finetune_prune_first", False):
            trainer.prune(model)
            logger.info("Pruned to the read path before the end-to-end pass: %d layers, %d neurons",
                        len(model.layers), sum(len(layer) for layer in model.layers))
        logger.info("End-to-end pass over all parameters")
        trainer.train_finetune(model, train_dl, dev_dl)

    logger.info("--- Results ---")
    CLASS_NAMES = [f"Class_{i + 1}" for i in range(config.model.num_classes)]

    def evaluate(dl, label):
        out, tgt = trainer.infer(model, dl, verbose=False)
        proba   = torch.exp(out).cpu().numpy()
        true_np = tgt.cpu().numpy()
        ll  = log_loss(true_np, proba)
        acc = (proba.argmax(axis=1) == true_np).mean()
        logger.info("  %-8s  log_loss = %.4f   accuracy = %.4f", label, ll, acc)
        return proba, true_np

    evaluate(train_dl, "Train")
    evaluate(dev_dl,   "Dev")
    test_proba, test_true = evaluate(test_dl, "Test")
    logger.info("Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    test_pred = test_proba.argmax(axis=1)
    logger.info("--- Test classification report ---\n%s",
                classification_report(test_true, test_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(test_true, test_pred)
    col_w = max(len(n) for n in CLASS_NAMES) + 1
    rows = [" " * col_w + "".join(f"{n:>{col_w}}" for n in CLASS_NAMES)]
    for i, row in enumerate(cm):
        rows.append(f"{CLASS_NAMES[i]:<{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))
    logger.info("--- Confusion matrix (test) ---\n%s", "\n".join(rows))

    # -- Prune ---------------------------------------------------------------
    # Keep only the neurons that reach the head. Pruning drops neurons nothing
    # reads, so the predictions must not move.
    trainer.prune(model)
    pruned_out, _ = trainer.infer(model, test_dl, verbose=False)
    moved = np.abs(torch.exp(pruned_out).cpu().numpy() - test_proba).max()
    logger.info("Pruned to the read path: %s neurons per layer, %s parameters; "
                "max test probability change %.2e",
                [len(layer) for layer in model.layers],
                f"{sum(p.numel() for p in model.parameters()):,}", moved)

    logger.info("Run folder: %s", trainer.run_dir)


if __name__ == "__main__":
    main()
