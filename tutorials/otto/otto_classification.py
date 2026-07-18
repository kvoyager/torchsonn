"""Otto Group Product Classification with SONN.

Downloads the Otto Group dataset from OpenML (ARFF format), performs a
stratified 70/15/15 split, trains a SONN multi-class classifier followed by
an out_proj fine-tuning pass, and reports multi-class log loss on all splits.

Run from any directory:
    python tutorials/otto/otto_classification.py
"""

import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

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
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torchsonn.data.dataset import SONNDataset
from torchsonn.data.preprocessing import train_preprocessing
from torchsonn.logger import setup_logger
from torchsonn.model import SONN
from torchsonn.trainer import Trainer

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
TUTORIAL_DIR = _script_dir
DATA_DIR = TUTORIAL_DIR / "data"
CKPT_DIR = TUTORIAL_DIR / "checkpoints"
LOG_PATH = TUTORIAL_DIR / "otto_classification.log"

DATA_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

logger = setup_logger(str(LOG_PATH))

# ---------------------------------------------------------------------------
# Download & parse dataset
# ---------------------------------------------------------------------------
DATASET_URL = "https://api.openml.org/data/download/22116516/dataset"
CACHE_FILE = DATA_DIR / "otto_dataset.arff"


def _progress(block_num, block_size, total_size):
    if total_size > 0:
        pct = min(100, block_num * block_size * 100 // total_size)
        print(f"\rDownloading... {pct}%", end="", flush=True)


def load_dataset():
    if not CACHE_FILE.exists():
        print(f"Downloading from {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, CACHE_FILE, reporthook=_progress)
        print(f"\nSaved to {CACHE_FILE} ({CACHE_FILE.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Using cached file: {CACHE_FILE} ({CACHE_FILE.stat().st_size / 1e6:.1f} MB)")

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

    print(f"X: {X.shape}   classes: {sorted(set(labels))}")
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    X, y, feature_names = load_dataset()

    # -- Config --------------------------------------------------------------
    with initialize_config_dir(version_base="1.3", config_dir=str(TUTORIAL_DIR)):
        config = compose(config_name="otto")
    config.train.checkpoint_dir = str(CKPT_DIR)
    logger.info(OmegaConf.to_yaml(config))

    # -- Split ---------------------------------------------------------------
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, test_size=0.30, random_state=config.train.seed, stratify=y
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_rem, y_rem, test_size=0.50, random_state=config.train.seed, stratify=y_rem
    )
    print(f"train: {len(X_train)}   dev: {len(X_dev)}   test: {len(X_test)}")

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

    # -- Class weights (inverse frequency, normalized to sum to num_classes) --
    counts = np.bincount(y_train)
    cw = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
    print("class weights:", cw.numpy().round(3))

    preprocessing = torch.nn.Sequential(torch.nn.LayerNorm(x_train.shape[1], elementwise_affine=False))
    # -- Model ---------------------------------------------------------------
    model = SONN(config, d_model=x_train.shape[1], feature_names=feature_names, preprocessing=preprocessing)
    model = model.to(config.train.device)
    print(model)

    # -- Train SONN layers ---------------------------------------------------
    trainer = Trainer(config, feature_names=feature_names, class_weights=cw)
    Trainer.set_seed(config.train.seed)
    trainer.train(model, train_dl, dev_dl, test_dl, resume=config.resume)

    # -- Train out_proj ------------------------------------------------------
    if model.out_proj is not None:
        print("\nTraining out_proj...")
        trainer.train_out_proj(model, train_dl, dev_dl)

    # -- Load best checkpoint & evaluate -------------------------------------
    trainer.load_model_checkpoint(model, config.train.device)

    print("\n--- Results ---")
    CLASS_NAMES = [f"Class_{i + 1}" for i in range(config.model.num_classes)]

    def evaluate(dl, label):
        out, tgt = trainer.infer(model, dl, verbose=False)
        proba   = torch.exp(out).cpu().numpy()
        true_np = tgt.cpu().numpy()
        ll  = log_loss(true_np, proba)
        acc = (proba.argmax(axis=1) == true_np).mean()
        print(f"  {label:<8}  log_loss = {ll:.4f}   accuracy = {acc:.4f}")
        return proba, true_np

    train_proba, train_true = evaluate(train_dl, "Train")
    dev_proba,   dev_true   = evaluate(dev_dl,   "Dev")
    test_proba,  test_true  = evaluate(test_dl,  "Test")

    print("\n--- Test classification report ---")
    test_pred = test_proba.argmax(axis=1)
    print(classification_report(test_true, test_pred, target_names=CLASS_NAMES))

    print("--- Confusion matrix (test) ---")
    cm = confusion_matrix(test_true, test_pred)
    col_w = max(len(n) for n in CLASS_NAMES) + 1
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:<{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))


if __name__ == "__main__":
    main()
