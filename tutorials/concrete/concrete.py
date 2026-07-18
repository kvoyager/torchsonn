"""UCI Concrete Compressive Strength regression — Hydra entry point.

Fetches the 1030-sample UCI Concrete dataset (Yeh, 1998), trains a SONN
regressor with `linear_cov + quadratic` reference functions, and reports
MSE / MAE / R² on a held-out 15 % test split.

Run from the repo root:
    python -m tutorials.concrete.concrete

Hydra overrides work on every config key. Examples:
    python -m tutorials.concrete.concrete resume=true
    python -m tutorials.concrete.concrete train.optimizer.optimizer_params.lr=5e-2
    python -m tutorials.concrete.concrete model.ref_functions='[polyquad]' train.max_layer_count=3
    python -m tutorials.concrete.concrete hydra.run.dir=/tmp/concrete_run
"""
import urllib.request
from io import BytesIO
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from torchsonn.data.dataset import SONNDataset
from torchsonn.logger import setup_logger
from torchsonn.model import SONN
from torchsonn.plot_model import PlotModel
from torchsonn.trainer import Trainer


# --- Dataset access ----------------------------------------------------------
# The Concrete dataset ships as an Excel file under the UCI legacy ML-DB URL.
# Reading .xls reliably from disk requires `xlrd`, which isn't a hard
# dependency of this repo, so we instead pull from a stable CSV mirror of the
# same Yeh-1998 data and cache locally.  The CSV variant is byte-identical to
# the canonical UCI XLS export — same 1030 rows, same 9 columns, same units.
DATASET_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"
)
# UCI's column-name convention (Yeh 1998). Re-applied after load to give the
# trained SONN nice feature names regardless of how the mirror named them.
FEATURE_NAMES = [
    "Cement",                  # kg / m^3
    "BlastFurnaceSlag",        # kg / m^3
    "FlyAsh",                  # kg / m^3
    "Water",                   # kg / m^3
    "Superplasticizer",        # kg / m^3
    "CoarseAggregate",         # kg / m^3
    "FineAggregate",           # kg / m^3
    "Age",                     # days
]
TARGET_NAME = "CompressiveStrength"   # MPa


def _load_concrete(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Download (once) + parse the Concrete dataset.

    Returns:
      X: (1030, 8) float32 feature matrix in UCI column order.
      y: (1030,)    float32 target (compressive strength in MPa).
      feature_names: canonical UCI names, attached to the model so
                     SONN.get_selected_features() prints something readable.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "concrete.csv"
    if not csv_path.exists():
        print(f"Downloading {DATASET_URL} → {csv_path.name}")
        with urllib.request.urlopen(DATASET_URL, timeout=60) as resp:
            payload = resp.read()
        csv_path.write_bytes(payload)
    else:
        print(f"Using cached file: {csv_path}")

    df = pd.read_csv(csv_path)
    assert df.shape[1] == 9, f"expected 9 columns (8 features + target), got {df.columns.tolist()}"
    # Normalize column names — the CSV mirror uses slightly different
    # capitalization / spacing per release. The last column is always the
    # strength target by Yeh's convention.
    df.columns = FEATURE_NAMES + [TARGET_NAME]
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df[TARGET_NAME].to_numpy(dtype=np.float32)
    return X, y, list(FEATURE_NAMES)


# `config_path` resolves relative to this file. `version_base="1.3"` keeps
# Hydra's auto-chdir disabled by default, so any relative paths in the YAML
# (e.g. `checkpoint_dir: ../concrete/checkpoints`) resolve from the launch
# cwd, not from Hydra's per-run output dir.
@hydra.main(version_base="1.3", config_path=".", config_name="concrete")
def main(config: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = setup_logger(str(run_dir / "train.log"))
    logger.info("Loaded config:")
    logger.info(OmegaConf.to_yaml(config))

    # --- Load + split ---------------------------------------------------------
    cache_dir = Path(__file__).resolve().parent / "data"
    X, y, feature_names = _load_concrete(cache_dir)
    print(f"Concrete dataset: X={X.shape}, y={y.shape}, features={feature_names}")
    print(f"  target stats: min={y.min():.2f} MPa, max={y.max():.2f} MPa, mean={y.mean():.2f} MPa")

    # Age in Yeh's dataset is log-spaced by experimental design (1, 3, 7, 14,
    # 28, 56, 90, 180, 365 days). z-scoring the raw column leaves most
    # samples clumped near 0 with a long right tail at 365 days, which makes
    # any `cement * age` interaction the polynomial neurons could exploit
    # get dominated by a handful of outlier rows. log1p brings the
    # early-vs-late curing range to comparable weight, matching what
    # textbook concrete-strength models do.
    age_col = feature_names.index("Age")
    X[:, age_col] = np.log1p(X[:, age_col])

    # 80 / 20 train / test, then split that 80 % into 80 / 20 train / dev.
    # Final fractions: 64 % train, 16 % dev, 20 % test. random_state pinned
    # to 42 so the same rows land in the same split across reruns regardless
    # of train.seed (which only governs optimizer / candidate generation).
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42,
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=42,
    )
    print(f"  split: train={len(X_train)}  dev={len(X_dev)}  test={len(X_test)}")

    # --- Standardize features AND target -------------------------------------
    # Features span 3 orders of magnitude; z-score them so polynomial neurons
    # see a comparable dynamic range across inputs.
    #
    # The target is also z-scored — *critical* for SONN's regression-mode
    # selection criterion. `regularity_error(y, ŷ) = Σ(y - ŷ)² / Σy²` uses
    # the uncentered sum-of-squares in the denominator. For raw-MPa y with
    # mean≈36, Σy² is dominated by N·μ² and the entire range from
    # "no-skill" to "near-perfect" collapses into a tiny absolute band
    # (~0.003 to ~0.18). Per-layer growth heuristics (relative-improvement
    # threshold, criterion_minimum_width) can't distinguish a great fit
    # from a mediocre one, so layer growth stalls early and R² flatlines
    # near OLS. After z-scoring, `Σy² ≈ N`, the "no-skill" baseline lands
    # at exactly 1.0, and improvements are crisp.
    #
    # Predictions are inverse-transformed inside report() so the reported
    # MSE / MAE / R² are still in physical MPa units.
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train).astype(np.float32)
    X_dev   = x_scaler.transform(X_dev).astype(np.float32)
    X_test  = x_scaler.transform(X_test).astype(np.float32)

    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_dev_s   = y_scaler.transform(y_dev.reshape(-1, 1)).ravel().astype(np.float32)

    bs = int(config.train.batch_size)
    # Train / dev see standardized y; test stays in raw MPa because the
    # trainer never sees it — only the user's `report()` does, and report()
    # inverts the prediction back to MPa for an MPa-vs-MPa comparison.
    train_dl = DataLoader(SONNDataset(X_train, y_train_s), batch_size=bs, shuffle=bool(config.train.shuffle))
    dev_dl   = DataLoader(SONNDataset(X_dev,   y_dev_s),   batch_size=bs)
    test_dl  = DataLoader(SONNDataset(X_test,  y_test),    batch_size=bs)

    # --- Build + train -------------------------------------------------------
    model = SONN(
        config,
        d_model=X_train.shape[1],
        feature_names=feature_names,
    )
    model = model.to(config.train.device)
    print(model)

    trainer = Trainer(config, feature_names=feature_names)
    Trainer.set_seed(int(config.train.seed))
    trainer.train(model, train_dl, dev_dl, test_dl, resume=bool(config.get("resume", False)))

    # Train the (num_out, 1) regression head against MSE on dev. Mirrors
    # the otto_classification.py flow: each layer-grown checkpoint persists
    # an *initial-random* out_proj, so without this pass `load_model_checkpoint`
    # below would reload a randomly-initialized head and the reported
    # metrics would reflect noise rather than the trained model.
    if model.out_proj is not None:
        print("\nTraining out_proj...")
        trainer.train_out_proj(model, train_dl, dev_dl)

    trainer.load_model_checkpoint(model, config.train.device)

    # --- Report --------------------------------------------------------------
    def report(tag: str) -> None:
        """Run inference on the test loader and print MSE / MAE / R²."""
        model_out, _ = trainer.infer(model, test_dl, verbose=False)
        y_pred = model_out.cpu().numpy()
        # SONN.infer for a `type: regressor` model without `out_proj` returns
        # the full last-layer output, shape (N, nbest_neurons). Collapse to
        # the best-error neuron's column so sklearn metrics see a 1-D vector
        # matching y_test. After `trainer.prune(...)` the last layer holds a
        # single neuron and the shape is (N, 1), still handled by the same
        # path via `_best_neuron_column` returning column 0.
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            best_col = model._best_neuron_column(model.layers[-1])
            y_pred = y_pred[:, best_col]
        elif y_pred.ndim == 2:
            y_pred = y_pred[:, 0]

        # Predictions are in z-scored target space (the trainer never saw the
        # raw-MPa scale). Inverse-transform so the reported numbers are in
        # MPa and directly comparable to y_test.
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        mse  = metrics.mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae  = metrics.mean_absolute_error(y_test, y_pred)
        r2   = metrics.r2_score(y_test, y_pred)
        print(f"--- {tag} ---")
        print(f"  MSE  on test set:           {mse:0.4f} MPa^2")
        print(f"  RMSE on test set:           {rmse:0.4f} MPa")
        print(f"  MAE  on test set:           {mae:0.4f} MPa")
        print(f"  R^2  on test set:           {r2:0.4f}")
        print(f"  selected feature indices:   {model.get_selected_features_indices()}")
        print(f"  unselected feature indices: {model.get_unselected_features_indices()}")
        print(f"  selected features:          {model.get_selected_features()}")
        print(f"  unselected features:        {model.get_unselected_features()}")

    report("Full trained model")

    # --- Plot ----------------------------------------------------------------
    out_dir = Path(__file__).parent
    PlotModel(
        model,
        filename=str(out_dir / "concrete_model"),
        plot_neuron_name=True,
        view=False,
    ).plot()

    # Prune to the single best-error path through the network for a more
    # readable "discovered formula" visualization.
    trainer.prune(model)
    report("Pruned model")

    PlotModel(
        model,
        filename=str(out_dir / "concrete_pruned_model"),
        plot_neuron_name=True,
        view=False,
    ).plot()

    print("Done!")


if __name__ == "__main__":
    main()
