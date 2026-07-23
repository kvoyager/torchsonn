"""UCI Combined Cycle Power Plant (CCPP) regression — Hydra entry point.

Fetches the 9568-sample UCI CCPP dataset (Tüfekci & Kaya, 2014), trains a SONN
regressor with `linear_cov + quadratic + polyquad` reference functions, and
evaluates it with **5×2 cross-validation** — the canonical CCPP benchmark
protocol — reporting mean ± std of MSE / RMSE / MAE / R² over the 10 folds.

The plant runs a gas turbine and a steam turbine in tandem. Four hourly-average
ambient variables set the operating point, and the model predicts the net
hourly electrical output PE (MW):

    AT  Ambient Temperature   (°C)        strong negative driver (r ≈ -0.95)
    V   Exhaust Vacuum        (cm Hg)      strong negative driver (r ≈ -0.87)
    AP  Ambient Pressure      (millibar)   mild positive driver   (r ≈ +0.52)
    RH  Relative Humidity     (%)          mild positive driver   (r ≈ +0.39)
    PE  Net electrical output (MW)         target, range 420–496 MW

The relationship is predominantly linear (a plain linear fit already reaches
R² ≈ 0.93), so this tutorial is a nice demonstration of SONN discovering that
structure with `linear_cov` and then squeezing out the residual curvature /
interactions (AT×V collinearity, humidity's temperature-dependent effect) with
`quadratic` and a full four-input `polyquad` neuron.

Why 5×2cv instead of one split? A single train/test split yields one number
whose third decimal is noise — a different random split can swing R² by
±0.005. Tüfekci & Kaya (2014), the paper every CCPP result is measured
against, use 5×2 cross-validation: shuffle the data and cut it in half, train
on half A / test on half B, then swap (train B / test A) — that is one "2-fold"
repeat; do it for five independent shuffles → 5×2 = 10 train/test measurements,
and report the mean and standard deviation. The spread across the 10 tells you
how stable the number actually is.

Output layout — every run is self-contained and never overwrites a previous
one. Under `train.checkpoint_dir` a per-run folder named `YYYY-MM-DD-HH-MM` is
created; inside it each of the 10 folds gets its own `fold_NN/` checkpoint
subfolder, and a single `train.log` for the whole run sits alongside them. All
console output (this script's messages, the trainer's records, and each
progress bar's final line) is routed through logging into that one log file.

NOTE ON COST: this trains the model 10 times (once per fold), so it is ~10× a
single fit. The committed config is deliberately light (small survivor pool /
candidate cap) so the full 10-fold run takes ~10 min on CPU rather than hours.
For this small per-fold problem CPU is usually faster than GPU (kernel-launch
overhead dominates), so `train.device=cpu` is often the quicker choice despite
the committed `device: cuda`. The heavier two-polyquad setup used while
exploring the model is preserved in ccpp_heavy.yaml and is much slower:
    python -m tutorials.ccpp.ccpp --config-name=ccpp_heavy

Run from the repo root:
    python -m tutorials.ccpp.ccpp

Hydra overrides work on every config key. Examples:
    python -m tutorials.ccpp.ccpp train.optimizer.optimizer_params.lr=5e-2
    python -m tutorials.ccpp.ccpp model.ref_functions='[linear_cov]' train.max_layer_count=3
    python -m tutorials.ccpp.ccpp train.device=cpu
    python -m tutorials.ccpp.ccpp hydra.run.dir=/tmp/ccpp_run
"""
import datetime
import logging
import urllib.request
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm as _TqdmBase

import torchsonn.trainer as _trainer_module
from torchsonn.data.dataset import SONNDataset
from torchsonn.logger import setup_logger
from torchsonn.model import SONN
from torchsonn.plot_model import PlotModel
from torchsonn.trainer import Trainer

logger = logging.getLogger(__name__)


# --- 5×2 cross-validation protocol -------------------------------------------
# FIVE independent shuffles of the full dataset, each cut into two equal halves;
# for every shuffle we train twice — on half A testing on B, then on B testing
# on A — giving 5×2 = 10 train/test measurements. Both are fixed by the
# protocol (Tüfekci & Kaya, 2014); they are module constants, not config, so
# the reported number stays comparable to the published benchmark.
N_REPEATS = 5   # the "5": five different shuffle-and-halve rounds
N_FOLDS = 2     # the "2": the A/B swap within each round
# SONN's `validate` selection criterion needs a held-out validation split, so
# each training half is further cut into train/dev at this ratio. The test half
# is never touched during training.
DEV_FRACTION = 0.20
# Fixed seed for that inner train/dev cut — deterministic given a training half;
# the 5 outer shuffles use random_state = 0..N_REPEATS-1.
INNER_SPLIT_SEED = 0


# --- Progress-bar → log mirroring --------------------------------------------
# The trainer draws tqdm progress bars straight to the console, so their final
# state (100%, timings, the val_loss postfix) never reaches the log file. We
# patch the trainer's `tqdm` with a subclass whose close() appends that final
# line to the run log — verbatim, no timestamp/level prefix — interleaved with
# the trainer's INFO records, so the file reads just like the console.
_bar_file_handler: logging.FileHandler | None = None


def _log_bar_line(line: str) -> None:
    """Append one finished progress-bar line to the run log, verbatim.

    Written through the FileHandler's own lock so it can't tear a concurrent
    log record sharing the same stream. No-op until main() wires up the
    handler (or if the bar produced no text).
    """
    fh = _bar_file_handler
    if fh is None or not line:
        return
    fh.acquire()
    try:
        fh.stream.write(line + "\n")
        fh.flush()
    finally:
        fh.release()


class _LoggingTqdm(_TqdmBase):
    """tqdm that also records its final rendered line to the run log on close().

    `format_meter(**format_dict)` is exactly what tqdm renders to the console
    (it is what tqdm's own __repr__ returns), so the logged line matches the
    bar the user sees. Disabled bars (e.g. `disable=True` inference passes) log
    nothing. Console behaviour is unchanged.
    """

    def close(self) -> None:
        if not self.disable:
            try:
                _log_bar_line(self.format_meter(**self.format_dict))
            except Exception:
                # Never let log mirroring break a training run.
                pass
        super().close()


# Route the trainer's bars through the logging subclass. `torchsonn.trainer`
# did `from tqdm import tqdm`, so rebinding the module global makes every
# `tqdm(...)` call inside the trainer use _LoggingTqdm.
_trainer_module.tqdm = _LoggingTqdm


# --- Dataset access ----------------------------------------------------------
# The canonical UCI release ships the data as `CCPP/Folds5x2_pp.xlsx` inside a
# zip (archive.ics.uci.edu/static/public/294/...). Reading .xlsx needs
# `openpyxl`, which isn't a dependency of this repo, so — exactly as the
# Concrete tutorial does for its .xls source — we instead pull from a stable
# CSV mirror of the same data and cache locally. The mirror is byte-identical
# to the canonical XLS export: same 9568 rows, same 5 columns, same units
# (verified by column-sorted comparison against the UCI xlsx).
DATASET_URL = (
    "https://raw.githubusercontent.com/YungChunLu/UCI-Power-Plant/master/data.csv"
)
# UCI's column-name convention (Tüfekci 2014). Re-applied by position after
# load so the trained SONN gets readable feature names regardless of how the
# mirror happened to spell its header.
FEATURE_NAMES = [
    "AT",   # Ambient Temperature, °C
    "V",    # Exhaust Vacuum, cm Hg
    "AP",   # Ambient Pressure, millibar
    "RH",   # Relative Humidity, %
]
TARGET_NAME = "PE"   # Net hourly electrical energy output, MW


def _load_ccpp(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Download (once) + parse the CCPP dataset.

    Returns:
      X: (9568, 4) float32 feature matrix in UCI column order (AT, V, AP, RH).
      y: (9568,)    float32 target (net electrical output PE, in MW).
      feature_names: canonical UCI names, attached to the model so
                     SONN.get_selected_features() prints something readable.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "ccpp.csv"
    if not csv_path.exists():
        logger.info("Downloading %s → %s", DATASET_URL, csv_path.name)
        with urllib.request.urlopen(DATASET_URL, timeout=60) as resp:
            payload = resp.read()
        csv_path.write_bytes(payload)
    else:
        logger.info("Using cached file: %s", csv_path)

    df = pd.read_csv(csv_path)
    assert df.shape[1] == 5, f"expected 5 columns (4 features + target), got {df.columns.tolist()}"
    # Normalize column names by position — the last column is always the PE
    # target by Tüfekci's convention, the first four are AT, V, AP, RH.
    df.columns = FEATURE_NAMES + [TARGET_NAME]
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df[TARGET_NAME].to_numpy(dtype=np.float32)
    return X, y, list(FEATURE_NAMES)


def _predict_mw(trainer: Trainer, model: SONN, test_dl: DataLoader,
                y_scaler: StandardScaler) -> np.ndarray:
    """Infer on the test loader and return predictions in physical MW units."""
    model_out, _ = trainer.infer(model, test_dl, verbose=False)
    y_pred = model_out.cpu().numpy()
    # SONN.infer for a `type: regressor` model without `out_proj` returns the
    # full last-layer output, shape (N, nbest_neurons). Collapse to the
    # best-error neuron's column so sklearn metrics see a 1-D vector matching
    # the target. (After `trainer.prune(...)` the last layer holds a single
    # neuron, shape (N, 1) — the same path via `_best_neuron_column` → col 0.)
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        y_pred = y_pred[:, model._best_neuron_column(model.layers[-1])]
    elif y_pred.ndim == 2:
        y_pred = y_pred[:, 0]
    # Predictions live in z-scored target space (the trainer never saw raw MW);
    # inverse-transform so the numbers are in MW and comparable to y_test.
    return y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()


def _run_fold(config: DictConfig, feature_names: list[str],
              X_tr_half: np.ndarray, y_tr_half: np.ndarray,
              X_te_half: np.ndarray, y_te_half: np.ndarray,
              fold_ckpt_dir: Path, tag: str) -> tuple[SONN, Trainer, dict]:
    """Train on one half, evaluate on the other. Returns (model, trainer, metrics).

    Checkpoints for this fold land in its own `fold_ckpt_dir`, so folds never
    clobber each other and there is nothing to clear between them.
    Standardization is fit on the fold's *training* rows only — no feature or
    target statistics leak from the dev split or the held-out test half.
    """
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
    config.train.checkpoint_dir = str(fold_ckpt_dir)

    # SONN's `validate` criterion needs a validation set, so carve dev out of
    # the training half (the test half stays fully held out for this fold).
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_tr_half, y_tr_half, test_size=DEV_FRACTION, random_state=INNER_SPLIT_SEED,
    )

    # Features span very different scales (AP ≈ 1013 mbar vs V ≈ 54 cm Hg);
    # z-score so polynomial neurons see a comparable dynamic range. The target
    # is z-scored too — *critical* for SONN's regression criterion, whose
    # `regularity_error = Σ(y-ŷ)²/Σy²` denominator would otherwise be swamped by
    # N·μ² (mean PE ≈ 454 MW), collapsing the whole no-skill→perfect range into
    # a razor-thin band and stalling layer growth. After z-scoring Σy² ≈ N, the
    # no-skill baseline is exactly 1.0 and improvements are crisp.
    x_scaler = StandardScaler().fit(X_train)
    X_train = x_scaler.transform(X_train).astype(np.float32)
    X_dev   = x_scaler.transform(X_dev).astype(np.float32)
    X_test  = x_scaler.transform(X_te_half).astype(np.float32)

    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_dev_s   = y_scaler.transform(y_dev.reshape(-1, 1)).ravel().astype(np.float32)

    bs = int(config.train.batch_size)
    # Train / dev see standardized y; the test loader carries raw MW because the
    # trainer never reads it — only _predict_mw does, and it inverse-transforms
    # its predictions to MW for an MW-vs-MW comparison.
    train_dl = DataLoader(SONNDataset(X_train, y_train_s), batch_size=bs, shuffle=bool(config.train.shuffle))
    dev_dl   = DataLoader(SONNDataset(X_dev,   y_dev_s),   batch_size=bs)
    test_dl  = DataLoader(SONNDataset(X_test,  y_te_half), batch_size=bs)

    # Fresh model + trainer for the fold. Re-seed with the same training seed
    # every fold so the ONLY thing that varies across the 10 measurements is the
    # data split — that is exactly the variance 5×2cv is meant to expose.
    Trainer.set_seed(int(config.train.seed))
    model = SONN(config, d_model=X_train.shape[1], feature_names=feature_names)
    model = model.to(config.train.device)
    trainer = Trainer(config, feature_names=feature_names)

    logger.info("=== %s  (train=%d  dev=%d  test=%d)  ckpt=%s ===",
                tag, len(X_train), len(X_dev), len(X_test), fold_ckpt_dir)
    trainer.train(model, train_dl, dev_dl, test_dl, resume=False)

    # Regression head (Linear(num_out, 1)); trained only when
    # use_output_projection is True. Off by default here, so this is a no-op.
    if model.out_proj is not None:
        trainer.train_out_proj(model, train_dl, dev_dl)

    trainer.load_model_checkpoint(model, config.train.device)

    y_pred = _predict_mw(trainer, model, test_dl, y_scaler)
    mse = float(metrics.mean_squared_error(y_te_half, y_pred))
    result = {
        "mse":  mse,
        "rmse": float(np.sqrt(mse)),
        "mae":  float(metrics.mean_absolute_error(y_te_half, y_pred)),
        "r2":   float(metrics.r2_score(y_te_half, y_pred)),
    }
    logger.info("    RMSE=%.4f MW   MAE=%.4f MW   R²=%.4f   features=%s",
                result["rmse"], result["mae"], result["r2"], model.get_selected_features())
    return model, trainer, result


def _report_5x2cv(results: list[dict]) -> None:
    """Log the per-fold table and the mean ± std across the 10 folds."""
    lines = [
        "=" * 66,
        f"5×2 cross-validation summary — {len(results)} train/test measurements",
        "=" * 66,
        f"{'repeat':>6} {'fold':>6} {'RMSE(MW)':>10} {'MAE(MW)':>10} {'R²':>9}",
    ]
    for m in results:
        lines.append(f"{m['repeat']:>6} {m['fold']:>6} "
                     f"{m['rmse']:>10.4f} {m['mae']:>10.4f} {m['r2']:>9.4f}")
    lines.append("-" * 66)
    # Sample standard deviation (ddof=1): the 10 folds are a sample used to
    # estimate the spread of the procedure, matching how CV std is reported.
    for key, label, unit in [("rmse", "RMSE", " MW"),
                             ("mae",  "MAE",  " MW"),
                             ("r2",   "R²",   "")]:
        vals = np.array([m[key] for m in results], dtype=np.float64)
        lines.append(f"  {label:<4} = {vals.mean():.4f} ± {vals.std(ddof=1):.4f}{unit}")
    # One record so the table stays visually aligned in both console and file.
    logger.info("\n".join(lines))


def _make_run_dir(config: DictConfig) -> Path:
    """Create this run's checkpoint/log root: <checkpoint_dir>/YYYY-MM-DD-HH-MM.

    A per-run timestamp folder means a new run never overwrites an earlier one
    (checkpoints or log). If two runs somehow start in the same minute, a
    numeric suffix keeps them distinct.
    """
    base = Path(str(config.train.checkpoint_dir) or "checkpoints")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_dir = base / stamp
    suffix = 2
    while run_dir.exists():
        run_dir = base / f"{stamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# `config_path` resolves relative to this file. `version_base="1.3"` keeps
# Hydra's auto-chdir disabled by default, so any relative paths in the YAML
# (e.g. `checkpoint_dir: checkpoints`) resolve from the launch
# cwd, not from Hydra's per-run output dir.
@hydra.main(version_base="1.3", config_path=".", config_name="ccpp")
def main(config: DictConfig) -> None:
    # One run folder holds this run's log + all 10 fold checkpoint subfolders,
    # and never overwrites a previous run.
    run_dir = _make_run_dir(config)
    setup_logger(str(run_dir / "train.log"))
    # Hand the run's FileHandler to the progress-bar mirror so bar lines land in
    # the same log as the trainer's records.
    global _bar_file_handler
    _bar_file_handler = next(
        (h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)), None
    )
    logger.info("Run directory (log + per-fold checkpoints): %s", run_dir)
    logger.info("Loaded config:")
    logger.info(OmegaConf.to_yaml(config))

    # --- Load -----------------------------------------------------------------
    cache_dir = Path(__file__).resolve().parent / "data"
    X, y, feature_names = _load_ccpp(cache_dir)
    logger.info("CCPP dataset: X=%s, y=%s, features=%s", X.shape, y.shape, feature_names)
    logger.info("  target stats: min=%.2f MW, max=%.2f MW, mean=%.2f MW", y.min(), y.max(), y.mean())
    # Unlike the Concrete tutorial's log-spaced `Age` column, all four CCPP
    # inputs are continuous ambient measurements on roughly symmetric,
    # comparable scales — per-fold z-scoring is the only feature transform they
    # need, so there's no per-column reshaping step here.

    # --- 5×2 cross-validation -------------------------------------------------
    logger.info("Running %d×%d cross-validation (%d folds; each trains the model from scratch)...",
                N_REPEATS, N_FOLDS, N_REPEATS * N_FOLDS)
    results: list[dict] = []
    last_model: SONN | None = None
    last_trainer: Trainer | None = None
    fold_num = 0
    for repeat in range(N_REPEATS):
        # One shuffle of the full data into two equal halves. A different
        # random_state per repeat gives the five independent shuffles; the two
        # folds below reuse this same split, just swapping train ↔ test.
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.50, random_state=repeat, shuffle=True)
        folds = [("A→B", Xa, ya, Xb, yb),
                 ("B→A", Xb, yb, Xa, ya)]
        for name, X_tr, y_tr, X_te, y_te in folds:
            fold_num += 1
            fold_ckpt_dir = run_dir / f"fold_{fold_num:02d}"
            tag = f"repeat {repeat + 1}/{N_REPEATS}  fold {name}  ({fold_num}/{N_REPEATS * N_FOLDS})"
            model, trainer, result = _run_fold(
                config, feature_names, X_tr, y_tr, X_te, y_te, fold_ckpt_dir, tag)
            result["repeat"], result["fold"] = repeat + 1, name
            results.append(result)
            last_model, last_trainer = model, trainer

    _report_5x2cv(results)

    # --- Plot one representative fold -----------------------------------------
    # The 5×2cv number is the benchmark; the diagram is illustrative, so we plot
    # just the final fold's discovered network (all 10 share the same
    # architecture search, differing only in fitted weights).
    out_dir = Path(__file__).parent
    logger.info("Plotting the discovered network from the final fold...")
    PlotModel(
        last_model,
        filename=str(out_dir / "ccpp_model"),
        plot_neuron_name=True,
        view=False,
    ).plot()

    # Prune to the single best-error path for a readable "discovered formula".
    last_trainer.prune(last_model)
    PlotModel(
        last_model,
        filename=str(out_dir / "ccpp_pruned_model"),
        plot_neuron_name=True,
        view=False,
    ).plot()

    logger.info("Done!")


if __name__ == "__main__":
    main()
