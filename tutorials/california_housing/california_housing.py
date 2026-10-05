"""California-housing regression tutorial — Hydra entry point.

Load fetch_california_housing, hold out a random tutorial.test_size fraction,
split the rest into train / dev (tutorial.dev_split), fit a SONN regressor, then
report MSE / MAE on the held-out rows and plot the network.

Input / target handling is set by the `tutorial:` section of the config,
read by tutorial_params() (which documents every key and its default):
  * feature_engineering: master switch for the two feature groups below;
    off (the default) feeds the eight raw columns. Every config here sets it.
  * log_features: the skewed features (MedInc and the per-household counts)
    are log-transformed before standardization, AveBedrms is replaced by the
    log bedrooms-per-room ratio, and log rooms-per-person is appended.
  * location_features: rotated coordinates, log distances to the four big
    cities, and a k-nearest-neighbour price encoding of location fitted on
    the training rows only (knn_price_k neighbours).
  * z_clip: standardized features are clipped to +-z_clip sigma.
  * log_target: fit log(price) instead of price; predictions are exp'd back.
  * clip_predictions: predictions are clipped to the training target range
    (the target is censored at $500k, so nothing above 5.0 can be right).
  * censor_cap: the training loss treats rows at the cap as "at least 5.0"
    (train.censor_target_at), so they stop dragging expensive areas down.
  * test_size / dev_split / val_split: how much data is held out, how the
    rest is divided between the neuron fits (train) and neuron selection
    (dev), and the optional validation slice of the training rows.
  * n_ensemble: train this many models on different seeds and train/dev
    shuffles of the non-test rows, report each one (the seed-noise floor)
    and the metrics of their averaged prediction.

Run from the repo root:
    python -m tutorials.california_housing.california_housing

Hydra overrides work on every config key. Examples:
    python -m tutorials.california_housing.california_housing resume=true
    python -m tutorials.california_housing.california_housing --config-name california_housing_legendre
    python -m tutorials.california_housing.california_housing --config-name california_housing_legendre_finetune
    python -m tutorials.california_housing.california_housing train.optimizer.optimizer_params.lr=1e-2
    python -m tutorials.california_housing.california_housing hydra.run.dir=/tmp/ca_run
    python -m tutorials.california_housing.california_housing tutorial.n_ensemble=5 tutorial.val_split=0.1
"""
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from torch.utils.data import DataLoader

from torchsonn.data.dataset import SONNDataset
from torchsonn.data.preprocessing import (
    SequenceTypeSet,
    split_dataset,
    train_preprocessing,
)
from torchsonn.logger import setup_logger
from torchsonn.model import SONN
from torchsonn.plot_model import PlotModel
from torchsonn.trainer import Trainer


# (lat, lon) of Los Angeles, San Francisco, San Diego, San Jose.
CITIES = {
    "LA": (34.05, -118.24),
    "SF": (37.77, -122.42),
    "SD": (32.72, -117.16),
    "SJ": (37.34, -121.89),
}
EARTH_RADIUS_KM = 6371.0


def tutorial_params(config: DictConfig) -> SimpleNamespace:
    """Read the `tutorial:` section of the config into a namespace, filling in
    the defaults below for every key the YAML / CLI leaves out.

    Unknown keys raise, so a typo in `tutorial.<key>=...` cannot silently fall
    back to the default.
    """
    cfg = config.get("tutorial") or {}
    p = SimpleNamespace()

    # Data split. The gmdhpy-style layout held out half the data and then split
    # the other half 50/50 (sqMode1), so the neuron fits saw a quarter of the
    # rows (5160). Comparability with gmdhpy was already gone once the split
    # became random, so use the data: hold out 20%, and give the fits 3 of
    # every 4 remaining rows (sqMode4_1) -> train 12384 / dev 4128 / test 4128.
    # Both the polynomial fits and the kNN price encoding get denser data; the
    # price is a smaller, noisier test set (standard error of its MSE is
    # roughly 0.01), so do not compare its numbers with the 50/50-era table at
    # the third decimal. Keep train.batch_size above the train-row count so
    # LBFGS stays full-batch. `dev_split` names a SequenceTypeSet member.
    p.test_size = float(cfg.get("test_size", 0.2))
    p.dev_split = SequenceTypeSet[str(cfg.get("dev_split", "sqMode4_1"))]

    # Validation split: this fraction of the *training* rows (after the dev
    # split) is set aside and used by nothing that selects - not the candidate
    # fits, not selection. The trainer reports the best neuron's error on it
    # next to the dev error after every layer (the gap is the selection bias),
    # and `train.stop_source: val` lets the growth criterion and the end-to-end
    # early stop read it instead of dev. Taken as every k-th training row, so
    # it is deterministic. 0 = no validation split.
    # Measured 2026-09-30 (three repeats each, candidate stop on train): at 0.1
    # the Legendre baseline loses 0.006 MSE to the 10% fewer training rows
    # (0.1950 vs 0.1892), RBF-8 moves 0.001; reading the stop rules from the
    # 1239-row split (train.stop_source=val) costs another 0.003-0.010; and the
    # dev-vs-val gap narrows with depth for both families, i.e. no sign of the
    # search fitting the dev split. Off by default so the published numbers
    # keep every training row; set 0.1 for a diagnostic run.
    p.val_split = float(cfg.get("val_split", 0.0))

    # Ensemble size. Member m uses seed train.seed + m and, for m > 0, a
    # reshuffle of the non-test rows before the train/dev split, so the members
    # differ in both their random state and the data each neuron fit and
    # selection saw (the kNN price encoding is refitted per member too). The
    # test rows are the same for all members, so the per-member metrics are
    # directly comparable: their spread is the noise floor a single-run
    # comparison has to beat, and the mean of the members' predictions is the
    # ensemble. 1 = plain single run. Cost is n_ensemble full trainings; each
    # member writes its own run folder under train.checkpoint_dir, and the
    # network diagrams are drawn for the last member, into its run folder.
    p.n_ensemble = int(cfg.get("n_ensemble", 1))

    # Standardized features are clipped to +-z_clip sigma; see the comment at
    # the clipping site for why.
    p.z_clip = float(cfg.get("z_clip", 5.0))

    # Feature engineering master switch. Off: the model sees the eight raw
    # dataset columns, standardized and clipped, and nothing else. On: the
    # transforms enabled by `log_features` and `location_features` below are
    # applied. Off by default so the bare pipeline is the baseline; every
    # config in this folder turns it on.
    p.feature_engineering = bool(cfg.get("feature_engineering", False))

    # (With feature_engineering.) Log-transform the heavy-tailed features and
    # append ratio features before the scaler. AveRooms / AveBedrms /
    # Population / AveOccup span 2-3 decades with a long right tail;
    # standardizing them as-is puts the bulk of the rows into a narrow band
    # around zero and the tail out past +-5, where the clip truncates it. log()
    # makes them near-Gaussian, so the Legendre squash spends its resolution on
    # the data instead of on the tail. The two ratios are the usual
    # California-housing engineered features: bedrooms per room (a density /
    # quality proxy) and rooms per person (log-transformed: it is a ratio of
    # two skewed quantities).
    p.log_features = bool(cfg.get("log_features", True))

    # (With feature_engineering.) Location features. Price is a rough,
    # non-smooth function of (Latitude, Longitude) that a degree-3 pair
    # polynomial cannot draw, and location is where tree ensembles get most of
    # their edge on this dataset. Three cheap proxies:
    #   * rotated coordinates lat+lon / lat-lon - California's coast and its
    #     price gradient run diagonally (NW-SE), so the rotated axes let a pair
    #     neuron model "distance inland" as a near-1-D function;
    #   * log haversine distance to the four largest metros (CITIES) - the
    #     price surface is a set of peaks centred on them;
    #   * a k-nearest-neighbour encoding: mean log price of the knn_price_k
    #     nearest *training* blocks, leave-one-out for the training rows so a
    #     row never sees its own target. Dev / test rows look up training
    #     neighbours only, so neuron selection on dev stays leak-free.
    p.location_features = bool(cfg.get("location_features", True))
    p.knn_price_k = int(cfg.get("knn_price_k", 20))

    # Fit log(price) instead of price. The log compresses the target's right
    # skew and a log model estimates a conditional *median*, which favours MAE;
    # the raw target estimates the conditional *mean*, which is what MSE
    # rewards. The trade is real and was measured twice: under the early
    # pipeline (no location features, no fine-tune) log won on MSE (0.3013 vs
    # 0.3191); under the current one raw wins on MSE (0.1877 vs 0.1912) and log
    # wins on MAE (0.2719 vs 0.2791), the same direction as gradient boosting
    # on these features. Off, since the tutorial's headline metric is MSE; set
    # true to trade ~0.0035 MSE for ~0.007 MAE. The censoring cap (censor_cap)
    # follows the target's units.
    p.log_target = bool(cfg.get("log_target", False))

    # Clip predictions to the training target range. The dataset caps prices
    # at 5.0, so any prediction above it is wrong by construction, and nothing
    # sells below the observed minimum either. Free, and it removes the worst
    # residuals on high-income rows the model extrapolates past the cap.
    p.clip_predictions = bool(cfg.get("clip_predictions", True))

    # Treat the target's cap as censoring in the training loss. 4.8% of rows
    # are recorded at the 5.0 ceiling; with plain squared error each of them
    # says "worth exactly 5.0" and drags the fitted surface down in expensive
    # areas (measured before this: the capped rows carried 24% of the test MSE
    # and the uncapped rows priced 4-5 were under-predicted by 0.74 on
    # average). With `train.censor_target_at` set to the cap, NormMSE clips the
    # prediction to the cap for those rows, so predicting above it costs
    # nothing. The cap is taken from the training targets (5.00001 in this
    # dataset) in the loss's units.
    p.censor_cap = bool(cfg.get("censor_cap", True))

    unknown = set(cfg) - set(vars(p))
    if unknown:
        raise KeyError(f"unknown tutorial key(s): {sorted(unknown)}; known: {sorted(vars(p))}")
    return p


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def knn_price_feature(
    train_xy: np.ndarray, train_log_price: np.ndarray, others: list[np.ndarray], k: int
) -> list[np.ndarray]:
    """Mean log price of the k nearest training blocks, for the training rows
    (leave-one-out) and for each array in `others` (training neighbours only).

    `train_xy` / `others` hold (Latitude, Longitude) in degrees.
    """
    tree = BallTree(np.radians(train_xy), metric="haversine")
    # Training rows: query k+1 and drop the row itself from its own neighbour
    # list (it is at distance 0; if duplicates share the coordinates, drop the
    # one whose index matches, else the farthest).
    idx = tree.query(np.radians(train_xy), k=k + 1, return_distance=False)
    self_pos = idx == np.arange(len(train_xy))[:, None]
    has_self = self_pos.any(axis=1)
    keep = ~self_pos
    keep[~has_self, -1] = False
    nbr = idx[keep].reshape(len(train_xy), k)
    out = [train_log_price[nbr].mean(axis=1).astype(np.float32)]
    for xy in others:
        idx = tree.query(np.radians(xy), k=k, return_distance=False)
        out.append(train_log_price[idx].mean(axis=1).astype(np.float32))
    return out


def engineer_features(x: np.ndarray, names: list[str], p: SimpleNamespace) -> tuple[np.ndarray, list[str]]:
    """Row-wise feature transform (no fitted state, so no train/test leakage).

    With p.log_features: log-transforms the skewed columns in place (MedInc,
    AveRooms, Population, AveOccup), replaces AveBedrms by the log
    bedrooms-per-room ratio, and appends log rooms-per-person. With
    p.location_features: appends the rotated coordinates and the log city
    distances (the kNN price column is added per split, in main). Both need
    p.feature_engineering. Returns (features, names).
    """
    new_names = list(names)
    if not p.feature_engineering:
        return x, new_names
    col = {n: i for i, n in enumerate(names)}
    x = x.copy()
    if p.log_features:
        x, new_names = _log_features(x, col, new_names)
    if p.location_features:
        lat, lon = x[:, col["Latitude"]], x[:, col["Longitude"]]
        cols = [lat + lon, lat - lon]
        new_names += ["LatPlusLon", "LatMinusLon"]
        for city, (clat, clon) in CITIES.items():
            cols.append(np.log1p(_haversine_km(lat, lon, clat, clon)))
            new_names.append(f"logDist{city}")
        x = np.column_stack([x] + cols).astype(np.float32)
    return x, new_names


def _log_features(x: np.ndarray, col: dict[str, int], new_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """The log_features part of engineer_features (x is already a copy)."""
    bedrooms_per_room = x[:, col["AveBedrms"]] / x[:, col["AveRooms"]]
    rooms_per_person = x[:, col["AveRooms"]] / x[:, col["AveOccup"]]
    # MedInc: skew 1.6 raw, -0.2 after log (capped at 15, so the top end
    # otherwise sits past the +-5 sigma clip).
    for n in ("MedInc", "AveRooms", "Population", "AveOccup"):
        x[:, col[n]] = np.log(x[:, col[n]])
        new_names[col[n]] = f"log{n}"
    # AveBedrms stays at skew ~6 / kurtosis ~60 even after a log (a few
    # institutional blocks define its scale, squeezing the bulk into +-0.3
    # sigma), and log(AveBedrms) == log(AveRooms) + log(bedrooms_per_room)
    # exactly. Replace the column by the log ratio (skew 0.5, kurtosis 1): same
    # information, well-behaved distribution.
    x[:, col["AveBedrms"]] = np.log(bedrooms_per_room)
    new_names[col["AveBedrms"]] = "logBedrmsPerRoom"
    x = np.column_stack([x, np.log(rooms_per_person)]).astype(np.float32)
    new_names += ["logRoomsPerPerson"]
    return x, new_names


# `config_path` resolves relative to this file. `version_base="1.3"` keeps
# Hydra's auto-chdir disabled by default, so any relative paths in the YAML
# (e.g. `checkpoint_dir: ../california_housing/checkpoint/<config name>`) resolve
# from the launch cwd, not from Hydra's per-run output dir.
@hydra.main(version_base="1.3", config_path=".", config_name="california_housing")
def main(config: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = setup_logger(str(run_dir / "train.log"))
    # Every config in this folder keeps its checkpoints under
    # ../california_housing/checkpoint/<config name>/ through the
    # `${hydra:job.config_name}` interpolation. Reading the value resolves it;
    # store the plain string so the copies the model and the checkpoints keep
    # do not carry an interpolation that only resolves inside a Hydra run.
    config.train.checkpoint_dir = str(config.train.checkpoint_dir)
    logger.info("Loaded config:")
    logger.info(OmegaConf.to_yaml(config))
    p = tutorial_params(config)

    # --- Load California housing ----------------------------------------------
    housing = fetch_california_housing()
    housing_data, feature_names = engineer_features(
        housing.data.astype(np.float32), list(housing.feature_names), p
    )
    housing_target = housing.target.astype(np.float32)

    # Shuffle and hold out the test rows; same random_state as before, so the
    # 20% test set is a subset of the earlier 50% one. Fixed for every
    # ensemble member.
    user_train_x, test_x, user_train_y, test_y = train_test_split(
        housing_data, housing_target, test_size=p.test_size, random_state=42
    )

    def predict(trainer: Trainer, model: SONN, test_dl: DataLoader, y_lo: float, y_hi: float) -> np.ndarray:
        """Test-set predictions in raw $100k units."""
        model_out, _ = trainer.infer(model, test_dl, verbose=False)
        y_pred = model_out.cpu().numpy()
        if p.log_target:
            y_pred = np.exp(y_pred)  # back to raw $100k
        if p.clip_predictions:
            y_pred = np.clip(y_pred, y_lo, y_hi)
        return y_pred

    def fit_member(member: int) -> dict:
        """Train one model end to end; return it with its test predictions."""
        seed = int(config.train.seed) + member
        names = list(feature_names)
        ux, uy = user_train_x, user_train_y
        if member > 0:
            # Reshuffle the non-test rows so this member's train/dev split (and
            # its kNN encoding) differ from the other members'.
            perm = np.random.default_rng(seed).permutation(len(ux))
            ux, uy = ux[perm], uy[perm]

        # Train rows fit the neurons, dev rows select them (and early-stop the
        # fine-tune passes); see dev_split in tutorial_params for the ratio.
        train_x, train_y, dev_x, dev_y = split_dataset(ux, uy, p.dev_split)
        tx, ty = test_x, test_y
        val_x = val_y = None
        if p.val_split:
            every = int(round(1.0 / p.val_split))
            is_val = (np.arange(len(train_x)) % every) == 0
            val_x, val_y = train_x[is_val], train_y[is_val]
            train_x, train_y = train_x[~is_val], train_y[~is_val]

        if p.feature_engineering and p.location_features:
            # Fitted on the training rows only, so it has to come after the split.
            ll = [names.index("Latitude"), names.index("Longitude")]
            others = [dev_x[:, ll], tx[:, ll]] + ([val_x[:, ll]] if val_x is not None else [])
            knn_tr, knn_dev, knn_te, *knn_val = knn_price_feature(
                train_x[:, ll], np.log(train_y), others, p.knn_price_k
            )
            train_x = np.column_stack([train_x, knn_tr]).astype(np.float32)
            dev_x = np.column_stack([dev_x, knn_dev]).astype(np.float32)
            tx = np.column_stack([tx, knn_te]).astype(np.float32)
            if val_x is not None:
                val_x = np.column_stack([val_x, knn_val[0]]).astype(np.float32)
            names = names + [f"knnLogPrice{p.knn_price_k}"]

        # train_preprocessing normalizes orientation + sanity-checks shapes
        train_x, train_y = train_preprocessing(train_x, train_y, names)
        dev_x, dev_y = train_preprocessing(dev_x, dev_y, names)
        tx, ty = train_preprocessing(tx, ty, names)
        if val_x is not None:
            val_x, val_y = train_preprocessing(val_x, val_y, names)

        # Prediction bounds in raw $100k units, from the training rows only.
        y_lo, y_hi = float(train_y.min()), float(train_y.max())
        if p.log_target:
            # Train / dev see log(price). The test loader keeps the raw target:
            # the trainer never reads it, only predict() does, after exp'ing.
            train_y = np.log(train_y).astype(np.float32)
            dev_y = np.log(dev_y).astype(np.float32)
            if val_y is not None:
                val_y = np.log(val_y).astype(np.float32)
        # Censoring cap in the units the loss sees (see censor_cap). Set before
        # the model is built, since SONN passes it to its NormMSE.
        config.train.censor_target_at = float(train_y.max()) if p.censor_cap else None

        # ---- Standardize features (StandardScaler, same as gmdhpy) -----------
        feature_scaler = StandardScaler()
        feature_scaler.fit(train_x)
        # Clip the z-scores to +-z_clip sigma. The quadratic / polyquad neurons
        # take the raw z-scores (only the orthogonal-polynomial families squash
        # their inputs), and California housing has a handful of rows far
        # outside the bulk of the data: AveRooms up to 142 and AveBedrms up to
        # 34 against 99.9th percentiles of ~29 and ~6, i.e. z-scores of 60-80
        # when such a row lands in the test half. Fed through several
        # polynomial layers, with the shortcut re-injecting them at every
        # layer, those rows produce predictions in the tens of thousands and
        # dominate the test MSE on their own. Clipping caps the extrapolation
        # without touching typical rows (every feature's 99.9th percentile is
        # well inside +-5).
        train_x = np.clip(feature_scaler.transform(train_x), -p.z_clip, p.z_clip)
        dev_x = np.clip(feature_scaler.transform(dev_x), -p.z_clip, p.z_clip)
        tx = np.clip(feature_scaler.transform(tx), -p.z_clip, p.z_clip)
        if val_x is not None:
            val_x = np.clip(feature_scaler.transform(val_x), -p.z_clip, p.z_clip)

        bs = config.train.batch_size
        train_dl = DataLoader(SONNDataset(train_x, train_y), batch_size=bs, shuffle=bool(config.train.shuffle))
        dev_dl = DataLoader(SONNDataset(dev_x, dev_y), batch_size=bs)
        test_dl = DataLoader(SONNDataset(tx, ty), batch_size=bs)
        val_dl = DataLoader(SONNDataset(val_x, val_y), batch_size=bs) if val_x is not None else None
        logger.info("rows: train %d, dev %d, val %s, test %d", len(train_x), len(dev_x),
                    len(val_x) if val_x is not None else "-", len(tx))

        # Seed first: everything random from here on, the model included,
        # follows this member's seed.
        Trainer.set_seed(seed)
        model = SONN(config, d_model=train_x.shape[1], feature_names=names)
        model = model.to(config.train.device)
        trainer = Trainer(config, feature_names=names)

        # --- Train ------------------------------------------------------------
        resume = bool(config.get("resume", False)) and member == 0
        trainer.train(model, train_dl, dev_dl, test_dl, resume=resume, val_dl=val_dl)

        # Regression head (Linear(num_out, 1)); built and trained only when
        # model.use_output_projection is True (the *_finetune configs). Saves
        # the checkpoint itself, so the load below picks the fitted head up.
        if model.out_proj is not None:
            trainer.train_out_proj(model, train_dl, dev_dl)

        trainer.load_model_checkpoint(model, config.train.device)

        # Optional final pass: fine-tune every parameter at once against the
        # readout the model is scored on (config `finetune_end_to_end`; see
        # california_housing_legendre_finetune.yaml). Runs after
        # load_model_checkpoint so it starts from the best checkpointed
        # weights. The pass saves the model it ends with to the run folder's
        # model_last.ckpt; the metrics are computed from the in-memory model,
        # which holds the same weights.
        # Same wiring as tutorials/ccpp.
        if bool(config.get("finetune_end_to_end", False)):
            if bool(config.get("finetune_drop_head", False)) and model.out_proj is not None:
                logger.info("Dropping the out_proj head before end-to-end fine-tuning")
                model.out_proj = None
            if bool(config.get("finetune_prune_first", False)):
                trainer.prune(model)
                logger.info("Pruned to the read path before fine-tuning: %d layers", len(model.layers))
            logger.info("End-to-end fine-tune of all parameters (head %s)",
                        "removed" if model.out_proj is None else "kept and trained")
            trainer.train_finetune(model, train_dl, dev_dl, val_dl=val_dl)
            if model.out_proj is None and bool(config.model.use_output_projection):
                logger.info("The saved model has no head (finetune_drop_head): load %s into a "
                            "model built with model.use_output_projection=false; a model with "
                            "a head would keep an untrained one",
                            trainer.run_dir / "model_last.ckpt")

        return {
            "seed": seed, "model": model, "trainer": trainer, "test_dl": test_dl,
            "y_lo": y_lo, "y_hi": y_hi, "test_y": ty,
            "y_pred": predict(trainer, model, test_dl, y_lo, y_hi),
        }

    # --- Train the ensemble -----------------------------------------------------
    members = []
    for m in range(p.n_ensemble):
        logger.info("=== ensemble member %d/%d (seed %d) ===", m + 1, p.n_ensemble, int(config.train.seed) + m)
        members.append(fit_member(m))
        mem = members[-1]
        logger.info("    member %d: test mse %.4f  mae %.4f  run folder %s", m + 1,
                    metrics.mean_squared_error(mem["test_y"], mem["y_pred"]),
                    metrics.mean_absolute_error(mem["test_y"], mem["y_pred"]),
                    mem["trainer"].run_dir)

    y_true = members[0]["test_y"]
    preds = np.stack([mem["y_pred"] for mem in members])
    mses = np.array([metrics.mean_squared_error(y_true, yp) for yp in preds])
    maes = np.array([metrics.mean_absolute_error(y_true, yp) for yp in preds])
    print(f"--- {p.n_ensemble} member(s), same test rows ---")
    for mem, mse, mae in zip(members, mses, maes):
        print(f"  seed {mem['seed']:>3d}:  mse {mse:0.4f}   mae {mae:0.4f}   layers {len(mem['model'].layers)}")
    if p.n_ensemble > 1:
        print(f"  mean +- std:  mse {mses.mean():0.4f} +- {mses.std(ddof=1):0.4f}"
              f"   mae {maes.mean():0.4f} +- {maes.std(ddof=1):0.4f}   (single-run noise floor)")
        ens = preds.mean(axis=0)
        print(f"--- Ensemble (mean of {p.n_ensemble} predictions) ---")
        print(f"  mse on test set:           {metrics.mean_squared_error(y_true, ens):0.4f}")
        print(f"  mae on test set:           {metrics.mean_absolute_error(y_true, ens):0.4f}")

    # --- Feature report + plots for the last member ------------------------------
    last = members[-1]
    model, trainer, test_dl = last["model"], last["trainer"], last["test_dl"]

    def report(tag: str) -> None:
        """Print the last member's MSE / MAE and selected / unselected features."""
        y_pred = predict(trainer, model, test_dl, last["y_lo"], last["y_hi"])
        print(f"--- {tag} ---")
        print(f"  mse on test set:           {metrics.mean_squared_error(y_true, y_pred):0.4f}")
        print(f"  mae on test set:           {metrics.mean_absolute_error(y_true, y_pred):0.4f}")
        print(f"  selected feature indices:  {model.get_selected_features_indices()}")
        print(f"  unselected feature indices:{model.get_unselected_features_indices()}")
        print(f"  selected features:         {model.get_selected_features()}")
        print(f"  unselected features:       {model.get_unselected_features()}")

    report("Full trained model" + (" (last member)" if p.n_ensemble > 1 else ""))

    # --- Plot -----------------------------------------------------------------
    # Into the last member's run folder, next to its checkpoints and train.log,
    # so no run overwrites another run's plots.
    out_dir = trainer.run_dir
    PlotModel(
        model,
        filename=str(out_dir / "california_housing_model"),
        plot_neuron_name=True,
        view=False,
    ).plot()

    trainer.prune(model)
    report("Pruned model" + (" (last member)" if p.n_ensemble > 1 else ""))

    PlotModel(
        model,
        filename=str(out_dir / "california_housing_pruned_model"),
        plot_neuron_name=True,
        view=False,
    ).plot()

    print(f"Run folder: {trainer.run_dir}")
    print("Done!")


if __name__ == "__main__":
    main()