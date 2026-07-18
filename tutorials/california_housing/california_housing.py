"""California-housing regression tutorial — Hydra entry point.

Mirrors the gmdhpy example layout: load fetch_california_housing, split 50/50
(test = first half, train = second half — same default as gmdhpy), fit a SONN
regressor with linear_cov reference functions, then report MSE / MAE on the
held-out half and plot the network.

Run from the repo root:
    python -m tutorials.california_housing.california_housing

Hydra overrides work on every config key. Examples:
    python -m tutorials.california_housing.california_housing resume=true
    python -m tutorials.california_housing.california_housing train_on_first_half=true
    python -m tutorials.california_housing.california_housing train.optimizer.optimizer_params.lr=1e-2
    python -m tutorials.california_housing.california_housing hydra.run.dir=/tmp/ca_run
"""
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
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


# `config_path` resolves relative to this file. `version_base="1.3"` keeps
# Hydra's auto-chdir disabled by default, so any relative paths in the YAML
# (e.g. `checkpoint_dir: tutorials/california_housing/checkpoints`) resolve
# from the launch cwd, not from Hydra's per-run output dir.
@hydra.main(version_base="1.3", config_path=".", config_name="california_housing")
def main(config: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = setup_logger(str(run_dir / "train.log"))
    logger.info("Loaded config:")
    logger.info(OmegaConf.to_yaml(config))

    # --- Load California housing ----------------------------------------------
    housing = fetch_california_housing()
    housing_data = housing.data.astype(np.float32)
    housing_target = housing.target.astype(np.float32)

    n = housing_data.shape[0] // 2
    if bool(config.get("train_on_first_half", False)):
        user_train_x, user_train_y = housing_data[:n], housing_target[:n]
        test_x, test_y = housing_data[n:], housing_target[n:]
    else:
        user_train_x, user_train_y = housing_data[n:], housing_target[n:]
        test_x, test_y = housing_data[:n], housing_target[:n]

    # gmdhpy splits the user's train_x 50/50 internally — same split here so
    # the dev MSE is on the same statistical footing as gmdhpy's layer error.
    train_x, train_y, dev_x, dev_y = split_dataset(
        user_train_x, user_train_y, SequenceTypeSet.sqMode1
    )

    # train_preprocessing normalizes orientation + sanity-checks shapes
    train_x, train_y = train_preprocessing(train_x, train_y, list(housing.feature_names))
    dev_x, dev_y = train_preprocessing(dev_x, dev_y, list(housing.feature_names))
    test_x, test_y = train_preprocessing(test_x, test_y, list(housing.feature_names))

    # ---- Standardize features (StandardScaler, same as gmdhpy) ---------------
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_x)
    train_x = feature_scaler.transform(train_x)
    dev_x = feature_scaler.transform(dev_x)
    test_x = feature_scaler.transform(test_x)

    train_ds = SONNDataset(train_x, train_y)
    dev_ds = SONNDataset(dev_x, dev_y)
    test_ds = SONNDataset(test_x, test_y)

    train_dl = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=bool(config.train.shuffle))
    dev_dl = DataLoader(dev_ds, batch_size=config.train.batch_size)
    test_dl = DataLoader(test_ds, batch_size=config.train.batch_size)

    model = SONN(
        config,
        d_model=train_x.shape[1],
        feature_names=list(housing.feature_names),
    )
    model = model.to(config.train.device)

    trainer = Trainer(config, feature_names=list(housing.feature_names))
    trainer.set_seed(model.param.train.seed)

    # --- Train ----------------------------------------------------------------
    trainer.train(model, train_dl, dev_dl, test_dl, resume=bool(config.get("resume", False)))

    # --- Predict on the held-out half -----------------------------------------
    trainer.load_model_checkpoint(model, config.train.device)

    def report(tag: str) -> None:
        """Run inference on the test loader, print MSE / MAE + selected / unselected features."""
        model_out, _ = trainer.infer(model, test_dl)
        y_pred = model_out.cpu().numpy()  # target is raw $100k — no denorm needed
        mse = metrics.mean_squared_error(test_y, y_pred)
        mae = metrics.mean_absolute_error(test_y, y_pred)
        print(f"--- {tag} ---")
        print(f"  mse on test set:           {mse:0.4f}")
        print(f"  mae on test set:           {mae:0.4f}")
        print(f"  selected feature indices:  {model.get_selected_features_indices()}")
        print(f"  unselected feature indices:{model.get_unselected_features_indices()}")
        print(f"  selected features:         {model.get_selected_features()}")
        print(f"  unselected features:       {model.get_unselected_features()}")

    report("Full trained model")

    # --- Plot -----------------------------------------------------------------
    out_dir = Path(__file__).parent
    PlotModel(
        model,
        filename=str(out_dir / "california_housing_model"),
        plot_neuron_name=True,
        view=False,
    ).plot()

    trainer.prune(model)
    report("Pruned model")

    PlotModel(
        model,
        filename=str(out_dir / "california_housing_pruned_model"),
        plot_neuron_name=True,
        view=False,
    ).plot()

    print("Done!")


if __name__ == "__main__":
    main()