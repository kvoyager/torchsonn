"""Iris recognition tutorial — Hydra entry point.

Run from the repo root:
    python -m tutorials.iris.iris_recognition

Hydra overrides work on every config key. Examples:
    python -m tutorials.iris.iris_recognition resume=true
    python -m tutorials.iris.iris_recognition train.batch_size=64 train.optimizer.optimizer_params.lr=5e-3
    python -m tutorials.iris.iris_recognition hydra.run.dir=/tmp/iris_run     # redirect Hydra output dir
"""
from pathlib import Path

import matplotlib.pyplot as plt

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from torchsonn.data.dataset import SONNDataset
from torchsonn.data.preprocessing import train_preprocessing
from torchsonn.logger import setup_logger
from torchsonn.model import SONN
from torchsonn.plot_model import PlotModel
from torchsonn.trainer import Trainer


def iris_class(value):
    if value > 1.5:
        return 2
    elif 0.5 <= value <= 1.5:
        return 1
    else:
        return 0


def plot_confusion_matrix(cm, iris, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# `config_path` is resolved relative to *this file*; `config_name` is the YAML
# filename minus extension. `version_base="1.3"` keeps Hydra in non-chdir mode
# by default so relative paths inside the config still resolve from cwd, and
# silences the version-base deprecation warning.
@hydra.main(version_base="1.3", config_path=".", config_name="iris")
def main(config: DictConfig) -> None:
    # Hydra creates a per-run output directory (default: ./outputs/YYYY-MM-DD/HH-MM-SS/);
    # put our training log alongside Hydra's own .hydra/ artefacts there.
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = setup_logger(str(run_dir / "train.log"))

    logger.info("Loaded config:")
    logger.info(OmegaConf.to_yaml(config))

    iris = load_iris()
    iris.data = iris.data.astype(np.float32)

    n_samples = iris.data.shape[0]

    data = np.empty_like(iris.data)
    target = np.empty_like(iris.target)
    j = 0
    n = n_samples // 3
    for i in range(0, n):
        data[j] = iris.data[i]
        data[j+1] = iris.data[i+n]
        data[j+2] = iris.data[i+2*n]
        target[j] = iris.target[i]
        target[j+1] = iris.target[i+n]
        target[j+2] = iris.target[i+2*n]
        j += 3

    indices = np.arange(data.shape[0])
    train_size = int(0.7 * data.shape[0])
    dev_size = int(0.15 * data.shape[0])

    train_idx = indices[:train_size]
    dev_idx = indices[train_size:train_size + dev_size]
    test_idx = indices[train_size + dev_size:]

    x_train, y_train = data[train_idx], target[train_idx]
    x_dev, y_dev = data[dev_idx], target[dev_idx]
    x_test, y_test = data[test_idx], target[test_idx]

    x_train, y_train = train_preprocessing(x_train, y_train, iris.feature_names)
    x_dev, y_dev = train_preprocessing(x_dev, y_dev, iris.feature_names)
    test_x, test_y = train_preprocessing(x_test, y_test, iris.feature_names)

    train_ds = SONNDataset(x_train, y_train)
    dev_ds = SONNDataset(x_dev, y_dev)
    test_ds = SONNDataset(test_x, test_y)

    train_dl = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=config.train.batch_size)
    test_dl = DataLoader(test_ds, batch_size=config.train.batch_size)

    model = SONN(
        config,
        d_model=x_train.shape[1],
        feature_names=iris.feature_names,
    )
    model = model.to(config.train.device)

    trainer = Trainer(config, feature_names=iris.feature_names)
    trainer.set_seed(model.param.train.seed)

    # `resume` is now a top-level config field (default False); override via:
    #   python -m tutorials.iris.iris_recognition resume=true
    trainer.train(model, train_dl, dev_dl, test_dl, resume=bool(config.get("resume", False)))

    trainer.load_model_checkpoint(model, config.train.device)
    model_out, test_targets = trainer.infer(model, test_dl)
    pred_y = model_out.argmax(dim=1).cpu().numpy()
    test_y = test_targets.cpu().numpy()

    print(f"Selected features indices: {model.get_selected_features_indices()}")
    print(f"Unselected features indices: {model.get_unselected_features_indices()}")
    print(f"Selected features: {model.get_selected_features()}")
    print(f"Unselected features: {model.get_unselected_features()}")

    fig = plt.figure()
    cm = confusion_matrix(test_y, pred_y)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    fig.add_subplot(121)
    plot_confusion_matrix(cm, iris)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    fig.add_subplot(122)
    plot_confusion_matrix(cm_normalized, iris, title='Normalized confusion matrix')

    model.plot_layer_error()
    plt.show(block=False)
    plt.pause(0.1)

    plot_dir = Path(__file__).parent
    PlotModel(model, filename=str(plot_dir / 'iris_model'),
              plot_neuron_name=True, view=False).plot()

    trainer.prune(model)
    model_out2, _ = trainer.infer(model, test_dl)
    PlotModel(model, filename=str(plot_dir / 'iris_pruned_model'),
              plot_neuron_name=True, view=False).plot()
    print("Done!")


if __name__ == "__main__":
    main()
