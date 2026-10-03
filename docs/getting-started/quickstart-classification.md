# Quickstart: classification

This page fits a three-class model to the iris data, one block at a time.
It follows the same steps as the
[regression quickstart](quickstart-regression.md), which explains the parts
the two share in more detail. On a CPU the run takes under a minute.

## 1. Load the data and split it

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x, y = iris.data.astype(np.float32), iris.target
feature_names = list(iris.feature_names)

# 70% train, 15% dev, 15% test, each split with the same class proportions.
x_train, x_rest, y_train, y_rest = train_test_split(
    x, y, test_size=0.30, stratify=y, random_state=0)
x_dev, x_test, y_dev, y_test = train_test_split(
    x_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=0)
```

Iris has 150 flowers, 50 of each of three species, with 4 measurements per
flower: sepal length and width, petal length and width. The targets are
the class indices 0, 1 and 2. As in regression, the train split fits the
neurons, the dev split chooses them and the depth, and the test split
measures the result. Stratifying keeps all three classes in every split.

## 2. Standardize and wrap in loaders

```python
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchsonn import SONNDataset

scaler = StandardScaler().fit(x_train)
x_train, x_dev, x_test = (scaler.transform(a).astype(np.float32)
                          for a in (x_train, x_dev, x_test))

batch = 128  # larger than the 105 training rows
train_dl = DataLoader(SONNDataset(x_train, y_train), batch_size=batch,
                      shuffle=True)
dev_dl = DataLoader(SONNDataset(x_dev, y_dev), batch_size=batch)
test_dl = DataLoader(SONNDataset(x_test, y_test), batch_size=batch)
```

The scaler is fitted on the training rows only. One batch holds the whole
training split, as the LBFGS optimizer prefers.

## 3. Configure the model

```python
from omegaconf import OmegaConf
from torchsonn import SONN

config = OmegaConf.merge(SONN.default_config(), {
    "model": {
        "type": "multi-class",
        "num_classes": 3,
        "ref_functions": ["linear_cov"],
        "nbest_neurons": 4,
        "max_neuron_models": 6,
        "shortcut": False,
    },
    "train": {
        "max_layer_count": 5,
        "steps": 900,
        "eval_step_interval": 5,
        "early_stop_patience": 1e-4,
        "early_stop_tolerance_steps": 5,
        "checkpoint_dir": "checkpoints/iris",
        "optimizer": {
            "name": "lbfgs",
            "optimizer_params": {
                "lr": 0.1, "min_lr": 0.01, "history_size": 10,
            },
        },
    },
})
```

- **`type: multi-class`** and **`num_classes: 3`**: a classifier over
  three classes.
- **`ref_functions`**, **`nbest_neurons`** and **`max_neuron_models`**:
  `linear_cov` pair neurons, 4 survivors per layer. With 4 inputs there are
  exactly 6 pairs, and the cap of 6 tries them all.
- **`shortcut: False`**: every layer after the first reads only the 4
  survivors of the layer before, not the original features.
- **`train`**: the same kind of candidate fit as in the regression
  quickstart, with up to 5 layers and a slightly more patient early stop
  (5 evaluations without improvement before the learning rate halves).

Each neuron outputs one number. The default multi-class head, the **soft
binner**, turns that number into class probabilities. It places the classes
at fixed points between 0.05 and 0.95, in label order, and scores each
class by how close the neuron's output lies to the class's point. The
softmax of those scores gives the probabilities, so each neuron learns to
output the point of the right class. With `soft_binner: False` the model
learns a linear map from the neuron's output to the class scores instead
(see [Classification](../concepts/classification.md)).

For imbalanced classes, `Trainer(config, class_weights=...)` takes a tensor
with one loss weight per class. Iris is balanced and needs none.

## 4. Train

```python
from torchsonn import Trainer
from torchsonn.logger import setup_logger

setup_logger("train.log")
model = SONN(config, d_model=x.shape[1], feature_names=feature_names)
trainer = Trainer(config, feature_names=feature_names)
trainer.set_seed(config.train.seed)
trainer.train(model, train_dl, dev_dl, test_dl)
```

The growth-rule lines of the log read the same way as in regression:

```text
Layer #0: error 0.0195, best 0.0195 at layer 0; first layer; 0 of 5 layers without improvement
Layer #1: error 0.0132, best 0.0132 at layer 1; improved by +0.0063 (margin 0.0000); 0 of 5 layers without improvement
Layer #2: error 0.0172, best 0.0132 at layer 1; improved by -0.0040 (margin 0.0000); 1 of 5 layers without improvement
Layer #3: error 0.0109, best 0.0109 at layer 3; improved by +0.0023 (margin 0.0000); 0 of 5 layers without improvement
Layer #4: error 0.0148, best 0.0109 at layer 3; improved by -0.0039 (margin 0.0000); 1 of 5 layers without improvement
```

For a classifier, the error is the normalized cross-entropy on the dev
split: the average negative log-likelihood of the true class, divided by
the entropy of the class frequencies. 1 is no better than predicting the
class frequencies, and 0 is a perfect fit (see
[Criteria](../concepts/criteria.md)). The search reaches `max_layer_count`
after layer 4, which is worse than layer 3, so the model keeps layers 0 to
3.

## 5. Restore and evaluate

```python
model = SONN(config, d_model=x.shape[1], feature_names=feature_names)
trainer.load_model_checkpoint(model, "cpu")
print("layers kept:", len(model.layers))

from sklearn.metrics import confusion_matrix, log_loss

log_probs, targets = trainer.infer(model, test_dl)
probs = log_probs.exp().numpy()
y_true, y_pred = targets.numpy(), probs.argmax(axis=1)
print(f"test log loss {log_loss(y_true, probs, labels=[0, 1, 2]):.4f}")
print(f"test accuracy {(y_pred == y_true).mean():.3f}")
print(confusion_matrix(y_true, y_pred))
print("features used:", model.get_selected_features())
```

```text
layers kept: 4
test log loss 0.0029
test accuracy 1.000
[[8 0 0]
 [0 8 0]
 [0 0 7]]
features used: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
```

For a multi-class model, `infer` returns log-probabilities, one column per
class, so `exp()` gives the probabilities. The model classifies all 23 test
flowers correctly. With 23 test rows, that shows the model works on this
data; it is too few rows to measure its accuracy precisely.

## What to change next

- **The head**: `soft_binner: False` replaces the soft binner with a
  learned projection shared by a layer's candidates; adding
  `use_neuron_proj: True` (which needs `soft_binner: False`) gives every
  candidate its own. `use_output_projection` adds a linear head over the
  last layer's survivors (see [Classification](../concepts/classification.md)).
- **Class weights**: pass `class_weights` to the `Trainer` when some
  classes are rare.
- **The search**: more neuron families, more survivors and deeper networks,
  as in the [regression quickstart](quickstart-regression.md#what-to-change-first).
  The [Otto tutorial](../tutorials/otto.md) is a larger classification
  example, scored by multi-class log loss.
