# Otto

Multi-class classification of 61,878 products into 9 categories from 93
count features, on the Otto Group Product Classification dataset. The
score is the multi-class log loss, the metric of the Kaggle competition
the dataset comes from. The tutorial is the largest of the five: a
5-input quadratic family, 93 survivors per layer, a projection per
candidate, the per-layer fine-tune and an output head, on a GPU. How the tuned
configuration was found, its calibration and its comparison with MLPs and
XGBoost are on the page
[Otto: against an MLP and XGBoost](otto-comparison.md).

```bash
python tutorials/otto/otto_classification.py
```

The script reads `tutorials/otto/otto.yaml` with Hydra's compose API and
takes the same command-line arguments as the other tutorials: `key=value`
overrides any setting, and `--config-name` picks another YAML file in the
same folder. Three ship with it: `otto_tuned.yaml` (see
[A tuned configuration](#a-tuned-configuration)), and `otto_deep.yaml` and
`otto_deep3.yaml` (see [A deeper configuration](#a-deeper-configuration)):

```bash
python tutorials/otto/otto_classification.py --config-name otto_tuned
python tutorials/otto/otto_classification.py --config-name otto_deep
python tutorials/otto/otto_classification.py train.seed=43
```

It does not plot, so it runs on a base install. A notebook version,
`tutorials/otto/otto.ipynb`, draws the confusion matrix and the layer
errors inline with matplotlib.

Each run writes its checkpoints and its `train.log` to a folder of its
own, `tutorials/otto/checkpoints/YYYY-MM-DD-HH-MM-SS/`, and the script
logs it at the end. The log holds the trainer's records, the final line of
every progress bar and the results. With `resume: true`, the script
continues the newest run folder there that holds a step checkpoint (see
[Training](../guides/training.md#resuming)), so move away run folders of
another configuration first. `resume` is off as shipped.

After the results, the script prunes the model to the neurons the head
reads (see [Pruning](../concepts/heads-and-finetune.md#pruning)) and
logs the pruned size with the largest change of a test probability, which
stays at zero up to rounding.

With `finetune_end_to_end=true` the script runs the
[end-to-end pass](../concepts/heads-and-finetune.md#the-end-to-end-pass) after
the head fit: every parameter trains at once, with
`train.finetune_train`, starting from the best checkpoint, and the results
are those of the model it ends with. `finetune_prune_first=true` prunes the
model before the pass, so it trains only the neurons the head reads.

With `train.calibration=temperature` or `train.calibration=vector` the
script fits a [calibration](../concepts/classification.md#calibration) of
the probabilities on dev after the last training pass, and the results are
those of the calibrated model. It is off as shipped.

## Data

On its first run the script downloads the dataset from OpenML (12.4 MB)
and caches it in `tutorials/otto/data/`. It splits the products 70/15/15
into train, dev and test, stratified by class: 43,314, 9,282 and 9,282
rows. The features are counts with heavy tails, so the script takes
`log1p` of them and standardizes them on the training rows; the model
then applies a LayerNorm without learned parameters to every input row
(`SONN(..., preprocessing=...)`).

The classes are imbalanced: the two largest hold about half of the
products, the smallest about 3%. The model trains without class weights
all the same. The score is the unweighted log loss, and inverse-frequency
weights would make the model predict as if every class were equally
common, which over-predicts the small classes and worsens both the log
loss and the accuracy (see [Classification](../concepts/classification.md#class-weights)).

## The configuration

| Setting | Value | Why |
|---|---|---|
| `model.type`, `model.num_classes` | `multi-class`, 9 | |
| `model.ref_functions` | `polyquad` with `dim: 5` and squares | quadratic neurons over 5 of the inputs at a time |
| `model.nbest_neurons`, `model.max_neuron_models` | 93, 10000 | a wide layer and up to 10,000 candidates per layer |
| `model.soft_binner`, `model.use_neuron_proj` | false, true | a projection per candidate turns its output into class scores (see [Classification](../concepts/classification.md#a-projection-per-candidate)) |
| `model.use_output_projection`, `model.num_out_neurons` | true, 93 | a head over all 93 survivors makes the prediction |
| `train.layer_finetune` | true | each layer's survivors train together through a temporary head (see [Heads and fine-tuning](../concepts/heads-and-finetune.md#the-per-layer-fine-tune)) |
| `train.out_proj_train` | `lbfgs`, `weight_decay` 2.5e-5 | the head fit and the per-layer fine-tune; the small L2 penalty keeps the head's weights finite on nearly separable classes |
| `train.optimizer`, `train.steps` | `adam`, `lr` 1e-3, `min_lr` 1e-4; 1101 | batches of 512 rows, evaluated every 100 steps; no candidate fit stops early, so each runs 1,101 steps |
| `train.neuron_selection_method` | `omp_mixed`, reserve 0.2, threshold 0.05 | 80% of the survivors chosen for how much of the class targets they explain, the rest for outputs that differ from them (see [Survivor selection](../concepts/selection.md)) |
| `train.log_layer_diagnostics` | true | logs the survivors' correlations and effective rank |
| `train.max_layer_count` | 3 | |
| `train.device` | `cuda` | |

## Results

One run on an RTX 5080 GPU took about 3 minutes. The search built all three
layers allowed, and neither later layer beat the first, so the model keeps
layer 0 alone:

| Layer | 0 | 1 | 2 |
|---|---|---|---|
| Dev error | 0.7371 | 0.7553 | 0.7559 |

After the head fit:

| Split | Log loss | Accuracy |
|---|---|---|
| Train | 0.4996 | 0.8052 |
| Dev | 0.5834 | 0.7849 |
| Test | 0.5852 | 0.7825 |

The script also prints a per-class report and the test confusion matrix.
The small classes are the hard ones: the smallest is found 43% of the time,
at a precision of 55%. The largest share of the errors, about 43%, is
`Class_2` and `Class_3` taken for each other.

## A tuned configuration

`tutorials/otto/otto_tuned.yaml` starts from `otto.yaml` and changes five
settings, each chosen on seeds 42 to 44 by the mean dev log loss:

| Setting | `otto.yaml` | `otto_tuned.yaml` | Why |
|---|---|---|---|
| `train.max_layer_count` | 3 | 1 | at the same parameter count, deeper layers fit worse than one wide layer |
| `model.nbest_neurons`, `model.num_out_neurons` | 93, 93 | 512, 512 | more survivors, all read by the head |
| `train.out_proj_train.weight_decay` | 2.5e-5 | 2e-3 | a stronger L2 penalty in the head fit and the per-layer fine-tune |
| `model.use_neuron_proj` | true | false | one class map shared by all neurons in place of one per candidate, 37% fewer parameters (see [Classification](../concepts/classification.md#a-projection-per-candidate)) |

The model has 15,387 parameters. One run on an RTX 5080 GPU took about
2.5 minutes. Over five seeds, set with `train.seed`:

| Seed | Dev | Test |
|---|---|---|
| 42 | 0.5145 / 0.8045 | 0.5216 / 0.8019 |
| 43 | 0.5047 / 0.8075 | 0.5192 / 0.8012 |
| 44 | 0.5278 / 0.7975 | 0.5299 / 0.7977 |
| 45 | 0.5267 / 0.7991 | 0.5390 / 0.7916 |
| 46 | 0.5313 / 0.7967 | 0.5255 / 0.8003 |
| Mean | 0.5210 / 0.8011 | 0.5270 / 0.7985 |

Each cell is log loss / accuracy. On seed 42 the test log loss drops from
0.5852 for `otto.yaml` to 0.5216, and the accuracy rises from 0.7825 to
0.8019. The smallest class is found 54% of the time, at a precision of
69%, and `Class_2` and `Class_3` taken for each other are 46% of the
errors.

## A deeper configuration

`tutorials/otto/otto_deep.yaml` builds two layers of 360 survivors and
keeps both, with the head reading 256 of the second, one class map shared
by all neurons and L2 2e-3 in the head fit. The script then prunes the
model to the neurons the head reads and runs the
[end-to-end pass](../concepts/heads-and-finetune.md#the-end-to-end-pass)
with AdamW (lr 1e-2, weight decay 0.5) and dropout 0.4 on the head's
inputs. A run took 5 to 6 minutes on an RTX 5080. Over seeds 42 to 46 it
keeps 14,469 to 14,742 parameters and scores a mean test log loss of
0.5130 at an accuracy of 0.8026 (dev 0.5092 / 0.8049). Without the pass,
`finetune_end_to_end=false`, seed 42 scores 0.5391 / 0.7942, against
0.5106 / 0.8033 with it.

`tutorials/otto/otto_deep3.yaml` is the same with three layers of 220
survivors, each layer after the first fed the outputs of all earlier
layers, the head reading all 220 of the last and the pass at learning rate
3e-3. Over the same seeds it keeps 14,712 to 15,174 parameters and scores
0.5199 / 0.8015 (dev 0.5163 / 0.8048); without the pass, seed 42 scores
0.5387 / 0.7934. Both deep models beat `otto_tuned.yaml` on every seed.
The settings and the comparison of depths are on the page
[Otto: against an MLP and XGBoost](otto-comparison.md#the-main-result-a-deep-sonn).

## Baselines

`tutorials/otto/baselines.py` trains two other models on the same data
(compared seed by seed in
[Otto: against an MLP and XGBoost](otto-comparison.md#against-an-mlp-and-xgboost-same-data)):
the split drawn by the seed, the log1p and standardized features, no class
weights and the same test score.

```bash
python tutorials/otto/baselines.py mlp
python tutorials/otto/baselines.py xgboost
```

`mlp` is a network with one hidden layer, trained with AdamW and an early
stop on the dev log loss. Its defaults are the tuned settings: 149 `gelu`
units, weight decay 0.5 and dropout 0.2. `--hidden`, `--activation`,
`--weight-decay` and `--dropout` change them. `xgboost` grows trees of
depth 8 with learning rate 0.05 until the dev log loss stops improving; it
needs `pip install xgboost`. `--max-params` caps its size: it keeps the
best dev round among those whose trees hold at most that many parameters,
and `--max-depth` and `--learning-rate` change the trees. Both run seeds 42 to 46 unless `--seeds` says
otherwise, and log each seed and the mean. The seed draws the split and the
model, so seed 42 of a baseline is scored on the same test rows as the
SONN run with `train.seed=42`.

Means over seeds 42 to 46, test log loss / accuracy:

| Model | Parameters | Test |
|---|---|---|
| `otto_deep.yaml` | 14,469 to 14,742 | 0.5130 / 0.8026 |
| `otto_deep3.yaml` | 14,712 to 15,174 | 0.5199 / 0.8015 |
| `otto_tuned.yaml` | 15,387 | 0.5270 / 0.7985 |
| `otto.yaml` | 4,473 to 8,100 | 0.5945 / 0.7769 |
| MLP, 149 `gelu` units, weight decay 0.5, dropout 0.2 | 15,356 | 0.5146 / 0.8022 |
| MLP, 149 `gelu` units, no weight decay, no dropout | 15,356 | 0.5318 / 0.7974 |
| MLP, 256 `silu` units, weight decay 1e-4, no dropout | 26,377 | 0.5273 / 0.7974 |
| MLP, 2048 `sigmoid` units, weight decay 1e-4, no dropout | 210,953 | 0.5020 / 0.8115 |
| XGBoost | 1.14M to 1.23M | 0.4685 / 0.8225 |
| XGBoost, at most 15,356 parameters, depth 3, learning rate 0.7 | 14,916 to 15,249 | 0.5485 / 0.7916 |

An MLP counts every weight and bias. XGBoost counts a feature and a
threshold per split and a value per leaf, over the trees up to its best
round. `otto.yaml` keeps one layer on seeds 42 to 44 and two on seeds 45
and 46, hence its range. The capped XGBoost has the tuned MLP's size; its
depth and learning rate are the ones with the lowest mean dev log loss on
seeds 42 to 44 among depths 1 to 5 and learning rates 0.1 to 1.0:

```bash
python tutorials/otto/baselines.py xgboost --max-params 15356 --max-depth 3 --learning-rate 0.7
```

<small>Checked against TorchSONN 0.1.7.</small>
