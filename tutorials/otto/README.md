# Otto Group product classification: results and comparisons

This tutorial fits a self-organizing neural network, TorchSONN, to the
Otto Group product classification task: 9 product categories from 93
count features, scored by the multi-class log loss. It then compares the
result with single hidden layer MLPs and XGBoost on the same split and
features. This page covers the dataset, the tutorial's setup, the deep
SONN that is its main result, what depth and the end-to-end pass are
worth, the layer by layer retune the deep models build on, calibration,
and where the SONN stands against the other models. The short version of
the tutorial, "run it and read the output", is the docs page
[Otto Group product classification](../../docs/tutorials/otto.md) [[3]](#ref-3).

The answer in one line: a two layer SONN trained end to end,
`otto_deep.yaml` (14,469 to 14,742 parameters), scores a mean test log
loss of **0.5130** and an accuracy of **0.8026** over five seeds. That
ties an MLP of the same size tuned with weight decay and dropout (0.5146 /
0.8022; the SONN is lower on four seeds of five), and beats every other
model of its size on every seed: an MLP without regularization, XGBoost
capped to the same size, and the SONN with one layer (0.5270) or three
(0.5199). XGBoost without a size limit, at about 1.2M parameters, leads at
0.4685.

## The dataset

The data comes from the Otto Group Product Classification Challenge, a
Kaggle competition of 2015 [[1]](#ref-1). Each row is a product,
described by 93 anonymized features that count events of some kind, and
labeled with one of 9 product categories. Only the competition's training
file is public: 61,878 products. The tutorial downloads it from OpenML
(12.4 MB) [[2]](#ref-2) and caches it in `tutorials/otto/data/`.

The classes are imbalanced:

| class | products | share |
|---|---|---|
| Class_1 | 1,929 | 3.1% |
| Class_2 | 16,122 | 26.1% |
| Class_3 | 8,004 | 12.9% |
| Class_4 | 2,691 | 4.3% |
| Class_5 | 2,739 | 4.4% |
| Class_6 | 14,135 | 22.8% |
| Class_7 | 2,839 | 4.6% |
| Class_8 | 8,464 | 13.7% |
| Class_9 | 4,955 | 8.0% |

The features are small integer counts with heavy tails: most entries are
zero, a few are large.

## Published results

The competition scored submissions by the multi-class log loss on a test
set whose labels were never released. The leaderboard scores are on that
hidden test set, and the top ones come from large ensembles of models.
This page splits the public training file instead, so its numbers are not
comparable with the leaderboard, and it does not quote it. The fair
comparison is the baselines run on the tutorial's own split, at the end of
this page.

## The tutorial's setup

- **Split.** The seed (`train.seed`) draws a stratified 70/15/15 split
  of the products: 43,314 train rows, which fit the neurons and the head,
  9,282 dev rows, which select the survivors and stop the fits, and 9,282
  test rows. The seed also draws the model, so every model on this page
  that runs with seed 42 is scored on the same test rows.
- **Features.** Each count goes through `log1p` and is then standardized
  with the training rows' mean and standard deviation. The model applies a
  LayerNorm without learned parameters to every input row before its first
  layer.
- **No class weights.** The score is the unweighted log loss, and
  inverse-frequency class weights would make the model over-predict the
  small classes, which costs log loss and accuracy.
- **Metrics.** The test log loss (sklearn's `log_loss`), then the
  accuracy.
- **Parameters.** The count is every tensor of the model, `model.parameters()`,
  after the best checkpoint is loaded.

## Running it

From any directory:

```bash
python tutorials/otto/otto_classification.py --config-name otto_deep
python tutorials/otto/otto_classification.py --config-name otto_deep train.seed=43
python tutorials/otto/otto_classification.py --config-name otto_deep3
python tutorials/otto/otto_classification.py --config-name otto_tuned
python tutorials/otto/otto_classification.py
```

The script reads `otto.yaml` unless `--config-name` picks another YAML
file in the folder, and `key=value` overrides any setting. Each run writes
its checkpoints and its `train.log` to a folder of its own,
`tutorials/otto/checkpoints/YYYY-MM-DD-HH-MM-SS/`, and logs the results, a
per-class report and the test confusion matrix. The configs run on a GPU
as shipped. On an RTX 5080 a run takes 5 to 6 minutes with `otto_deep.yaml`
or `otto_deep3.yaml`, about 2.5 minutes with `otto_tuned.yaml` and 3 to 5
minutes with `otto.yaml`, the quick run of the docs page and of the
notebook `tutorials/otto/otto.ipynb`.

## The main result: a deep SONN

`otto_deep.yaml` builds two layers and then trains the whole network at
once. It changes `otto.yaml` as follows:

| Setting | `otto.yaml` | `otto_deep.yaml` |
|---|---|---|
| `train.max_layer_count`, `train.layer_keep` | 3, `best` | 2, `all` |
| `model.nbest_neurons`, `model.num_out_neurons` | 93, 93 | 360, 256 |
| `model.use_neuron_proj` | true | false |
| `train.out_proj_train.weight_decay` | 2.5e-5 | 2e-3 |
| `train.neuron_selection_method`, `train.neuron_selection_min_gain` | `omp_mixed`, 1e-3 | `omp`, 0 |
| `finetune_end_to_end`, `finetune_prune_first` | false, false | true, true |
| `train.finetune_train` | | AdamW, lr 1e-2, weight decay 0.5, dropout 0.4 |

How the run goes:

1. **The search.** It builds two layers of 360 survivors and keeps both
   (`layer_keep: all`), each neuron a quadratic polynomial of 5 inputs.
   The second layer's neurons read the first layer's outputs and the
   network's inputs. Survivors are chosen by OMP for how much of the class
   targets they explain, with no minimum gain, so both layers fill up.
   One class map shared by all neurons scores the candidates, and the head
   fit and the per-layer fine-tune use L2 2e-3; both settings come from
   the one layer retune below.
2. **The head.** A linear head reads the 256 best survivors of the second
   layer.
3. **Pruning.** The model is cut to the neurons the head reads, which
   leaves 322 to 335 neurons in the first layer.
4. **The end-to-end pass.** Every parameter trains at once, the head
   included, with AdamW (learning rate 1e-2, decoupled weight decay 0.5)
   and dropout 0.4 on the head's inputs. The pass evaluates dev every 100
   steps, stops after 10 evaluations without a gain and keeps the best
   weights.

Over five seeds, log loss / accuracy, against the tuned MLP of the
comparison below:

| seed | parameters | dev | test | MLP tuned, test | difference |
|---|---|---|---|---|---|
| 42 | 14,553 | 0.5013 / 0.8094 | 0.5106 / 0.8033 | 0.5108 / 0.8036 | -0.0002 |
| 43 | 14,742 | 0.4945 / 0.8098 | 0.5030 / 0.8053 | 0.5075 / 0.8084 | -0.0045 |
| 44 | 14,574 | 0.5127 / 0.8054 | 0.5193 / 0.8032 | 0.5220 / 0.8000 | -0.0027 |
| 45 | 14,469 | 0.5223 / 0.8002 | 0.5197 / 0.7977 | 0.5205 / 0.7963 | -0.0008 |
| 46 | 14,469 | 0.5151 / 0.7995 | 0.5122 / 0.8036 | 0.5121 / 0.8027 | +0.0001 |
| **mean** | | 0.5092 / 0.8049 | **0.5130 / 0.8026** | 0.5146 / 0.8022 | **-0.0016** |

The deep SONN is lower than the MLP on four seeds of five and on the
mean, with 614 to 887 fewer parameters. The differences are small next to
their spread across seeds (standard deviation 0.0019, a paired t of about
-1.9 with 4 degrees of freedom), so this is a tie, not a win.

The errors on seed 42 follow the class sizes. The smallest class,
Class_1, is found 46% of the time, at a precision of 65%. Class_2 and
Class_3 taken for each other are 46% of all errors.

## Depth and the end-to-end pass

At about 15,000 parameters the SONN can be one wide layer or several
narrower ones. Trained layer by layer, depth loses; with the end-to-end
pass, it wins. The three configurations:

- **1 layer:** `otto_tuned.yaml`, 512 survivors, all read by the head,
  15,387 parameters, no end-to-end pass (see the next section);
- **2 layers:** `otto_deep.yaml`, above;
- **3 layers:** `otto_deep3.yaml`, `otto_deep.yaml` with three layers of
  220 survivors, the head reading all 220 of the last, every layer after
  the first fed the outputs of all earlier layers
  (`model.shortcut.prev_layers: all`) and a pass at learning rate 3e-3;
  14,712 to 15,174 parameters.

Five seeds, test log loss:

| seed | 1 layer | 2 layers | 3 layers |
|---|---|---|---|
| 42 | 0.5216 | 0.5106 | 0.5161 |
| 43 | 0.5192 | 0.5030 | 0.5124 |
| 44 | 0.5299 | 0.5193 | 0.5220 |
| 45 | 0.5390 | 0.5197 | 0.5314 |
| 46 | 0.5255 | 0.5122 | 0.5174 |
| **mean, with accuracy** | 0.5270 / 0.7985 | **0.5130 / 0.8026** | 0.5199 / 0.8015 |

Both deep models beat the one layer model on every seed: two layers by
0.014 on average, three by 0.007. At this budget two wide layers beat
three narrower ones on every seed.

What the pass is worth, seed 42, test log loss / accuracy:

| model | without the pass | with the pass |
|---|---|---|
| 1 layer | 0.5216 / 0.8019 | 0.5177 / 0.8007 |
| 2 layers | 0.5391 / 0.7942 | **0.5106 / 0.8033** |
| 3 layers | 0.5387 / 0.7934 | 0.5161 / 0.8024 |

Without the pass both deep models lose to the one layer model, by 0.017.
The pass gains 0.004 on one layer and 0.023 to 0.029 on two or three,
enough to reverse the order. The rows without the pass set
`finetune_end_to_end=false`; the one layer row with it adds the pass of
`otto_deep.yaml` to `otto_tuned.yaml`:

```bash
python tutorials/otto/otto_classification.py --config-name otto_deep finetune_end_to_end=false
python tutorials/otto/otto_classification.py --config-name otto_deep3 finetune_end_to_end=false
python tutorials/otto/otto_classification.py --config-name otto_tuned finetune_end_to_end=true \
    finetune_prune_first=true train.finetune_train.optimizer=adamw train.finetune_train.lr=1e-2 \
    train.finetune_train.weight_decay=0.5 train.finetune_train.dropout=0.4 \
    train.finetune_train.max_steps=20000 train.finetune_train.eval_interval=100 \
    train.finetune_train.early_stop_patience=10 train.finetune_train.keep_best_weights=true
```

## Layer by layer: the one layer retune

The deep configurations reuse two settings found by tuning the one layer
SONN without the end-to-end pass: one class map shared by all neurons and
L2 2e-3 in the head fit. That retune went from `otto.yaml` to
`otto_tuned.yaml`, one setting at a time, each chosen by the mean dev log
loss over seeds 42 to 44. Each row adds one change to the row above it;
means over seeds 42 to 44:

| step | parameters | dev log loss | test log loss / accuracy |
|---|---|---|---|
| `otto.yaml` | 4,473 | 0.5886 | 0.5962 / 0.7779 |
| one layer | 4,473 | 0.5789 | 0.5873 / 0.7793 |
| L2 1e-3 in the head fit and the per-layer fine-tune | 4,473 | 0.5619 | 0.5706 / 0.7820 |
| 512 survivors, all read by the head | 24,585 | 0.5366 | 0.5430 / 0.7948 |
| L2 2e-3 | 24,585 | 0.5191 | 0.5271 / 0.7968 |
| one class map shared by all neurons (`otto_tuned.yaml`) | 15,387 | 0.5157 | 0.5236 / 0.8003 |

The three levers:

- **Survivors.** More of them is the largest single gain, 0.028 test log
  loss, at five and a half times the parameters.
- **L2 on the head.** `train.out_proj_train.weight_decay` applies to the
  head fit and to the per-layer fine-tune, where the layer's survivors and
  a temporary head train together. The shipped 2.5e-5, about the default
  penalty of scikit-learn's logistic regression at this many rows, mostly
  keeps the head's weights finite; 1e-3 at 93 survivors and 2e-3 at 512
  gain 0.017 and 0.016.
- **The shared class map.** With `use_neuron_proj: false`, one affine map
  (9 weights and 9 biases) turns any neuron's output into class scores to
  score it as a candidate, in place of a map per candidate. It drops 9,198
  parameters, 37% of the model, and the log loss does not suffer for it.

`otto_tuned.yaml` is one layer of 512 neurons with 21 coefficients each
(a constant, 5 linear terms, 5 squares and 10 cross products), 10,752 in
all, the shared class map (18) and a head of 512 × 9 + 9 = 4,617: 15,387
parameters. Over five seeds, log loss / accuracy:

| seed | dev | test |
|---|---|---|
| 42 | 0.5145 / 0.8045 | 0.5216 / 0.8019 |
| 43 | 0.5047 / 0.8075 | 0.5192 / 0.8012 |
| 44 | 0.5278 / 0.7975 | 0.5299 / 0.7977 |
| 45 | 0.5267 / 0.7991 | 0.5390 / 0.7916 |
| 46 | 0.5313 / 0.7967 | 0.5255 / 0.8003 |
| **mean** | 0.5210 / 0.8011 | **0.5270 / 0.7985** |

The shipped `otto.yaml` scores 0.5945 / 0.7769 over the same five seeds.
Its search keeps one layer on seeds 42 to 44 (4,473 parameters) and two
on seeds 45 and 46 (8,100).

## Calibration

A model whose probabilities are too confident pays for it in log loss.
`train.calibration=temperature` fits one temperature T to the model's
log-probabilities on dev after training; T above 1 softens them and T
below 1 sharpens them. The predicted class does not change, so the
accuracy stays the same. Fitted on the five seeds' checkpoints:

| model | T | test log loss, mean | calibrated |
|---|---|---|---|
| `otto_deep.yaml` | 1.02 to 1.06 | 0.5130 | 0.5124 |
| `otto_deep3.yaml` | 1.06 to 1.09 | 0.5199 | 0.5183 |
| `otto_tuned.yaml` | 0.99 to 1.03 | 0.5270 | 0.5269 |
| `otto.yaml` | 1.14 to 1.22 | 0.5945 | 0.5864 |

The shipped model is overconfident, and a temperature recovers 0.008 of
log loss. The others gain 0.002 at most: their L2 penalties already keep
the probabilities in line, and a temperature does not change their order.
The retune steps show the same: 512 survivors at L2 1e-3 come out with T
of 1.15 to 1.17, and L2 2e-3 brings it back to 1.00 to 1.03. The stronger
penalty is worth more than a temperature, though: on the 512 survivor
model a temperature recovers 0.007, the stronger penalty 0.016.

```bash
python tutorials/otto/otto_classification.py --config-name otto_deep train.calibration=temperature
```

## Against an MLP and XGBoost, same data

`baselines.py` trains the other models on the tutorial's split and
features, without class weights, scored the same way:

```bash
python tutorials/otto/baselines.py mlp
python tutorials/otto/baselines.py mlp --hidden 149 --activation gelu --weight-decay 0 --dropout 0
python tutorials/otto/baselines.py mlp --hidden 256 --activation silu --weight-decay 1e-4 --dropout 0
python tutorials/otto/baselines.py mlp --hidden 2048 --activation sigmoid --weight-decay 1e-4 --dropout 0
python tutorials/otto/baselines.py xgboost
python tutorials/otto/baselines.py xgboost --max-params 15356 --max-depth 3 --learning-rate 0.7
```

The MLPs have one hidden layer and read the same LayerNorm rows as the
SONN. They train with AdamW at learning rate 1e-3 on batches of 512,
halve the learning rate when the dev log loss stalls for 5 epochs, and
keep their best dev epoch. XGBoost grows trees of depth 8 at learning rate
0.05 and keeps its best dev round. `--max-params` caps it: it keeps the
best dev round among those whose trees hold at most that many parameters.
The capped run's depth and learning rate are the best by mean dev log
loss on seeds 42 to 44, among depths 1 to 5 and learning rates 0.1 to
1.0. XGBoost needs `pip install xgboost`. On an RTX 5080 a seed of the
tuned MLP takes about 20 seconds and one of XGBoost about 30.

Five seeds, 42 to 46, mean test log loss / accuracy:

| model | parameters | test |
|---|---|---|
| **torchsonn `otto_deep.yaml`, 2 layers, end-to-end pass** | 14,469 to 14,742 | **0.5130 / 0.8026** |
| torchsonn `otto_deep3.yaml`, 3 layers, end-to-end pass | 14,712 to 15,174 | 0.5199 / 0.8015 |
| torchsonn `otto_tuned.yaml`, 1 layer × 512 | 15,387 | 0.5270 / 0.7985 |
| torchsonn `otto.yaml` | 4,473 to 8,100 | 0.5945 / 0.7769 |
| MLP 149 `gelu`, weight decay 0.5, dropout 0.2 (tuned) | 15,356 | 0.5146 / 0.8022 |
| MLP 149 `gelu`, no weight decay, no dropout | 15,356 | 0.5318 / 0.7974 |
| MLP 256 `silu`, weight decay 1e-4 | 26,377 | 0.5273 / 0.7974 |
| MLP 2048 `sigmoid`, weight decay 1e-4 | 210,953 | 0.5020 / 0.8115 |
| XGBoost, at most 15,356 parameters | 14,916 to 15,249 | 0.5485 / 0.7916 |
| XGBoost | 1.14M to 1.23M | 0.4685 / 0.8225 |

"Parameters" counts everything the model learned:

- **SONN:** every tensor of the model after its last pass, which for the
  deep models comes after pruning to the neurons the head reads;
- **MLP:** every weight and bias; 93 inputs, 149 hidden units and 9
  outputs give 93·149 + 149 + 149·9 + 9 = 15,356;
- **XGBoost:** a feature and a threshold per split and a value per leaf,
  over the trees up to its best round, 9 trees a round.

Paired by seed, test log loss:

| seed | SONN 2 layers | SONN 1 layer | MLP tuned | MLP, no regularization | MLP 256 | XGBoost capped | XGBoost |
|---|---|---|---|---|---|---|---|
| 42 | 0.5106 | 0.5216 | 0.5108 | 0.5283 | 0.5237 | 0.5351 | 0.4584 |
| 43 | 0.5030 | 0.5192 | 0.5075 | 0.5242 | 0.5212 | 0.5516 | 0.4671 |
| 44 | 0.5193 | 0.5299 | 0.5220 | 0.5385 | 0.5322 | 0.5518 | 0.4740 |
| 45 | 0.5197 | 0.5390 | 0.5205 | 0.5366 | 0.5345 | 0.5549 | 0.4740 |
| 46 | 0.5122 | 0.5255 | 0.5121 | 0.5317 | 0.5250 | 0.5492 | 0.4693 |

At about the same size:

- **The tuned MLP ties the deep SONN.** The SONN is lower on four seeds
  of five, by 0.0016 on the mean, within the seed spread. Against the one
  layer SONN the MLP wins on every seed, by 0.012 on average. Weight decay
  and dropout are worth 0.017 to the MLP; in the SONN the L2 on the head
  and the pass's weight decay and dropout play that part.
- **The MLP without regularization and the MLP of 256 units lose to the
  deep SONN on every seed**, the second with 80% more parameters.
- **The capped XGBoost loses to every SONN on every seed**, to the deep
  one by 0.024 to 0.049.

Without a size limit, the MLP of 2048 units gets to 0.5020 at 14 times
the deep SONN's size, and XGBoost leads everything on this page by a wide
margin, at about 80 times.

## Summary

torchsonn does not beat full size XGBoost on Otto; at the size of a
tuned MLP, a deep SONN trained end to end ties it. On identical splits and
features, over five seeds:

- XGBoost reaches a mean test log loss of **0.469**, at about 1.2M
  parameters;
- the deep SONN, two layers with the end-to-end pass, reaches **0.513**
  with 14,469 to 14,742 parameters;
- the tuned MLP of 15,356 parameters reaches **0.515**;
- the SONN with three layers and the pass reaches **0.520**;
- the one layer SONN of 15,387 parameters, trained layer by layer,
  reaches **0.527**;
- XGBoost capped at the MLP's size reaches **0.549**.

Depth pays only with the end-to-end pass. Layer by layer, two or three
layers lose to one wide layer at the same budget; with the pass, both beat
it on every seed, and two wide layers do best. The layer by layer retune
still matters: its shared class map and L2 2e-3 carry over into the deep
models.

What the SONN offers that the MLP does not:

- every neuron is an explicit quadratic polynomial of 5 inputs, the
  network's own inputs or earlier neurons, so each feature the head reads
  can be written down;
- the head is a linear map of those features, one weight per neuron and
  class;
- pruning cuts the network to the neurons the head reads, without moving
  a prediction.

## References

1. <a id="ref-1"></a>Kaggle, *Otto Group Product Classification
   Challenge* (2015):
   <https://www.kaggle.com/competitions/otto-group-product-classification-challenge>
2. <a id="ref-2"></a>OpenML dataset 45548,
   *Otto-Group-Product-Classification-Challenge*, the competition's
   training file: <https://www.openml.org/d/45548>
3. <a id="ref-3"></a>The docs page [Otto Group product classification](../../docs/tutorials/otto.md)
   and the results in the headers of [`otto_deep.yaml`](otto_deep.yaml),
   [`otto_deep3.yaml`](otto_deep3.yaml), [`otto_tuned.yaml`](otto_tuned.yaml)
   and [`otto.yaml`](otto.yaml).
