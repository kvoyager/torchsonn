# Otto

Multi-class classification of 61,878 products into 9 categories from 93
count features, on the Otto Group Product Classification dataset. The
score is the multi-class log loss, the metric of the Kaggle competition
the dataset comes from. The tutorial is the largest of the five: a
5-input quadratic family, 93 survivors per layer, a projection per
candidate, the per-layer fine-tune, an output head and class weights, on a
GPU.

```bash
python tutorials/otto/otto_classification.py
```

The script reads `tutorials/otto/otto.yaml` with Hydra's compose API, so it
takes no command-line overrides: change the YAML file to change a setting.
It does not plot, so it runs on a base install. A notebook version,
`tutorials/otto/otto.ipynb`, draws the confusion matrix and the layer
errors inline with matplotlib.

Each run writes its checkpoints and its `train.log` to a folder of its
own, `tutorials/otto/checkpoints/YYYY-MM-DD-HH-MM-SS/`, and the script
prints it at the end. With `resume: true` in `otto.yaml`, the script
continues the newest run folder there that holds a step checkpoint (see
[Training](../guides/training.md#resuming)), so move away run folders of
another configuration first. `resume` is off as shipped.

## Data

On its first run the script downloads the dataset from OpenML (12.4 MB)
and caches it in `tutorials/otto/data/`. It splits the products 70/15/15
into train, dev and test, stratified by class: 43,314, 9,282 and 9,282
rows. The features are counts with heavy tails, so the script takes
`log1p` of them and standardizes them on the training rows; the model
then applies a LayerNorm without learned parameters to every input row
(`SONN(..., preprocessing=...)`).

The classes are imbalanced: the two largest hold about half of the
products, the smallest about 3%. The script weights each class by the
inverse of its frequency (see [Classification](../concepts/classification.md#class-weights)):

```text
class weights: [3.565 0.426 0.859 2.554 2.511 0.486 2.422 0.812 1.387]
```

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
| `train.neuron_selection_method` | `omp_mixed`, threshold 0.2 | survivors whose outputs differ |
| `train.log_layer_diagnostics` | true | logs the survivors' correlations and effective rank |
| `train.max_layer_count` | 3 | |
| `train.device` | `cuda` | |

## Results

One run on an RTX 5080 GPU took about 2 minutes. The search used all three
layers allowed, each better than the one before:

| Layer | 0 | 1 | 2 |
|---|---|---|---|
| Dev error | 0.8318 | 0.7339 | 0.6943 |

After the head fit:

| Split | Log loss | Accuracy |
|---|---|---|
| Train | 0.6144 | 0.7587 |
| Dev | 0.7257 | 0.7271 |
| Test | 0.7317 | 0.7334 |

The script also prints a per-class report and the test confusion matrix.
The class weights trade precision on the large classes for recall on the
small ones: the smallest class is found 68% of the time, at a precision of
43%.

<small>Checked against TorchSONN 0.1.5.</small>
