# Tutorials

Five end-to-end examples live in the `tutorials/` folder of the
repository. Each is a script with its configuration, from a four-feature
classifier that trains in seconds to a cross-validated benchmark with
fine-tuning that runs for hours.

| Tutorial | Task | Families | Shows | Run time |
|---|---|---|---|---|
| [Iris](iris.md) | `multi-class`, 3 classes, 150 rows | `linear_cov` | a whole run in seconds; a shared projection as the class map | about 11 s on a CPU |
| [California housing](california-housing.md) | `regressor`, 20,640 rows | `legendre` or `rbf` | feature engineering, an output head and the end-to-end pass, a comparison with gradient boosting | about 45 s on a GPU |
| [Concrete](concrete.md) | `regressor`, 1,030 rows | `linear_cov`, `quadratic`, `cubic`, `polyquad` | four families side by side, a 30-layer network on a small dataset | about 8 min on a CPU |
| [CCPP](ccpp.md) | `regressor`, 9,568 rows, 5×2 cross-validation | power-basis and `legendre` | the published benchmark protocol, the squash, the three fine-tuning passes | 12 min on a CPU to 14.7 h on a GPU, by config |
| [Otto](otto.md) | `multi-class`, 9 classes, 61,878 rows | `polyquad` | class weights, a projection per candidate, the per-layer fine-tune, a head over 93 survivors | about 2 min on a GPU |

The run times are single runs of the shipped configurations; the CCPP
range covers its configs, each over all ten folds.

## Running them

The tutorials are part of the repository, not of the installed package.
Clone the repository and install it with the `test` extra, which adds
pandas and the `viz` extra (see
[Installation](../getting-started/install.md#development-install)):

```bash
git clone https://github.com/kvoyager/torchsonn.git
cd torchsonn
pip install -e ".[test]"
```

Then run a tutorial from the repository root:

```bash
python -m tutorials.iris.iris_recognition
python -m tutorials.california_housing.california_housing --config-name california_housing_legendre_finetune \
    train.early_stop_source=train train.checkpoint_dir=checkpoints/california
python -m tutorials.concrete.concrete train.checkpoint_dir=checkpoints/concrete
python -m tutorials.ccpp.ccpp
python tutorials/otto/otto_classification.py
```

Iris loads its data from scikit-learn. The others download theirs on the
first run and cache it: California housing through scikit-learn, the
others into a `data` folder next to the script.

Every tutorial except Otto reads its configuration with Hydra, so any key
can be overridden on the command line (see
[Configuration](../guides/configuration.md#command-line-overrides)):

```bash
python -m tutorials.iris.iris_recognition train.max_layer_count=3
python -m tutorials.ccpp.ccpp --config-name ccpp_legendre train.device=cuda
```

Every tutorial except Otto draws its network with Graphviz, which needs
the `viz` extra and the Graphviz programs. Iris, Concrete and CCPP draw
into `.svg` files next to the script, overwriting the copies in the
repository; California housing draws into its checkpoint folder. Each
tutorial page explains its command's extra settings.

<small>Checked against TorchSONN 0.1.5.</small>
