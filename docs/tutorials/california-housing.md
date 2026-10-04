# California housing: torchsonn against gradient boosting

This page is the California housing tutorial's README, kept next to its
code in `tutorials/california_housing/`. The script's own settings are
listed in [Tutorial settings](#tutorial-settings) at the end.

Run it from the repository root with one of the two configs the results
below come from, with the candidate early stop the results use:

```bash
python -m tutorials.california_housing.california_housing --config-name california_housing_legendre_finetune \
    train.early_stop_source=train train.checkpoint_dir=checkpoints/california_legendre
python -m tutorials.california_housing.california_housing --config-name california_housing_rbf \
    train.early_stop_source=train train.checkpoint_dir=checkpoints/california_rbf
```

Both configs set `train.checkpoint_dir` relative to the tutorial folder,
which from the repository root points outside it, so the commands set it.
The script draws the network into that folder; the diagrams need the `viz`
extra. Both configs run on the CPU as shipped, and `train.device=cuda`
moves a run to a GPU: the Legendre command took 46 s on an RTX 5080 and
scored a test MSE of 0.1843.

--8<-- "tutorials/california_housing/README.md"

## Tutorial settings

The `tutorial` section of the configuration holds the script's own
settings, read by `tutorial_params()` in `california_housing.py`. A key the
script does not know stops the run with `KeyError`.

| Key | Default | Meaning |
|---|---|---|
| `tutorial.test_size` | 0.2 | Fraction of the rows held out as the test set. |
| `tutorial.dev_split` | `sqMode4_1` | How the remaining rows split into train and dev, a `SequenceTypeSet` name (see [Data](../guides/data.md#deterministic-splits)); `sqMode4_1` sends every 4th row to dev. |
| `tutorial.val_split` | 0.0 | Fraction of the training rows set aside as a validation split; 0 for none (see [Splits and stopping](../concepts/splits-and-stopping.md#the-validation-split)). |
| `tutorial.n_ensemble` | 1 | Number of models to train, each with its own seed and, after the first, its own shuffle of the non-test rows; the script reports each and the mean of their predictions. |
| `tutorial.z_clip` | 5.0 | Standardized features are clipped to ±this value. |
| `tutorial.feature_engineering` | false | Master switch for the two feature groups below; off, the model sees the eight raw columns. Every config in the folder turns it on. |
| `tutorial.log_features` | true | Log-transform the skewed features and add the bedrooms-per-room and rooms-per-person ratios. |
| `tutorial.location_features` | true | Add the rotated coordinates, the log distances to four cities and the kNN price feature. |
| `tutorial.knn_price_k` | 20 | Neighbours averaged by the kNN price feature. |
| `tutorial.log_target` | false | Fit the log of the price instead of the price. |
| `tutorial.clip_predictions` | true | Clip predictions to the range of the training targets. |
| `tutorial.censor_cap` | true | Treat targets at the $500k cap as censored in the training loss (`train.censor_target_at`). |
