# Configuration keys

Every key of the configuration schema, `torchsonn.config.SONNConfig`,
grouped by what it controls. A configuration is the schema's defaults with
your changes merged in (see [Configuration](../guides/configuration.md)). In
YAML, `null` stands for no value.

Two tables at the end list what the schema leaves open: the options of a
`model.ref_functions` entry, and the keys of
`train.optimizer.optimizer_params`.

## Model type and output

| Key | Type | Default | Meaning |
|---|---|---|---|
| `model.type` | str | `multi-class` | `regressor`, `binary` or `multi-class`. Sets the training loss, the criterion and what `infer` returns. |
| `model.num_classes` | int | 3 | Number of classes. Must be 2 for `binary` and more than 2 for `multi-class`. |
| `model.use_output_projection` | bool | false | Add a linear head over the last layer's survivors; fit it with `Trainer.train_out_proj`. Regression and multi-class only: a binary model ignores it. See [Heads and fine-tuning](../concepts/heads-and-finetune.md). |
| `model.num_out_neurons` | int | null | How many of the last layer's survivors the head reads, best first. Null reads `model.nbest_neurons`, every survivor. A head that reads more columns than the layer has gets zeros for the rest. |
| `model.output_clamp_value` | float | 1000.0 | Every layer's outputs are clamped to ±this value before the next layer reads them. |
| `model.use_layer_norm` | bool | false | Apply a LayerNorm without learned parameters to each layer's input, after the outputs and features are concatenated. `Trainer.prune` refuses such a model (see [Pruning](../concepts/heads-and-finetune.md#pruning)). |

## Neuron families and layer inputs

| Key | Type | Default | Meaning |
|---|---|---|---|
| `model.ref_functions` | list | `[linear_cov]` | The neuron families. Each entry is a family name or a name with a mapping of options; see [Neuron family options](#neuron-family-options) and [Neuron families](../concepts/neurons/index.md). |
| `model.nbest_neurons` | int | null, required | Survivors per layer. Must be at least 2. See [Survivor selection](../concepts/selection.md). |
| `model.max_neuron_models` | int | null | The most candidates a family tries per layer. Above it, a random subset is drawn from the seeded generator. Null tries every pair or tuple. |
| `model.shortcut` | bool or mapping | `{raw_features: true, prev_layers: null}` | What feeds every layer after the first, besides the previous layer. `true` and `false` are short for `raw_features: true` or `false` with `prev_layers: null`. |
| `model.shortcut.raw_features` | bool | true | Feed the model's input features to every layer. |
| `model.shortcut.prev_layers` | int, `all` or null | null | Also feed the outputs of this many layers before the previous one; null feeds none, `all` every earlier layer. See [The algorithm](../concepts/algorithm.md#1-layer-inputs). |

## Orthogonal-family squash

Defaults for the `legendre` and `chebyshev` families; an entry can override
each one. See [Orthogonal polynomials](../concepts/neurons/orthogonal.md#the-squash).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `model.squash_method` | str | `sigma` | `sigma`: linear within a band around the input's mean, saturating beyond. `tanh`: the hyperbolic tangent, no statistics needed. |
| `model.squash_n_sigma` | float | 2.0 | Half-width of the linear band, in standard deviations. `sigma` only. |
| `model.squash_core_range` | float | 0.75 | Squashed value at the edge of the band; between 0 and 1. `sigma` only. |

## Multi-class heads

How a multi-class neuron's single output becomes class scores. See
[Classification](../concepts/classification.md).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `model.soft_binner` | bool | true | Score each class by how close the neuron's output lies to a fixed point for that class, between 0.05 and 0.95 in label order. Cannot be combined with `model.use_neuron_proj`. |
| `model.soft_binner_scale` | float | 100.0 | Sharpness of the soft binner: a class's score is minus this value times the squared distance to its point. |
| `model.use_neuron_proj` | bool | false | Give every candidate its own linear map from its output to the class scores. Needs `model.soft_binner: false`. With both off, the candidates of a layer share one learned map. |
| `train.shared_proj_lr_multiplier` | float | 0.1 | Learning-rate factor for that shared map, relative to the candidates' learning rate. |

## Criterion

See [Criteria](../concepts/criteria.md).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `train.criterion_type` | str | `validate` | `validate`: regularity error on dev. `bias`: disagreement between fits on the even and odd training rows; the survivors are fitted on the whole train split, as under the others. `validate_bias`: a mix of the two. Any other value raises `ValueError` when the model is built. See [Criteria](../concepts/criteria.md). |
| `train.error_alpha` | float | 0.5 | Weight of the regularity error in `validate_bias`; the bias error gets 1 minus this. |
| `train.bias_ce_type` | str | `js` | Multi-class bias error: `js` (Jensen-Shannon divergence) or `l2` (squared difference of the class scores). |
| `train.error_normalization` | str | `variance` | Denominator of the regression criteria and training loss: `variance` (spread around the mean) or `energy` (sum of squared targets). |
| `train.layer_err_criterion` | str | `top` | A layer's error: the best survivor's criterion value (`top`) or the survivors' mean (`avg`). |
| `train.layer_err_source` | str | `neuron` | `neuron`: the survivors' own criterion values. `readout`: the dev loss of a linear head over all survivors; needs `model.use_output_projection: true`; regression models only. |

## Growth rule

See [Splits and stopping](../concepts/splits-and-stopping.md#the-growth-rule).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `train.max_layer_count` | int | 999 | The most layers the search builds. |
| `train.criterion_minimum_width` | int | 5 | Stop after this many layers in a row without an improvement. |
| `train.stop_train_epsilon_condition` | float | 0.001 | Relative part of the improvement margin: this fraction of the best layer error so far. |
| `train.stop_train_min_delta` | float | 0.0 | Absolute part of the improvement margin, in units of the layer error. The margin is the larger of the two. Set it near the noise level of the dev error. |
| `train.stop_source` | str | `dev` | The split the growth rule and the end-to-end pass's early stop read: `dev`, or `val` for the validation loader passed as `val_dl` (required then). |

## Survivor selection

See [Survivor selection](../concepts/selection.md).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `train.neuron_selection_method` | str | `plain` | `plain`: the lowest criterion values. `omp_mixed`: in order of criterion value, skipping candidates whose dev outputs the survivors already explain. `omp`: orthogonal matching pursuit on the dev outputs, ignoring the criterion. |
| `train.neuron_selection_orth_threshold` | float | 0.3 | `omp_mixed` keeps a candidate when at least this fraction of its output's norm remains after removing the survivors' directions. |
| `train.log_layer_diagnostics` | bool | false | Log the survivors' correlations, top singular values and effective rank on dev after each layer; one extra pass over dev. |

## Candidate fit

How each candidate's coefficients are fitted on the train split. See
[The algorithm](../concepts/algorithm.md#4-fit) and
[Optimizers](../guides/optimizers.md).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `train.optimizer.name` | str | `adam` | The batched optimizer: `adam`, `sgd` or `lbfgs`. Any other name raises `ValueError` when the `Trainer` is built. |
| `train.optimizer.optimizer_params` | mapping | `{lr: 1e-4, min_lr: 1e-5, gamma: 0.5, clip_value: 1.0, clip_norm: 5.0}` | Learning-rate schedule of the early stop and the optimizer's own arguments; see [Optimizer parameters](#optimizer-parameters). |
| `train.optimizer.verbose` | bool | true | Show a progress bar for each family's fit. |
| `train.steps` | int | 1000 | The most optimizer steps per family fit; a fit that does not stop early runs exactly this many. |
| `train.ridge_alpha` | float | 0.0 | L2 penalty on each candidate's neuron coefficients, added to the training loss only. |
| `train.censor_target_at` | float | null | Regression targets at or above this value count as "at least this much": the training loss clips the prediction to it on those rows. The criteria are unchanged. |
| `train.divergence_threshold` | float | infinity | A candidate whose evaluated loss exceeds this stops at once. Infinity turns the check off; a NaN loss always stops the candidate. |

## Candidate fit: evaluation and early stop

See [Splits and stopping](../concepts/splits-and-stopping.md#stopping-a-candidates-fit).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `train.eval_step_interval` | int | 1000 | Evaluate every candidate every this many steps; -1 evaluates at the end of each pass over the loader. The early stop acts only at evaluations. |
| `train.eval_smoothing_factor` | float | 0.2 | Weight of the newest evaluation in the exponential moving average of the loss. |
| `train.early_stop_source` | str | `dev` | The loss the early stop reads: the dev split (`dev`) or the current training batch (`train`). |
| `train.early_stop_patience` | float | 1e-4 | The smallest drop in the smoothed loss that counts as an improvement, in absolute units. |
| `train.early_stop_tolerance_steps` | int | 10 | Evaluations without an improvement before the learning rate drops; once it is at `min_lr`, before the candidate stops. |
| `train.early_stop_completion_percentage` | int | 100 | End the family's fit once more than this percentage of its candidates have stopped; at 100, only when all have. |
| `train.eval_display_interval` | int | 1 | Refresh the progress bar's numbers every this many evaluations. Does not affect training. |

## Input pass and RBF placement

See [Gaussian RBF](../concepts/neurons/rbf.md#where-the-bumps-start).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `train.input_sample_rows` | int | 65536 | The most rows sampled from the train split for RBF placement. |
| `train.rbf_kmeans_mode` | str | `auto` | `sample`: k-means on the sample. `stream`: the start on the sample, then mini-batch k-means over the whole split. `auto`: `stream` when the split has more rows than the sample holds, else `sample`. |
| `train.rbf_kmeans_iters` | int | 20 | Lloyd iterations of k-means on the sample. |
| `train.rbf_kmeans_passes` | int | 1 | Passes of streaming k-means over the split. |

## Head fit and per-layer fine-tune

`train.out_proj_train` configures `Trainer.train_out_proj` (the head fit)
and the per-layer fine-tune. See
[Heads and fine-tuning](../concepts/heads-and-finetune.md).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `train.layer_finetune` | bool | false | After selection, train each layer's survivors jointly through a temporary linear head, on every layer. Use it with `model.use_output_projection`: the pass turns the survivors into inputs for a head, and a model without one predicts poorly. Fails on binary models. |
| `train.out_proj_train.optimizer` | str | `adam` | `adam`, `sgd` (momentum 0.9) or `lbfgs` (full batch, no plateau schedule). |
| `train.out_proj_train.lr` | float | 0.001 | Learning rate. |
| `train.out_proj_train.weight_decay` | float | 0.0 | L2 penalty; added to the loss under `lbfgs`. |
| `train.out_proj_train.max_steps` | int | 5000 | The most steps; for `lbfgs`, outer steps. |
| `train.out_proj_train.eval_interval` | int | 100 | Evaluate on the dev split every this many steps. |
| `train.out_proj_train.early_stop_patience` | int | 10 | Stop after this many evaluations without an improvement. |
| `train.out_proj_train.early_stop_min_delta` | float | 1e-4 | The smallest drop in dev loss that counts as an improvement. |
| `train.out_proj_train.keep_best_weights` | bool | false | End the head fit and the per-layer fine-tune on the weights of their lowest dev loss instead of their last step. See [Last step or best evaluation](../concepts/heads-and-finetune.md#last-step-or-best-evaluation). |
| `train.out_proj_train.best_weights_copy` | str | `device` | Where `keep_best_weights` keeps the copy of the best weights: `device` (next to the parameters, in GPU memory on CUDA), `cpu` (host memory) or `disk` (a file in the run folder, deleted when the pass ends). |
| `train.out_proj_train.lr_patience` | int | 5 | Evaluations without improvement before the plateau schedule cuts the learning rate. Not used by `lbfgs`. |
| `train.out_proj_train.lr_factor` | float | 0.5 | Factor of each cut. Not used by `lbfgs`. |
| `train.out_proj_train.lr_min` | float | 1e-5 | Lowest learning rate of the plateau schedule. Not used by `lbfgs`. |
| `train.out_proj_train.lbfgs_history_size` | int | 20 | `lbfgs` only: correction pairs kept. |
| `train.out_proj_train.lbfgs_max_iter` | int | 20 | `lbfgs` only: inner iterations per step. |
| `train.out_proj_train.lbfgs_line_search` | str | `strong_wolfe` | `lbfgs` only: the line search; empty for none. |
| `train.out_proj_train.cuda_graph` | bool | true | No effect here; used by the end-to-end pass. |
| `train.out_proj_train.data_on_device` | str | `auto` | No effect here; used by the end-to-end pass. |

## End-to-end pass

`train.finetune_train` configures `Trainer.train_finetune`, which trains
every parameter at once. It has the same keys as `train.out_proj_train`;
the optimizers differ and two keys only matter here.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `train.finetune_train.optimizer` | str | `adam` | `adam`, `adamw` or `sgd` (momentum 0.9). |
| `train.finetune_train.lr` | float | 0.001 | Learning rate. A whole-network pass usually wants a much smaller one than the head fit. |
| `train.finetune_train.weight_decay` | float | 0.0 | Weight decay of the optimizer. |
| `train.finetune_train.max_steps` | int | 5000 | The most steps. |
| `train.finetune_train.eval_interval` | int | 100 | Evaluate every this many steps, on dev (or the validation split under `train.stop_source: val`). |
| `train.finetune_train.early_stop_patience` | int | 10 | Stop after this many evaluations without an improvement. |
| `train.finetune_train.early_stop_min_delta` | float | 1e-4 | The smallest drop in loss that counts as an improvement. |
| `train.finetune_train.keep_best_weights` | bool | false | End the pass on the parameters of its lowest evaluated loss instead of its last step. See [Last step or best evaluation](../concepts/heads-and-finetune.md#last-step-or-best-evaluation). |
| `train.finetune_train.best_weights_copy` | str | `device` | Where `keep_best_weights` keeps the copy of the whole model's parameters: `device`, `cpu` or `disk`, as for the head fit. |
| `train.finetune_train.lr_patience` | int | 5 | Evaluations without improvement before the plateau schedule cuts the learning rate. |
| `train.finetune_train.lr_factor` | float | 0.5 | Factor of each cut. |
| `train.finetune_train.lr_min` | float | 1e-5 | Lowest learning rate of the plateau schedule. |
| `train.finetune_train.lbfgs_history_size` | int | 20 | No effect: the end-to-end pass does not use LBFGS. |
| `train.finetune_train.lbfgs_max_iter` | int | 20 | No effect, as above. |
| `train.finetune_train.lbfgs_line_search` | str | `strong_wolfe` | No effect, as above. |
| `train.finetune_train.cuda_graph` | bool | true | On a CUDA device with `adam` or `adamw`, capture the training step once as a CUDA graph and replay it; much faster for these small models. |
| `train.finetune_train.data_on_device` | str | `auto` | With the CUDA graph: keep the training split on the device (`true`), stream it from the loader (`false`), or keep it when it fits in a quarter of the free memory (`auto`). |

## Checkpoints

| Key | Type | Default | Meaning |
|---|---|---|---|
| `train.checkpoint_dir` | str | `checkpoints` | Parent of the run folders: every `Trainer.train` run writes its step checkpoints, `model_last.ckpt` and its log to `<checkpoint_dir>/<YYYY-MM-DD-HH-MM-SS>/` (see [Training](../guides/training.md#checkpoints)). A relative path, the default included, is taken from the working directory. Empty raises `ValueError` when the `Trainer` is built. |
| `train.save_interval` | int | 1000 | Save a step checkpoint every this many steps of a family fit; -1 turns this off. |
| `train.skip_saving_at_epoch_end` | bool | true | Do not also save at the end of each pass over the training loader. |
| `train.save_last_layer` | bool | true | At the end of each family fit, also save a copy marked `_last`, which the step-checkpoint cleanup keeps. |
| `train.keep_last_n` | int | 10 | Keep this many of a run folder's most recent step checkpoints; nothing else in the folder is counted or deleted. |

## Runtime

| Key | Type | Default | Meaning |
|---|---|---|---|
| `train.device` | str | `cpu` | Where neurons are created and computed: `cpu`, `cuda`, `cuda:1` and so on. |
| `train.seed` | int | 10 | Seed of the input pass's sampling and k-means, and of the output head's starting weights. Pass it to `Trainer.set_seed`, before building the model, to seed the rest. |
| `train.use_deterministic_algorithms` | bool | false | When the `Trainer` is built, switch PyTorch to deterministic kernels for the whole process (and set `CUBLAS_WORKSPACE_CONFIG` if it is unset), so runs on a GPU repeat; slower, and an operation without a deterministic kernel raises. False leaves PyTorch's setting alone. See [Training](../guides/training.md#seeds). |
| `train.precompute_features` | bool | false | Run the frozen layers over each split once per layer and cache the result, instead of on every step. Faster; costs memory for the cached features. |
| `train.verbose` | bool | true | Show progress bars for the head fit, the per-layer fine-tune and the end-to-end pass. |

## Read by the tutorial scripts

The library does not read these keys; the tutorial scripts do. They are in
the schema so that tutorial YAML files can set them.

| Key | Type | Default | Meaning in the tutorials |
|---|---|---|---|
| `train.batch_size` | int | 1 | Batch size of the data loaders the scripts build. |
| `train.shuffle` | bool | false | Whether the scripts shuffle the training loader. |
| `resume` | bool | false | Passed to `Trainer.train(resume=...)`: continue the newest run folder that holds a step checkpoint. |
| `finetune_end_to_end` | bool | false | After the search and the head fit, run `Trainer.train_finetune`. |
| `finetune_drop_head` | bool | false | With `finetune_end_to_end`, remove the head first. |
| `finetune_prune_first` | bool | false | With `finetune_end_to_end`, prune the network first. |
| `tutorial` | mapping | empty | Free-form settings of one tutorial script, such as its data split and feature engineering; each tutorial page lists its own. |

## Neuron family options

The options of a `model.ref_functions` entry, as in
`- legendre: {degree: 3, dim: 4}`.

**Every family**

| Option | Default | Meaning |
|---|---|---|
| `activation` | identity | Output activation: `identity`, `relu`, `leaky_relu`, `elu`, `selu`, `celu`, `gelu`, `silu`, `mish`, `tanh`, `tanhshrink`, `softplus`, `softsign`, `sigmoid`, `log_sigmoid`, `hardtanh`, `hardswish` or `hardsigmoid`. An empty string means the identity. |
| `init_method` | `xavier` | Initial coefficients: `xavier` (Xavier uniform) or `uniform` (uniform in ±0.1). |
| `max_neuron_models` | `model.max_neuron_models` | This family's own candidate cap. |

**`polyquad`**

| Option | Default | Meaning |
|---|---|---|
| `dim` | 2 | Inputs per neuron. Above 2, `model.max_neuron_models` must be set. |
| `squares` | true | Include the squared terms. |

**`legendre` and `chebyshev`**

| Option | Default | Meaning |
|---|---|---|
| `degree` | 3 | Highest polynomial degree per input. |
| `dim` | 2 | Inputs per neuron. |
| `cross` | true | Include the pairwise products of the squashed inputs. |
| `squash` | true | Squash the inputs into $(-1, 1)$. |
| `squash_method` | `model.squash_method` | `sigma` or `tanh`. |
| `squash_n_sigma` | `model.squash_n_sigma` | Half-width of the linear band, in standard deviations. |
| `squash_core_range` | `model.squash_core_range` | Squashed value at the band's edge. |

**`rbf`**

| Option | Default | Meaning |
|---|---|---|
| `centers` | 16 | Gaussian bumps per neuron; at least 2. |
| `dim` | 2 | Inputs per neuron. |
| `placement` | `kmeans` | `kmeans` or `grid` (then `centers` must be a whole power of `dim`). |
| `seeding` | `pca_quantiles` | Start of k-means: `pca_quantiles` or `kmeans++`. |
| `width` | 1.0 | Starting width, times the bump's local spacing. |
| `learn_centers` | true | Learn the centres. |
| `learn_widths` | true | Learn the widths. |
| `center_radius` | 2.0 | How far a centre may move, times its local spacing; null for no bound. |
| `width_band` | 4.0 | Widths stay within this factor of their start; above 1. |
| `normalize` | true | Normalize the bumps to sum to 1. |
| `linear` | true | Add the standardized inputs to the design row. |
| `standardize` | true | Standardize the inputs first. |

The pair families `linear`, `linear_cov`, `quadratic` and `cubic` take only
the options of every family.

## Optimizer parameters

The keys of `train.optimizer.optimizer_params`.

**Read by the trainer**

| Key | Default | Meaning |
|---|---|---|
| `lr` | 1e-4 | Every candidate's starting learning rate. |
| `min_lr` | 1e-5 | The early stop lowers a candidate's learning rate to this before stopping it. |
| `gamma` | 0.5 | Factor of each learning-rate drop. |

**Passed to the optimizer**

| Key | Optimizers | Default | Meaning |
|---|---|---|---|
| `clip_value` | `adam`, `sgd`, `lbfgs` | 1.0 | Clamp every gradient entry to ±this value. |
| `clip_norm` | `adam`, `sgd`, `lbfgs` | 5.0 | Then scale each candidate's gradient down to at most this norm. |
| `betas` | `adam` | (0.9, 0.999) | Adam's moment decay rates. |
| `eps` | `adam` | 1e-8 | Adam's denominator term. |
| `momentum` | `sgd` | 0.9 | Momentum. |
| `nesterov` | `sgd` | true | Nesterov momentum. |
| `weight_decay` | `sgd` | 0.0 | Weight decay. |
| `history_size` | `lbfgs` | 10 | Correction pairs kept per candidate. |
| `max_step` | `lbfgs` | 1.0 | The largest update norm per candidate and parameter; null for no cap. |
| `curvature_eps` | `lbfgs` | 1e-8 | Keep a correction pair only when its curvature passes this test; null keeps every pair. |

The trainer sets two more arguments itself: `lr`, from the key above, and
`shared_param_lr_multiplier`, from `train.shared_proj_lr_multiplier`. Do not
put `shared_param_lr_multiplier` in `optimizer_params`.

<small>Checked against TorchSONN 0.1.5.</small>
