"""Structured-config dataclass schemas for SONN.

These mirror the keys that previously lived in `conf/default.yaml`, with two
upgrades over a plain YAML default:

  • Type-checked field values — `omegaconf` rejects `model.type: 42` at
    compose time instead of yielding cryptic AttributeErrors later.
  • Schema discoverable via Python introspection — `dataclasses.fields(...)`,
    IDE autocomplete, Sphinx autodoc all work.

Polymorphic / dynamically-typed fields are deliberately left as `List[Any]`
or `Optional[Dict[str, Any]]`:

  • `ModelConfig.ref_functions` — heterogeneous list (bare names, mappings
    carrying options); see `_parse_ref_function_entry` in model.py for the
    parser. Schema-typing it stricter would force every entry into the
    dict form, which would uglify the common `- linear_cov` case.
  • `ModelConfig.shortcut` — a bool shorthand or a mapping; parsed by
    `_parse_shortcut` in model.py.
  • `SchedulerConfig.scheduler_params` — varies per scheduler family.
  • `TrainConfig.criterion_type` — string here (e.g. `"validate"`); coerced
    to `CriterionType` enum inside `SONN.__init__` via `CriterionType.get`.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore


# ---------------------------------------------------------------------------
# Optimizer / scheduler — referenced from TrainConfig
# ---------------------------------------------------------------------------
@dataclass
class OutProjTrainConfig:
    """Optimizer settings for a head or fine-tune pass.

    Used twice in `TrainConfig`: `train.out_proj_train` (the output head fit
    and the per-layer fine-tune) and `train.finetune_train` (the end-to-end
    pass). The comments on the fields below document them.
    """
    max_steps: int = 5000
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    # 'adam' | 'sgd' | 'lbfgs'. lbfgs runs full-batch with a closure and ignores
    # the ReduceLROnPlateau schedule (lr_patience / lr_factor / lr_min) — it does
    # its own line search. weight_decay is honored for lbfgs as a manual L2
    # penalty added inside the closure.
    optimizer: str = "adam"
    eval_interval: int = 100
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_min: float = 1.0e-5
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1.0e-4

    # LBFGS-only tuning. history_size matches PyTorch's default; max_iter is the
    # number of inner LBFGS iterations per opt.step() call. For Otto-sized
    # problems max_steps=100 outer x max_iter=20 inner ≈ sklearn's default.
    lbfgs_history_size: int = 20
    lbfgs_max_iter: int = 20
    lbfgs_line_search: str = "strong_wolfe"  # "" / None to disable

    # End-to-end pass only (Trainer.train_finetune). The step (forward,
    # backward, adam / adamw update) is launch-bound on these models: several
    # hundred tiny kernels, 15-18 ms of launch latency for ~1 ms of work. With
    # cuda_graph on and a CUDA device the step is captured once over a
    # static batch buffer of train_dl's batch size and replayed per step
    # (about 3 ms on the California Legendre model); each step copies the
    # next batch into the buffer, so the dataset size does not matter. A
    # batch of another shape (the tail of an epoch) runs the same step
    # eagerly; CPU, sgd and a capture failure fall back to the eager loop.
    cuda_graph: bool = True
    # Where the training batches come from under cuda_graph:
    #   'auto'  - the whole split is kept on the device when it fits in a
    #             quarter of the free memory, else streamed from the loader
    #   'true'  - always resident (each step index-selects the next batch of
    #             a per-epoch permutation; shuffling as the loader would)
    #   'false' - always streamed (pinned host copy per step, overlapping the
    #             previous replay; the way for splits larger than the GPU).
    #             Build the loader with pin_memory=True (and workers) for
    #             this: pinning a pageable batch inside the step costs ~35 ms
    #             per 12k rows, more than the replay.
    data_on_device: str = "auto"


def _default_optimizer_params() -> Dict[str, Any]:
    """Shared optimizer/trainer kwargs.

    The trainer pops `min_lr` and `gamma` (LR-drop schedule) before handing
    the rest to the chosen optimizer class. The optimizer-specific extras
    (`betas`/`eps` for adam, `momentum`/`nesterov` for sgd, `history_size`,
    `max_step` and `curvature_eps` for lbfgs, `damping`/`max_damping` for
    newton-LM) are NOT enumerated here — see the note on
    `OptimizerConfig.optimizer_params` below. lbfgs defaults: `max_step`
    1.0 (cap on one member's update norm per parameter tensor, in parameter
    units; null = uncapped) and `curvature_eps` 1e-8 (a correction pair is
    kept only if y.s > eps |s||y|; null = keep every pair, the historical
    behaviour). The candidate-fit log reports how often either guard acted.
    """
    return {
        "lr": 1.0e-4,
        "min_lr": 1.0e-5,
        "gamma": 0.5,
        "clip_value": 1.0,
        "clip_norm": 5.0,
    }


@dataclass
class OptimizerConfig:
    """The `train.optimizer:` section: the optimizer for candidate neuron fits.

    The comments on the fields below document them.
    """
    # 'adam' | 'sgd' | 'lbfgs' | 'newton' | 'newton-lm' — see optimizer_map
    # in src/optimizers/__init__.py.
    name: str = "adam"
    verbose: bool = True
    # Deliberately typed `Dict[str, Any]` rather than a nested dataclass:
    # each optimizer family takes its own kwargs (lbfgs needs
    # `history_size`, adam takes `betas`/`eps`, sgd takes `momentum`/
    # `nesterov`, newton-lm takes `damping`/`max_damping`). A strict union
    # schema would force every YAML to set every field. The optimizer
    # constructor itself rejects unknown kwargs at instantiation time, so
    # validation happens at the right boundary.
    #
    # If you want strict per-family typing later, switch to Hydra config
    # groups: cs.store(group="optimizer", name="adam", node=AdamParams) and
    # reference via `defaults: [{optimizer: adam}, default, _self_]`.
    optimizer_params: Dict[str, Any] = field(default_factory=_default_optimizer_params)


@dataclass
class SchedulerConfig:
    """The `train.scheduler:` section: the learning-rate scheduler for candidate fits.

    The comments on the fields below document them.
    """
    # 'warmup_flat' (currently the only registered scheduler) | null to
    # disable the scheduler entirely.
    name: Optional[str] = None
    # Schema is loose here — each scheduler family takes its own kwargs.
    scheduler_params: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """The `model:` section: task type, neuron families and network structure.

    The comments on the fields below document them.
    """
    # 'regressor' | 'binary' | 'multi-class'
    type: str = "multi-class"
    soft_binner: bool = True
    soft_binner_scale: float = 100.0
    num_classes: int = 3

    # Heterogeneous polymorphic list — see _parse_ref_function_entry.
    ref_functions: List[Any] = field(default_factory=lambda: ["linear_cov"])

    # Which tensors feed every layer after the first, besides the outputs of
    # the layer right before it (always fed). Polymorphic like ref_functions,
    # see `_parse_shortcut` in model.py:
    #   raw_features: bool        — re-feed the (preprocessed) model inputs.
    #   prev_layers: int | 'all' | null — also feed the outputs of that many
    #       layers *before* the last one; null (= 0) feeds the last layer only,
    #       'all' every earlier layer.
    # The bool shorthand `shortcut: true|false` means
    # `{raw_features: true|false, prev_layers: null}`.
    shortcut: Any = field(default_factory=lambda: {"raw_features": True, "prev_layers": None})

    # How the orthogonal-polynomial families (legendre / chebyshev) map their
    # inputs into [-1, 1], the only interval where those bases are orthogonal
    # and bounded. Ignored by every other ref function, and by an entry that
    # sets `squash: false`. A ref_functions entry may override it per family,
    # e.g. `- legendre: {squash_method: tanh}`.
    #
    #   'sigma' (default) — SigmaSquashNorm (modules/sigma_squash_norm.py):
    #       standardize by the per-feature mean/std, pass everything within
    #       +/-squash_n_sigma through linearly onto +/-squash_core_range, and
    #       saturate only beyond that with a C2 rational tail. The statistics
    #       are measured over the whole training set once per layer, before
    #       that layer trains (Trainer.fit_layer_inputs), which costs one extra
    #       forward pass over the training split per layer.
    #   'tanh' — the historical stateless squash. No calibration pass, but it
    #       compresses the bulk of the distribution: tanh is at 0.76 by 1 sigma.
    squash_method: str = "sigma"
    # Half-width of the linear core, in standard deviations, and the output
    # magnitude reached there. 'sigma' only. core_range must be in (0, 1).
    squash_n_sigma: float = 2.0
    squash_core_range: float = 0.75

    # Required by the caller (tutorial YAMLs override). Left as Optional so
    # the schema can still load with the library default before the caller
    # supplies a value; SONN.__init__ asserts `nbest_neurons > 1`.
    nbest_neurons: Optional[int] = None
    max_neuron_models: Optional[int] = None

    use_output_projection: bool = False
    num_out_neurons: Optional[int] = None

    # Per-neuron linear projection: each neuron in the ensemble gets its own
    # nn.Linear(1, num_classes) trained with in_dims=0 (vs shared_proj which
    # uses in_dims=None). Mutually exclusive with soft_binner.
    use_neuron_proj: bool = False

    # torch.clamp threshold applied between layers. Raise above the worst-
    # case |xi*xj| at xavier init when feeding heavy-tailed unclipped
    # features (e.g. California-housing Population).
    output_clamp_value: float = 1000.0

    # Apply nn.LayerNorm(d_layer, elementwise_affine=False) to each layer's
    # post-clamp output, before the optional shortcut concat. Standardizes
    # the (batch, nbest_neurons) feature map so the next layer's polynomial
    # neurons see comparable scales regardless of how heavy-tailed the
    # previous layer's polynomial happens to be. No trainable params — pure
    # per-sample standardization, identical to the preprocessing LayerNorm
    # that the Otto tutorial puts on the model's raw input.
    use_layer_norm: bool = False


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """The `train:` section: fitting, selection, stopping and checkpointing.

    The comments on the fields below document them.
    """
    seed: int = 10

    # String form — coerced to CriterionType via CriterionType.get() in
    # SONN.__init__. Accepted: 'validate' | 'bias' | 'validate_bias' | 'bias_retrain'.
    criterion_type: str = "validate"
    # Bias criterion variant for multi-class: 'l2' (L2 on logits) | 'js' (Jensen-Shannon divergence).
    bias_ce_type: str = "js"
    error_alpha: float = 0.5

    # Denominator for the regression criteria — regularity_error, bias_error and
    # the NormMSE training loss. Also applies to `binary`, which shares those
    # criteria; there the centered denominator is the Bernoulli variance
    # N·p(1-p). Multi-class is unaffected: regularity_error_ce normalizes by
    # H(Y_B) and the bias criteria by label energy, both already proper no-skill
    # baselines. Validated in SONN.__init__.
    #
    #   'variance' (default) — Σ(y - ȳ)². A true fraction of variance
    #       unexplained (1 - R²): invariant to a constant offset on y, ≈1 at the
    #       mean predictor, and in [0, ~1] for any dataset — so the absolute
    #       thresholds below (early_stop_patience, divergence_threshold) mean the
    #       same thing regardless of target scale.
    #   'energy'  — Σ y², Ivakhnenko's original and what gmdhpy computes. Its
    #       baseline is the zero predictor, so it is only meaningful on targets
    #       that are already centered; on a target with mean 454 and sd 17 the
    #       denominator is ~700x the variance and the reported error collapses
    #       into a razor-thin band near zero. Set this only for exact parity with
    #       classical implementations.
    #
    # Neuron ranking is identical either way — the denominator is constant
    # within one evaluation — so switching affects reported values and the
    # absolute thresholds, not which neurons survive.
    error_normalization: str = "variance"

    max_layer_count: int = 999
    # Layer-growth stop rule (Trainer.train, GrowthCriterion). A layer counts
    # as an improvement when it lowers the error of the last accepted
    # improvement by at least
    #     max(stop_train_min_delta, stop_train_epsilon_condition * best)
    # (so several small steps down add up), and the search stops once
    # `criterion_minimum_width` consecutive layers have passed without one. The layers kept are those up to the best
    # error, whether or not that layer cleared the margin.
    #   stop_train_epsilon_condition: relative margin (gmdhpy's rule). At an
    #       error of 0.17 the default 1e-3 is 0.00017, below what a dev
    #       evaluation of a few thousand rows resolves, so rounding-level
    #       differences between runs decide the depth (California: 9 vs 11
    #       layers, 0.005 on test, from float32 summation order alone).
    #   stop_train_min_delta: absolute margin in the units of the layer error
    #       (NormMSE for regression). 0 = off. Set it at the noise floor of
    #       the dev evaluation (0.002 on the California tutorial) so "still
    #       improving" means something the data can resolve.
    # Before 2026-09-30 a new best that fell short of the relative margin
    # stopped the search on the spot; now it counts toward the window like
    # any other non-improving layer, so one flat layer does not end a run
    # that is still descending slowly.
    criterion_minimum_width: int = 5
    stop_train_epsilon_condition: float = 0.001
    stop_train_min_delta: float = 0.0

    manual_best_neurons_selection: bool = False
    min_best_neurons_count: int = 0
    max_best_neurons_count: int = 0

    # 'top' (smallest err_value wins) | 'avg'.
    layer_err_criterion: str = "top"

    # What `layer.err` - the number Trainer.train's layer-growth criterion
    # compares across layers - measures:
    #   'neuron'  - the survivors' individual dev errors, reduced per
    #               `layer_err_criterion` (historical default). Right for a
    #               model that is read out through its single best neuron.
    #   'readout' - the dev loss of a head fitted over *all* survivors, i.e.
    #               what a model with an output head is actually scored on.
    #               With `layer_finetune: true` that is the fine-tune's own
    #               head (with 'neuron' the fine-tune looks like a regression:
    #               it turns survivors into a basis, so each one alone gets
    #               worse); with `layer_finetune: false` a temporary head is
    #               fitted over the frozen survivors just to measure the
    #               readout, and the neurons themselves are untouched. Uses
    #               the `out_proj_train` hyperparameters. Requires
    #               `model.use_output_projection: true` (without a head,
    #               inference reads the best neuron and the readout error would
    #               describe a head that is never used); Trainer raises
    #               otherwise. Regressor / binary only.
    layer_err_source: str = "neuron"

    # Algorithm used by train_layer → neuron_selection to pick survivors from
    # the candidate pool:
    #   'plain'     — top-k by individual error (historical default).
    #   'omp_mixed' — walk candidates in error order, drop ones whose residual
    #                 (after Gram-Schmidt vs. already-selected survivors on the
    #                 dev set) falls below `neuron_selection_orth_threshold` of
    #                 its original norm. Trades a small NCE bump for
    #                 decorrelated outputs feeding the next layer.
    #   'omp'       — classical Orthogonal Matching Pursuit: at every step pick
    #                 the candidate whose orthogonalized residual has the
    #                 largest norm. Ignores `err` entirely; purely decorrelation-
    #                 driven, vectorized over the candidate axis.
    neuron_selection_method: str = "plain"
    # Used only by the 'omp_mixed' variant. 0.3 = "keep candidate if at least
    # 30 % of its centered norm survives orthogonalization."
    neuron_selection_orth_threshold: float = 0.3

    eval_step_interval: int = 1000
    # How often (in eval passes) to refresh the tqdm progress bar's scalar
    # readout. The bar's mean/min/max/lr/completed values each need a GPU->CPU
    # sync (.item()); they are cosmetic, so read them back only every
    # eval_display_interval evals rather than every eval. Does NOT affect
    # early-stop timing or results — the stop decision is computed on-device and
    # checked every eval regardless. 1 = refresh every eval (default cadence).
    eval_display_interval: int = 1
    eval_smoothing_factor: float = 0.2

    save_interval: int = 1000
    save_last_layer: bool = True
    keep_last_n: int = 10
    skip_saving_at_epoch_end: bool = True
    checkpoint_dir: str = ""

    early_stop_completion_percentage: int = 100
    early_stop_patience: float = 1.0e-4
    early_stop_tolerance_steps: int = 10
    # Which loss the per-candidate early stop (and its learning-rate drop)
    # watches during the candidate fit:
    #   'dev'   - the dev split, evaluated every eval_step_interval steps
    #             (historical). Every candidate is then tuned to the split
    #             that selects it, 120 times per layer; the minimum over
    #             thousands of dev-stopped candidates is an optimistic dev
    #             floor (California, RBF family: 0.145 on dev, worse on test).
    #   'train' - the training loss of the current batch, same smoothing,
    #             patience and lr drop. Candidates run to their own
    #             convergence; the dev split is used once per layer, by
    #             selection, which is the method's regularizer (classic
    #             GMDH: least squares on train, ranking on dev). A candidate
    #             that overfits its rows is rejected by selection rather than
    #             stopped early.
    early_stop_source: str = "dev"
    # Which split the two coarse stop decisions read when the caller hands
    # `Trainer.train` / `train_finetune` a validation loader (`val_dl`, a split
    # that selects nothing):
    #   'dev' - the growth criterion and the end-to-end early stop use the
    #           dev split, as before; `val_dl` is only reported (per layer
    #           the best neuron's error on it next to the dev error, and the
    #           finished model's loss).
    #   'val' - they use the validation split, leaving dev to selection (and
    #           to the candidate early stop when early_stop_source is 'dev').
    #           Requires `val_dl`.
    stop_source: str = "dev"

    shared_proj_lr_multiplier: float = 0.1

    # When True, train_layer runs every frozen layer over the full dataset
    # once and caches the resulting features in CPU-backed DataLoaders, so
    # the inner training step skips the SONN forward. Speeds training up but
    # caches (N_samples, d_layer) floats per split, which can spike CPU RAM
    # at deeper layers. False = recompute features per step (the previous
    # behavior); trades wallclock for memory.
    precompute_features: bool = False

    # When True, log per-layer survivor-pool diagnostics (off-diagonal
    # correlation stats + top singular values + effective rank) on the dev
    # split after neuron_selection. Streaming O(D²) implementation — safe on
    # large dev sets — but still adds one extra full forward pass over dev
    # per layer, so leave off in production runs.
    log_layer_diagnostics: bool = False

    # Absolute val-loss threshold for permanently flagging a diverged
    # neuron. `.inf` disables the guard (matches gmdhpy).
    divergence_threshold: float = float("inf")

    # L2 (ridge) penalty on the per-neuron polynomial `weight` during the
    # OLS-style fit. The training loss for each candidate neuron becomes
    #     loss = data_fit + ridge_alpha * (params["weight"] ** 2).sum()
    # Only the polynomial coefficients are penalized — projection weights
    # (proj_weight / proj_bias / shared_proj_*) stay unregularized.
    # Recommended range: 0.01–0.1 for higher-arity primitives (cubic / poly
    # quad dim≥4) on small datasets where over-fit is the dominant gap.
    # Evaluation and prediction partials always see ridge_alpha=0 so the
    # reported dev / test metrics measure pure predictive quality, not the
    # penalized training objective. Default 0.0 = ridge disabled.
    ridge_alpha: float = 0.0

    # Right-censored regression target: rows whose target is at or above this
    # value are treated as "at least this much" by the training loss (NormMSE
    # clips the prediction to the cap for those rows, so predicting above it is
    # free). In the units the loss sees - log price if the target is
    # log-transformed. Only the fits use it (neuron fits, head, end-to-end
    # pass, and their dev early stopping); the neuron-selection criterion is
    # unchanged. Clip predictions to the cap at inference. None = off.
    censor_target_at: Optional[float] = None

    # Per-layer input pass (Trainer.fit_layer_inputs). Besides the streamed
    # mean / std the orthogonal families' squash needs, a family can ask for
    # a row sample of the layer input (RBF centers are placed by k-means on
    # it). `input_sample_rows` caps that sample: a seeded reservoir over the
    # training split, the whole split when it is smaller. It is a memory
    # guard, not a quality knob - a handful of centers are pinned to within a
    # few percent from a few thousand rows.
    input_sample_rows: int = 65536
    # How RBF centers are initialized when a family asks for a sample:
    #   'sample' - exact Lloyd k-means on the reservoir sample;
    #   'stream' - k-means++ start on the sample, then mini-batch k-means over
    #              the whole split, `rbf_kmeans_passes` passes, never holding
    #              more than one batch (Sculley's algorithm);
    #   'auto'   - 'sample' when the split has at most input_sample_rows
    #              rows, 'stream' above it.
    rbf_kmeans_mode: str = "auto"
    rbf_kmeans_iters: int = 20
    rbf_kmeans_passes: int = 1

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    out_proj_train: OutProjTrainConfig = field(default_factory=OutProjTrainConfig)

    # Hyperparameters for `Trainer.train_finetune` — the end-to-end pass that
    # unfreezes every parameter in the network at once, after the structural
    # search is finished. Separate from `out_proj_train` on purpose: that block
    # is already shared by the out_proj head fit and the per-layer
    # `layer_finetune` pass, and an end-to-end pass over a whole network wants
    # its own (usually much smaller) learning rate. `train_finetune` reads this
    # block by default; pass an explicit config to override.
    finetune_train: OutProjTrainConfig = field(default_factory=OutProjTrainConfig)

    # When True, train_layer runs an extra per-layer fine-tune pass after
    # neuron_selection: jointly trains the surviving neurons' polynomial
    # `weight`s together with a temporary (d_layer, num_classes) Linear head
    # against the class-weighted CE on the dev split. Refines the polynomial
    # coefficients so they're CE-aligned before the next layer trains on top
    # of them. Hyperparameters are shared with `out_proj_train` to keep the
    # config surface small. Skipped on the planned last layer
    # (layer_index == max_layer_count - 1) since the subsequent
    # train_out_proj pass effectively replaces it.
    layer_finetune: bool = False

    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 1
    steps: int = 1000
    shuffle: bool = False

    train_loss_tol: float = 0.001
    train_loss_window: int = 20

    verbose: bool = True


# ---------------------------------------------------------------------------
# Top-level SONN config
# ---------------------------------------------------------------------------
@dataclass
class SONNConfig:
    """Root config schema: the `model:` and `train:` sections plus top-level flags.

    Registered with Hydra as `default`; `SONN` merges the user config into it,
    so unknown keys are rejected. The comments on the fields below document
    them.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Tutorial-level workflow flags. Included in the base schema so the
    # `defaults: [default, _self_]` composition pattern in tutorial YAMLs
    # doesn't trip strict-mode "unknown key" errors. Each tutorial reads
    # only the flag(s) it cares about.
    resume: bool = False
    # After the structural search (and any out_proj head fit) completes, drop
    # the head and run `Trainer.train_finetune` over every parameter at once,
    # against the readout the model will actually be scored on. Leaves a
    # head-free network, so the pruned "discovered formula" is the whole model.
    # Tuned via `train.finetune_train`. (CCPP only.)
    finetune_end_to_end: bool = False
    # With `finetune_end_to_end`: drop model.out_proj first, so the pass
    # optimizes the bare network against its own best-error neuron and leaves a
    # head-free model. Off by default — keeping the head means the pass refines
    # the whole network against the readout it was actually fitted for, which
    # measures better; dropping it collapses the single-column readout and the
    # pass has to rebuild one. (CCPP only.)
    finetune_drop_head: bool = False
    # With `finetune_end_to_end`: prune to the read path first, so the pass
    # optimizes only neurons that reach the output. How much that removes
    # depends on the head — without one the network collapses to a single
    # chain; with `out_proj` the last layer keeps the `num_out_neurons` columns
    # the head consumes, so the saving is smaller. (CCPP only.)
    finetune_prune_first: bool = False
    # Free-form tutorial knobs (data split, feature engineering, target
    # transforms, ...). Untyped on purpose: each tutorial script reads and
    # validates its own keys, with its own defaults, so they need no schema
    # entry here. (CA: see tutorial_params in california_housing.py.)
    tutorial: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Register the schema with Hydra's ConfigStore.
#
# `name="default"` makes `- default` resolvable in any tutorial YAML's
# `defaults:` list, e.g.:
#
#     defaults:
#       - default        # SONNConfig
#       - _self_
#     model:
#       type: regressor
# ---------------------------------------------------------------------------
ConfigStore.instance().store(name="default", node=SONNConfig)