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


def _default_optimizer_params() -> Dict[str, Any]:
    """Shared optimizer/trainer kwargs.

    The trainer pops `min_lr` and `gamma` (LR-drop schedule) before handing
    the rest to the chosen optimizer class. The optimizer-specific extras
    (`betas`/`eps` for adam, `momentum`/`nesterov` for sgd, `history_size`
    for lbfgs, `damping`/`max_damping` for newton-LM) are NOT enumerated
    here — see the note on `OptimizerConfig.optimizer_params` below.
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
    # 'regressor' | 'binary' | 'multi-class'
    type: str = "multi-class"
    soft_binner: bool = True
    soft_binner_scale: float = 100.0
    num_classes: int = 3

    # Heterogeneous polymorphic list — see _parse_ref_function_entry.
    ref_functions: List[Any] = field(default_factory=lambda: ["linear_cov"])

    shortcut: bool = True
    normalize: bool = True

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
    seed: int = 10

    # String form — coerced to CriterionType via CriterionType.get() in
    # SONN.__init__. Accepted: 'validate' | 'bias' | 'validate_bias' | 'bias_retrain'.
    criterion_type: str = "validate"
    # Bias criterion variant for multi-class: 'l2' (L2 on logits) | 'js' (Jensen-Shannon divergence).
    bias_ce_type: str = "js"
    error_alpha: float = 0.5

    max_layer_count: int = 999
    criterion_minimum_width: int = 5
    stop_train_epsilon_condition: float = 0.001

    manual_best_neurons_selection: bool = False
    min_best_neurons_count: int = 0
    max_best_neurons_count: int = 0

    # 'top' (smallest err_value wins) | 'avg'.
    layer_err_criterion: str = "top"

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
    eval_smoothing_factor: float = 0.2

    save_interval: int = 1000
    save_last_layer: bool = True
    keep_last_n: int = 10
    skip_saving_at_epoch_end: bool = True
    checkpoint_dir: str = ""

    early_stop_completion_percentage: int = 100
    early_stop_patience: float = 1.0e-4
    early_stop_tolerance_steps: int = 10

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

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    out_proj_train: OutProjTrainConfig = field(default_factory=OutProjTrainConfig)

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
    normalize: bool = True

    train_loss_tol: float = 0.001
    train_loss_window: int = 20

    verbose: bool = True


# ---------------------------------------------------------------------------
# Top-level SONN config
# ---------------------------------------------------------------------------
@dataclass
class SONNConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Tutorial-level workflow flags. Included in the base schema so the
    # `defaults: [default, _self_]` composition pattern in tutorial YAMLs
    # doesn't trip strict-mode "unknown key" errors. Each tutorial reads
    # only the flag(s) it cares about.
    resume: bool = False
    train_on_first_half: bool = False  # gmdhpy-style split toggle (CA only)


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