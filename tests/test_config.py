from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from torchsonn.config import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    SONNConfig,
    TrainConfig,
)


def test_default_sonnconfig_loads():
    cfg = OmegaConf.structured(SONNConfig)
    assert cfg.model.type == "multi-class"
    assert cfg.train.criterion_type == "validate"
    assert cfg.train.optimizer.name == "adam"
    assert cfg.train.scheduler.name is None
    assert isinstance(cfg.train.optimizer.optimizer_params, dict) or hasattr(
        cfg.train.optimizer.optimizer_params, "keys"
    )


def test_config_overrides_merge():
    base = OmegaConf.structured(SONNConfig)
    override = OmegaConf.create({"model": {"type": "regressor", "num_classes": 1}})
    merged = OmegaConf.merge(base, override)
    assert merged.model.type == "regressor"
    assert merged.model.num_classes == 1
    # other defaults preserved
    assert merged.train.criterion_type == "validate"


def test_optimizer_params_default_keys():
    cfg = OptimizerConfig()
    assert "lr" in cfg.optimizer_params
    assert "min_lr" in cfg.optimizer_params
    assert cfg.optimizer_params["lr"] == 1.0e-4


def test_scheduler_config_defaults():
    cfg = SchedulerConfig()
    assert cfg.name is None
    assert cfg.scheduler_params is None


def test_model_config_default_ref_functions():
    cfg = ModelConfig()
    assert cfg.ref_functions == ["linear_cov"]


def test_train_config_defaults_sane():
    cfg = TrainConfig()
    assert cfg.seed == 10
    assert cfg.batch_size == 1
    assert cfg.early_stop_tolerance_steps == 10
    assert cfg.error_normalization == "variance"


def test_config_store_has_default():
    cs = ConfigStore.instance()
    # Importing torchsonn.config registers this.
    nodes = cs.repo
    assert "default.yaml" in nodes or any(
        "default" in str(k) for k in nodes.keys()
    )
