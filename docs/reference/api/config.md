# Configuration

The configuration schema is a set of dataclasses: `SONNConfig` holds the
`model` and `train` sections and a few top-level flags. Their fields are
documented by comments in the source and, on this site, in
[Configuration keys](../config.md); this page shows the classes.
[Configuration](../../guides/configuration.md) shows how to build a
configuration from them.

::: torchsonn.config.schemas.SONNConfig
    options:
      members: false

::: torchsonn.config.schemas.ModelConfig
    options:
      members: false

::: torchsonn.config.schemas.TrainConfig
    options:
      members: false

::: torchsonn.config.schemas.OptimizerConfig
    options:
      members: false

::: torchsonn.config.schemas.SchedulerConfig
    options:
      members: false

::: torchsonn.config.schemas.OutProjTrainConfig
    options:
      members: false
