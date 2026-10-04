# Trainer

`Trainer` runs everything that learns. A script calls `set_seed`, `train`,
then optionally `train_out_proj`, `train_finetune` and `prune`, and
`infer` to predict (see [Training](../../guides/training.md) and
[Heads and fine-tuning](../../concepts/heads-and-finetune.md)). The other
methods are the steps `train` runs and the checkpoint helpers.

::: torchsonn.trainer.Trainer

::: torchsonn.trainer.GrowthCriterion

::: torchsonn.logger.setup_logger
