from torchsonn.optimizers.base import BaseOptimizer
from torchsonn.optimizers.adam import BatchedAdam
from torchsonn.optimizers.sgd import BatchedSGD
from torchsonn.optimizers.lbfgs import BatchedLBFGS
from torchsonn.optimizers.newton import BatchedNewton
from torchsonn.optimizers.newton_lm import BatchedNewtonLM


optimizer_map = {
    "adam": BatchedAdam,
    "sgd": BatchedSGD,
    "lbfgs": BatchedLBFGS,
    "newton": BatchedNewton,
    "newton-lm": BatchedNewtonLM,
}


__all__ = [
    "BaseOptimizer",
    "BatchedAdam",
    "BatchedSGD",
    "BatchedLBFGS",
    "BatchedNewton",
    "BatchedNewtonLM",
    "optimizer_map",
]