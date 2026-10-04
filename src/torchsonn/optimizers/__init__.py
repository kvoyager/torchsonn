from torchsonn.optimizers.base import BaseOptimizer
from torchsonn.optimizers.adam import BatchedAdam
from torchsonn.optimizers.sgd import BatchedSGD
from torchsonn.optimizers.lbfgs import BatchedLBFGS


optimizer_map = {
    "adam": BatchedAdam,
    "sgd": BatchedSGD,
    "lbfgs": BatchedLBFGS,
}


__all__ = [
    "BaseOptimizer",
    "BatchedAdam",
    "BatchedSGD",
    "BatchedLBFGS",
    "optimizer_map",
]