"""Shared domain types: enums + exceptions used across layers/neurons/model.

Lives here (rather than next to each consumer) so that layer.py / neuron.py /
model.py can import their tags without pulling each other in — both enums and
LayerCreationError have no torch dependencies, so this module sits at the
bottom of the import graph and breaks no cycles.
"""
from enum import Enum


class RefFunctionType(Enum):
    rfUnknown = -1
    rfLinear = 0
    rfLinearCov = 1
    rfQuadratic = 2
    rfCubic = 3
    rfPolyQuadratic = 4
    rfLegendre = 5
    rfChebyshev = 6

    @classmethod
    def get_name(cls, value: "RefFunctionType") -> str:
        if value == cls.rfUnknown:
            return 'Unknown'
        elif value == cls.rfLinear:
            return 'Linear'
        elif value == cls.rfLinearCov:
            return 'LinearCov'
        elif value == cls.rfQuadratic:
            return 'Quadratic'
        elif value == cls.rfCubic:
            return 'Cubic'
        elif value == cls.rfPolyQuadratic:
            return 'PolyQuadratic'
        elif value == cls.rfLegendre:
            return 'Legendre'
        elif value == cls.rfChebyshev:
            return 'Chebyshev'
        else:
            return 'Unknown'

    @classmethod
    def get(cls, arg: "RefFunctionType | str") -> "RefFunctionType":
        if isinstance(arg, RefFunctionType):
            return arg
        if arg == 'linear':
            return RefFunctionType.rfLinear
        elif arg in ('linear_cov', 'lcov'):
            return RefFunctionType.rfLinearCov
        elif arg in ('quadratic', 'quad'):
            return RefFunctionType.rfQuadratic
        elif arg == 'cubic':
            return RefFunctionType.rfCubic
        elif arg == 'polyquad':
            return RefFunctionType.rfPolyQuadratic
        elif arg in ('legendre', 'leg'):
            return RefFunctionType.rfLegendre
        elif arg in ('chebyshev', 'cheb'):
            return RefFunctionType.rfChebyshev
        else:
            raise ValueError(arg)


class CriterionType(Enum):
    cmpValidate = 1
    cmpBias = 2
    cmpComb_validate_bias = 4
    cmpComb_bias_retrain = 5

    @classmethod
    def get_name(cls, value: "CriterionType") -> str:
        if value == cls.cmpValidate:
            return 'validate error comparison'
        elif value == cls.cmpBias:
            return 'bias error comparison'
        elif value == cls.cmpComb_validate_bias:
            return 'bias and validate error comparison'
        elif value == cls.cmpComb_bias_retrain:
            return 'bias error comparison with retrain'
        else:
            return 'Unknown'

    @classmethod
    def get(cls, arg: "CriterionType | str") -> "CriterionType":
        if isinstance(arg, CriterionType):
            return arg
        elif arg == 'validate':
            return CriterionType.cmpValidate
        elif arg == 'bias':
            return CriterionType.cmpBias
        elif arg == 'validate_bias':
            return CriterionType.cmpComb_validate_bias
        elif arg in ('bias_retrain', 'bias_refit'):
            return CriterionType.cmpComb_bias_retrain
        else:
            raise ValueError(arg)


class LayerCreationError(Exception):
    """Raised when layer creation fails (e.g. no reference functions configured)."""

    def __init__(self, message: str, layer_index: int) -> None:
        super().__init__(message)
        self.layer_index = layer_index


__all__ = [
    "RefFunctionType",
    "CriterionType",
    "LayerCreationError",
]