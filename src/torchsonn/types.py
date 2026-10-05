"""Shared domain types: enums + exceptions used across layers/neurons/model.

Lives here (rather than next to each consumer) so that layer.py / neuron.py /
model.py can import their tags without pulling each other in — both enums and
LayerCreationError have no torch dependencies, so this module sits at the
bottom of the import graph and breaks no cycles.
"""
from enum import Enum


class RefFunctionType(Enum):
    """Neuron family (reference function) tag used in `model.ref_functions`."""
    rfUnknown = -1
    rfLinear = 0
    rfLinearCov = 1
    rfQuadratic = 2
    rfCubic = 3
    rfPolyQuadratic = 4
    rfLegendre = 5
    rfChebyshev = 6
    rfRBF = 7

    @classmethod
    def get_name(cls, value: "RefFunctionType") -> str:
        """Return the display name of `value`, e.g. 'LinearCov' or 'Legendre'."""
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
        elif value == cls.rfRBF:
            return 'RBF'
        else:
            return 'Unknown'

    @classmethod
    def get(cls, arg: "RefFunctionType | str") -> "RefFunctionType":
        """Resolve a config name or alias to a `RefFunctionType`.

        Accepted names: 'linear', 'linear_cov', 'quadratic', 'cubic',
        'polyquad', 'legendre', 'chebyshev', 'rbf'. A `RefFunctionType` is
        returned unchanged.

        Raises
        ------
        ValueError
            If `arg` is not a known name.
        """
        if isinstance(arg, RefFunctionType):
            return arg
        if arg == 'linear':
            return RefFunctionType.rfLinear
        elif arg == 'linear_cov':
            return RefFunctionType.rfLinearCov
        elif arg == 'quadratic':
            return RefFunctionType.rfQuadratic
        elif arg == 'cubic':
            return RefFunctionType.rfCubic
        elif arg == 'polyquad':
            return RefFunctionType.rfPolyQuadratic
        elif arg == 'legendre':
            return RefFunctionType.rfLegendre
        elif arg == 'chebyshev':
            return RefFunctionType.rfChebyshev
        elif arg == 'rbf':
            return RefFunctionType.rfRBF
        else:
            raise ValueError(arg)


class CriterionType(Enum):
    """Layer selection criterion tag used in `train.criterion_type`."""
    cmpValidate = 1
    cmpBias = 2
    cmpComb_validate_bias = 4

    @classmethod
    def get_name(cls, value: "CriterionType") -> str:
        """Return a human-readable description of `value`."""
        if value == cls.cmpValidate:
            return 'validate error comparison'
        elif value == cls.cmpBias:
            return 'bias error comparison'
        elif value == cls.cmpComb_validate_bias:
            return 'bias and validate error comparison'
        else:
            return 'Unknown'

    @classmethod
    def get(cls, arg: "CriterionType | str") -> "CriterionType":
        """Resolve a config name to a `CriterionType`.

        Accepted names: 'validate', 'bias', 'validate_bias'. A `CriterionType` is returned unchanged.

        Raises
        ------
        ValueError
            If `arg` is not a known name.
        """
        if isinstance(arg, CriterionType):
            return arg
        elif arg == 'validate':
            return CriterionType.cmpValidate
        elif arg == 'bias':
            return CriterionType.cmpBias
        elif arg == 'validate_bias':
            return CriterionType.cmpComb_validate_bias
        else:
            raise ValueError(
                f"train.criterion_type={arg!r}; expected 'validate', 'bias' or 'validate_bias'."
            )


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