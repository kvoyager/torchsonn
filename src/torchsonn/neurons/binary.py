"""Pair-input ("binary") polynomial neurons.

Each of these consumes exactly two inputs per neuron: they never pass `dim`, so
it defaults to 2 and BasePolynomNeuron.forward reshapes the gathered inputs into
pairs (the general `.view(..., -1, self.dim)` reshape resolves to width 2 here).
The distinction between classes is purely the polynomial-feature set produced by
get_args — the constant term, linear, covariance, square, cube monomials.
"""
import torch

from torchsonn.neurons.base import BasePolynomNeuron


class LinearPolynomNeuron(BasePolynomNeuron):
    """Pair neuron y = w0 + w1*xi + w2*xj (reference function 'linear')."""
    num_w = 3

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        """Design row [1, xi, xj] per candidate."""
        x_args = torch.cat([
            torch.ones((*x.shape[:-1], 1), device=x.device, dtype=x.dtype),
            x],
            dim=2)
        return x_args

    def get_short_name(self) -> str:
        """Return 'Linear'."""
        return 'Linear'

    def get_name(self) -> str:
        """Return the formula 'w0 + w1*xi + w2*xj'."""
        return 'w0 + w1*xi + w2*xj'


class LinearCovPolynomNeuron(BasePolynomNeuron):
    """Pair neuron y = w0 + w1*xi + w2*xj + w3*xi*xj (reference function 'linear_cov')."""
    num_w = 4

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        """Design row [1, xi, xj, xi*xj] per candidate."""
        x_args = torch.cat([
            torch.ones((*x.shape[:-1], 1), device=x.device, dtype=x.dtype),
            x,
            torch.prod(x, dim=2, keepdim=True)],
            dim=2)
        return x_args

    def get_short_name(self) -> str:
        """Return 'LinearCov'."""
        return 'LinearCov'

    def get_name(self) -> str:
        """Return the formula 'w0 + w1*xi + w2*xj + w3*xi*xj'."""
        return 'w0 + w1*xi + w2*xj + w3*xi*xj'


class QuadraticPolynomNeuron(BasePolynomNeuron):
    """Pair neuron with every monomial up to degree 2 (reference function 'quadratic')."""
    num_w = 6

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        """Design row [1, xi, xj, xi^2, xj^2, xi*xj] per candidate."""
        x_args = torch.cat([
            torch.ones((*x.shape[:-1], 1), device=x.device, dtype=x.dtype),
            x,
            x * x,
            torch.prod(x, dim=2, keepdim=True)],
            dim=2)
        return x_args

    def get_short_name(self) -> str:
        """Return 'Quadratic'."""
        return 'Quadratic'

    def get_name(self) -> str:
        """Return 'full polynom 2nd degree'."""
        return 'full polynom 2nd degree'


class CubicPolynomNeuron(BasePolynomNeuron):
    """Pair neuron with powers up to 3 and one cross term (reference function 'cubic').

    The row has no mixed cubic terms (xi^2*xj, xi*xj^2).
    """
    num_w = 8

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        """Design row [1, xi, xj, xi^2, xj^2, xi^3, xj^3, xi*xj] per candidate."""
        x2 = x * x
        x_args = torch.cat([
            torch.ones((*x.shape[:-1], 1), device=x.device, dtype=x.dtype),
            x,
            x2,
            x2 * x,
            torch.prod(x, dim=2, keepdim=True)],
            dim=2)
        return x_args

    def get_short_name(self) -> str:
        """Return 'Cubic'."""
        return 'Cubic'

    def get_name(self) -> str:
        """Return 'full polynom 3rd degree'."""
        return 'full polynom 3rd degree'