"""Pair-input ("binary") polynomial neurons.

Each of these consumes exactly two inputs per neuron (dim=2 implicit in the
.view(..., -1, 2) reshape in BasePolynomNeuron.forward / .get_args). The
distinction between classes is purely the polynomial-feature set produced by
get_args — the constant term, linear, covariance, square, cube monomials.
"""
import torch

from torchsonn.neurons.base import BasePolynomNeuron


class LinearPolynomNeuron(BasePolynomNeuron):
    num_w = 3

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        x_args = torch.cat([
            torch.ones((*x.shape[:-1], 1), device=x.device, dtype=x.dtype),
            x],
            dim=2)
        return x_args

    def get_short_name(self) -> str:
        return 'Linear'

    def get_name(self) -> str:
        return 'w0 + w1*xi + w2*xj'


class LinearCovPolynomNeuron(BasePolynomNeuron):
    num_w = 4

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        x_args = torch.cat([
            torch.ones((*x.shape[:-1], 1), device=x.device, dtype=x.dtype),
            x,
            torch.prod(x, dim=2, keepdim=True)],
            dim=2)
        return x_args

    def get_short_name(self) -> str:
        return 'LinearCov'

    def get_name(self) -> str:
        return 'w0 + w1*xi + w2*xj + w3*xi*xj'


class QuadraticPolynomNeuron(BasePolynomNeuron):
    num_w = 6

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        x_args = torch.cat([
            torch.ones((*x.shape[:-1], 1), device=x.device, dtype=x.dtype),
            x,
            x * x,
            torch.prod(x, dim=2, keepdim=True)],
            dim=2)
        return x_args

    def get_short_name(self) -> str:
        return 'Quadratic'

    def get_name(self) -> str:
        return 'full polynom 2nd degree'


class CubicPolynomNeuron(BasePolynomNeuron):
    num_w = 8

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
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
        return 'Cubic'

    def get_name(self) -> str:
        return 'full polynom 3rd degree'