"""Orthogonal-polynomial neurons: Legendre and Chebyshev.

These swap CubicPolynomNeuron's per-input monomial power basis {x, x^2, x^3, ...}
for an orthogonal-polynomial basis evaluated through its stable three-term
recurrence. They accept a variable input arity `dim` (>= 2): the classic pair
("binary") neuron — every neuron consuming one pair (xi, xj) — is the dim=2
default, and dim > 2 gives a multi-input neuron in the spirit of PolyQuadratic
(neurons/poly.py), but on the orthogonal basis.

Why an orthogonal basis:
  * Legendre  P_k — orthogonal under the uniform weight on [-1, 1].
  * Chebyshev T_k — orthogonal under the (1 - x^2)^-1/2 weight on [-1, 1]; the
    minimax basis, so a truncated expansion is near-optimal in the sup norm.
The raw power basis x^k has a Hilbert-matrix Gram matrix — catastrophically
ill-conditioned as the degree grows — which makes the per-neuron least-squares
fit blow up. The orthogonal bases stay well conditioned, so higher `degree`
neurons remain trainable.

Interaction structure (the dim > 2 design choice): a neuron over n = `dim`
inputs expanded to degree d uses
  * the constant 1,
  * the univariate columns P_1(u_i)..P_d(u_i) for each of the n inputs  (n*d), and
  * (cross=True) one bilinear u_i*u_j per input *pair* — C(n, 2) of them.
That is additive univariate terms plus pairwise interactions only — NOT the full
tensor product (every mixed P_a(u_i)*P_b(u_j)*..., whose C(n+d, d) columns
explode the width and wreck conditioning). It mirrors PolyQuadratic (likewise
pairwise, not full tensor product) and the "sparsity of interactions" principle
behind PCE hyperbolic truncation: real systems are dominated by low-order
interactions. For n=4, d=3 the width is 1 + 4*3 + 6 = 19.

Domain caveat: both families are orthogonal — and bounded, |P_k|, |T_k| <= 1 —
only on [-1, 1]. Real inputs (StandardScaler output, deeper-layer activations)
routinely land outside it, where T_k(x) ~ (2x)^k explodes and destroys the
fit's conditioning. `squash=True` (the default) therefore maps each input
through tanh into (-1, 1) before the recurrence, mirroring the standard trick
in Chebyshev-KAN. Set squash=False when inputs are already confined to [-1, 1].
"""
import itertools

import torch

from torchsonn.neurons.base import (
    ActivationLike,
    BasePolynomNeuron,
    generate_unique_combinations,
)


class BaseOrthogonalNeuron(BasePolynomNeuron):
    """Shared machinery for neurons built on an orthogonal-polynomial basis.

    For `dim` inputs (xi, xj, ...) — optionally squashed to (ui, uj, ...) —
    expanded to `degree` d, the design row is

        [ 1,
          P_1(u1), ..., P_1(un), P_2(u1), ..., P_d(un),   # n*d univariate columns
          u_i*u_j for each input pair (i, j) ]            # only when cross=True

    At dim=2 this is exactly CubicPolynomNeuron's layout when d=3, with the power
    basis replaced by the orthogonal one. Subclasses supply the family purely
    through `_recurrence_coeffs`, the three-term recurrence multipliers.
    """

    def __init__(self,
                 num_feat: int,
                 num_src_feat: int,
                 activation: ActivationLike,
                 layer_index: int,
                 start_index: int,
                 max_neuron_models: int | None = None,
                 init_method: str = "xavier",
                 degree: int = 3,
                 cross: bool = True,
                 squash: bool = True,
                 dim: int = 2) -> None:
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        self.degree = int(degree)
        self.cross = bool(cross)
        self.squash = bool(squash)
        self.dim = int(dim)
        # Input index pairs (i, j) for the bilinear cross terms, computed once
        # here so get_args iterates a fixed, construction-time list — vmap-safe,
        # no data-dependent control flow. Empty when cross=False. At dim=2 this
        # is the single pair [(0, 1)] (the classic ui*uj term); for dim > 2 it is
        # the C(dim, 2) input pairs, i.e. pairwise interactions only.
        self._cross_pairs = (
            list(itertools.combinations(range(self.dim), 2)) if self.cross else []
        )
        # constant + (P_1..P_degree evaluated at each of `dim` inputs) + one
        # bilinear term per input pair. Must be set before super().__init__ since
        # BasePolynomNeuron allocates self.weight with this width.
        self.num_w = 1 + self.dim * self.degree + len(self._cross_pairs)
        super().__init__(
            num_feat,
            num_src_feat,
            activation,
            layer_index,
            start_index,
            dim=self.dim,
            max_neuron_models=max_neuron_models,
            init_method=init_method,
        )
        # Persist the shape-affecting knobs so _construct_from_metadata can
        # rebuild an identically-shaped neuron from a checkpoint. `dim` is
        # already persisted by BasePolynomNeuron.__init__.
        self.params_metadata_names.extend(["degree", "cross", "squash"])

    @classmethod
    def _construct_from_metadata(cls, metadata: dict) -> "BaseOrthogonalNeuron":
        # num_w depends on degree/cross/dim, so the base implementation — which
        # forwards only `dim` — would rebuild the weight at the wrong width.
        # Forward the saved knobs (including dim) instead.
        return cls(
            num_feat=metadata["num_feat"],
            num_src_feat=metadata["num_src_feat"],
            activation=metadata["activation"],
            layer_index=metadata["layer_index"],
            start_index=metadata["start_index"],
            max_neuron_models=metadata["src_idxs"].shape[0],
            degree=metadata["degree"],
            cross=metadata["cross"],
            squash=metadata["squash"],
            dim=metadata["dim"],
        )

    def create_src_idxs(
        self, num_feat: int, max_neuron_models: int | None
    ) -> tuple[torch.Tensor, int]:
        # `dim`-ary input tuples (pairs at dim=2, triplets at dim=3, ...). The
        # orthogonal design row is symmetric over its input slots — permuted
        # tuples reach the same least-squares fit — so unordered tuples suffice
        # (cap C(n, dim), not P(n, dim)). Mirrors PolyQuadratic.create_src_idxs;
        # the base pair-only version can't express dim > 2.
        if max_neuron_models is not None:
            assert max_neuron_models > 0
            src_idxs = generate_unique_combinations(
                num_feat, self.dim, max_neuron_models, ordered=False
            )
        else:
            # Exhaustive enumeration of every unordered dim-tuple of inputs. At
            # dim=2 this reproduces the historical pair double-loop order
            # [(0,1), (0,2), ...]; itertools.combinations generalizes it to any
            # dim. Empty (num_neurons == 0) when num_feat < dim — create_layer
            # then skips this family for the layer until shortcut widening
            # supplies enough inputs.
            src_idxs = list(itertools.combinations(range(num_feat), self.dim))

        # Derive num_neurons from the actual list length: generate_unique_combinations
        # clamps when max_neuron_models exceeds the unique-tuple cap, so trusting
        # max_neuron_models here would leave self.weight and self.src_idxs with
        # inconsistent leading dims (vmap would then fail on mixed-size mapped dim).
        num_neurons = len(src_idxs)
        return torch.tensor(src_idxs), num_neurons

    @staticmethod
    def _recurrence_coeffs(k: int) -> tuple[float, float]:
        """Multipliers (a, c) of the three-term recurrence at index k:

            P_k(u) = a * u * P_{k-1}(u) - c * P_{k-2}(u),   with P_0 = 1, P_1 = u.

        Both Legendre and Chebyshev fit this shape (no u-free term), so a pair
        of scalars fully specifies the family.
        """
        raise NotImplementedError

    def _orthopoly_columns(self, u: torch.Tensor) -> list[torch.Tensor]:
        """Return [P_1(u), ..., P_degree(u)]; each entry keeps u's shape.

        u carries all `dim` inputs in its last axis, so each returned column is
        the polynomial evaluated element-wise across every input at once —
        concatenating them yields the n*degree univariate block.

        A plain Python loop over the fixed, construction-time `degree` — no
        data-dependent control flow — so it stays vmap-safe like the rest of
        the neuron forwards.
        """
        cols = [u]  # P_1 = u
        p_prev2 = torch.ones_like(u)  # P_0
        p_prev = u                    # P_1
        for k in range(2, self.degree + 1):
            a, c = self._recurrence_coeffs(k)
            p_k = a * u * p_prev - c * p_prev2
            cols.append(p_k)
            p_prev2, p_prev = p_prev, p_k
        return cols

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(x) if self.squash else x
        parts = [torch.ones((*u.shape[:-1], 1), device=u.device, dtype=u.dtype)]
        parts.extend(self._orthopoly_columns(u))
        if self.cross:
            # One bilinear u_i*u_j per input pair — squashed like the univariate
            # terms so every column shares the same bounded domain when
            # squash=True. Iterating the precomputed pair list gives the C(dim, 2)
            # pairwise products; note this is deliberately NOT torch.prod(u,
            # dim=-1), which for dim > 2 collapses to the single dim-way product
            # u_1*u_2*...*u_dim rather than the pairwise interactions we want.
            for i, j in self._cross_pairs:
                parts.append((u[..., i] * u[..., j]).unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def _basis_name(self) -> str:
        raise NotImplementedError

    def get_short_name(self) -> str:
        # Append the arity only when non-default so a pair neuron stays
        # "Legendre3" (matching the historical short name) while a multi-input
        # one reads e.g. "Legendre3x4" (degree 3 over 4 inputs).
        suffix = f"x{self.dim}" if self.dim != 2 else ""
        return f"{self._basis_name()}{self.degree}{suffix}"

    def get_name(self) -> str:
        cross = " + pairwise cross terms" if self.cross else ""
        squash = "tanh-squashed " if self.squash else ""
        return (f"{self._basis_name()} basis (degree {self.degree}) over "
                f"{squash}{self.dim} inputs{cross}")


class LegendrePolynomNeuron(BaseOrthogonalNeuron):
    """Neuron on the Legendre basis P_k (uniform-weight orthogonal on [-1, 1]).

    Bonnet's recursion: P_k = ((2k-1)/k) u P_{k-1} - ((k-1)/k) P_{k-2},
    with P_0 = 1, P_1 = u.
    """

    @staticmethod
    def _recurrence_coeffs(k: int) -> tuple[float, float]:
        return (2 * k - 1) / k, (k - 1) / k

    def _basis_name(self) -> str:
        return "Legendre"


class ChebyshevPolynomNeuron(BaseOrthogonalNeuron):
    """Neuron on the Chebyshev-(first-kind) basis T_k (minimax on [-1, 1]).

    Recurrence: T_k = 2 u T_{k-1} - T_{k-2}, with T_0 = 1, T_1 = u.
    """

    @staticmethod
    def _recurrence_coeffs(k: int) -> tuple[float, float]:
        return 2.0, 1.0

    def _basis_name(self) -> str:
        return "Chebyshev"
