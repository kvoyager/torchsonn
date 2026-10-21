"""N-ary polynomial neuron — variable input arity (dim >= 2)."""
import torch

from torchsonn.neurons.base import ActivationLike, BasePolynomNeuron, generate_unique_combinations


class PolyQuadratic(BasePolynomNeuron):
    """Quadratic neuron over `dim` inputs (reference function 'polyquad').

        y = w0 + sum_i w_i x_i + sum_{i<=j} w_ij x_i x_j

    With `squares=False` the squares x_i^2 are left out, keeping only the
    cross terms i < j. Candidates read unordered `dim`-tuples of inputs.
    """
    def __init__(self,
                 num_feat: int,
                 num_src_feat: int,
                 activation: ActivationLike,
                 layer_index: int,
                 start_index: int,
                 max_neuron_models: int | None = None,
                 init_method: str = "xavier",
                 dim: int = 2,
                 squares: bool = True) -> None:
        assert dim >= 2
        self.exclude_square = not squares
        self.num_w = 1 + dim + dim * (dim + 1) // 2
        if self.exclude_square:
            self.num_w -= dim
        super().__init__(
            num_feat,
            num_src_feat,
            activation,
            layer_index,
            start_index,
            dim,
            max_neuron_models,
            init_method,
        )
        # self.w = nn.Parameter(torch.empty((num_neurons, self.num_w)))
        # self.weight = nn.Parameter(torch.empty((self.num_w)))

    def get_short_name(self) -> str:
        """Return 'poly<dim>', e.g. 'poly5'."""
        return f"poly{self.dim}"

    def get_name(self) -> str:
        """Describe the polynomial: full, or cross terms only when squares are off."""
        if self.exclude_square:
            return f"polynom {self.dim} degree with covariance only"
        else:
            return f"full polynom {self.dim} degree"

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        """Design row of the degree-2 polynomial over `dim` inputs:

            [ 1, x_1..x_D, x_i*x_j for i <= j ]

        with the D diagonal (square) terms dropped when `squares=False`, so the
        width matches `num_w`. Only the last axis is consumed.
        """
        D = x.shape[-1]
        # All pairwise products x_i * x_j as a [..., D, D] outer product, then
        # keep the upper triangle so each unordered pair appears once.
        x_outer = x.unsqueeze(-1) * x.unsqueeze(-2)
        tri_mask = torch.triu(torch.ones(D, D, dtype=torch.bool, device=x.device))
        if self.exclude_square:
            tri_mask = tri_mask ^ torch.eye(D, dtype=torch.bool, device=x.device)
        x_pairs = x_outer[..., tri_mask]
        return torch.cat([
            torch.ones((*x.shape[:-1], 1), device=x.device, dtype=x.dtype),
            x,
            x_pairs],
            dim=-1)

    def create_src_idxs(
        self, num_feat: int, max_neuron_models: int | None
    ) -> tuple[torch.Tensor, int]:
        """Choose the unordered `dim`-tuples of inputs the candidates read.

        With `max_neuron_models` set, draws that many distinct tuples at random
        (fewer if fewer exist). Without it, enumerates all pairs; full
        enumeration is only implemented for `dim == 2`.

        Returns
        -------
        tuple of (torch.Tensor, int)
            The (num_neurons, dim) index tensor and `num_neurons`.
        """
        if max_neuron_models is not None:
            assert max_neuron_models > 0
            # Unordered k-tuples for the same reason ordered pairs are wasteful
            # in BasePolynomNeuron: the polynomial design matrix is symmetric
            # over its k input slots, so permuted k-tuples reach the same OLS
            # minimum. Cap goes from P(n,k) to C(n,k).
            src_idxs = generate_unique_combinations(num_feat, self.dim, max_neuron_models, ordered=False)
        else:
            if self.dim != 2:
                raise NotImplementedError
            src_idxs = []
            for u1 in range(0, num_feat):
                for u2 in range(u1 + 1, num_feat):
                    src_idxs.append((u1, u2))

        # Derive num_neurons from the actual list length — generate_unique_combinations
        # clamps when max_neuron_models exceeds the unique-tuple cap, so trusting
        # max_neuron_models here would leave self.weight and self.src_idxs with
        # inconsistent leading dims (vmap would then fail on mixed-size mapped dim).
        num_neurons = len(src_idxs)
        return torch.tensor(src_idxs), num_neurons