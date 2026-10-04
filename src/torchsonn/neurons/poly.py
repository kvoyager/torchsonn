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

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Compute the degree-2 polynomial of each neuron's `dim` inputs:
        y = w0 + sum_i w1_i*x_i + sum_{i<=j} w2_ij*x_i*x_j

        Parameters
        ----------
        inp : torch.Tensor
            Layer input, (..., num_feat). Each neuron reads its `dim`
            columns; its weights are 1 + D + D*(D+1)//2 coefficients for
            D = `dim` (fewer without the squares).

        Returns
        -------
        torch.Tensor
            (..., num_neurons) neuron outputs, or (...,) when the weights
            are 1-D (inside the trainer's vmap over candidates).
        """
        inp_x = inp.view(-1, inp.shape[-1])

        x = torch.index_select(inp_x, -1, self.src_idxs.view(-1)).view(inp_x.shape[0], -1, self.dim)

        B, T, D = x.shape
        idx = 0

        weight = self.weight.unsqueeze(dim=0) if len(self.weight.shape) == 1 else self.weight

        # constant term
        y = weight[:, idx]
        idx += 1

        # linear terms
        y = y + (x * weight[None, :, idx:idx + D]).sum(dim=2)
        idx += D

        # quadratic terms (x_i * x_j)
        x_expanded = x.unsqueeze(3) * x.unsqueeze(2)  # [B, D, D]

        # mask for upper right triangle (i <= j)
        tri_mask = torch.triu(torch.ones(D, D, dtype=torch.bool, device=x.device))

        # optionally remove diagonal (square) terms
        if self.exclude_square:
            tri_mask = tri_mask ^ torch.eye(D, dtype=torch.bool, device=x.device)

        # take only values that has mask==True and flatten them
        x_pairs = x_expanded[:, :, tri_mask]  # [B, D*(D+1)//2]
        y = y + (x_pairs * weight[None, :, idx:]).sum(dim=2)

        out = self.activation(y)

        out = out.view((*inp.shape[:-1], weight.shape[0]))
        if len(self.weight.shape) == 1:
            out = out.squeeze(dim=-1)

        return out

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
        """Not used: `forward` expands the terms itself. Raises NotImplementedError."""
        # PolyQuadratic.forward bypasses BasePolynomNeuron's get_args path —
        # it expands the polynomial inline using the variable-arity (dim)
        # cross-term mask. This abstract override exists only to satisfy
        # BasePolynomNeuron's interface; never called.
        raise NotImplementedError("PolyQuadratic.forward handles its own term expansion.")

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