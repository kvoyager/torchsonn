"""N-ary polynomial neuron — variable input arity (dim >= 2)."""
import torch

from torchsonn.neurons.base import ActivationLike, BasePolynomNeuron, generate_unique_combinations


class PolyQuadratic(BasePolynomNeuron):
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
        Compute polynomial of degree N:
        y = w0 + sum_i w1_i*x_i + sum_{i<=j} w2_ij*x_i*x_j
        Args:
            x: [B, D] input
            w: [1 + D + D*(D+1)//2] weights
        Returns:
            y: [B]
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
        return f"poly{self.dim}"

    def get_name(self) -> str:
        if self.exclude_square:
            return f"polynom {self.dim} degree with covariance only"
        else:
            return f"full polynom {self.dim} degree"

    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        # PolyQuadratic.forward bypasses BasePolynomNeuron's get_args path —
        # it expands the polynomial inline using the variable-arity (dim)
        # cross-term mask. This abstract override exists only to satisfy
        # BasePolynomNeuron's interface; never called.
        raise NotImplementedError("PolyQuadratic.forward handles its own term expansion.")

    def create_src_idxs(
        self, num_feat: int, max_neuron_models: int | None
    ) -> tuple[torch.Tensor, int]:
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