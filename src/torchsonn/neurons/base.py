import logging
import math
import random
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
from torch import nn

from torchsonn.modules import SONNModule
from torchsonn.types import CriterionType

logger = logging.getLogger(__name__)

# Anything accepted by BasePolynomNeuron.__init__'s `activation` arg.
# None / "" → nn.Identity; str → looked up in _ACTIVATIONS; otherwise a
# pre-built module or plain callable applied to the per-sample scalar output.
ActivationLike = str | nn.Module | Callable[[torch.Tensor], torch.Tensor] | None


def _max_unique_tuples(n: int, k: int, allow_self: bool, ordered: bool) -> int:
    """Number of distinct k-tuples drawn from {0..n-1} under the given flags.

    Used to clamp `max_neuron_models` so the rejection-sampling loop in
    generate_unique_pairs / generate_unique_combinations can't spin forever
    asking for more pairs than mathematically exist (the historical bug:
    n*(n-1)/2 candidates but caller requests > that → infinite loop).
    """
    if n <= 0 or k <= 0:
        return 0
    if allow_self:
        # Replacement allowed: ordered → n^k cartesian product;
        #                      unordered → multiset combinations C(n+k-1, k).
        return n ** k if ordered else math.comb(n + k - 1, k)
    # No replacement: need at least k distinct indices.
    if k > n:
        return 0
    return math.perm(n, k) if ordered else math.comb(n, k)


# Scalar element-wise activations registered for `activation: <name>` YAML
# config. Each value is a zero-arg constructor that returns a fresh
# nn.Module instance — fresh, not shared, because each neuron module owns
# its own activation submodule (avoids torch.func.vmap surprises where a
# stateful module is captured in the function being vmapped).
#
# Activations excluded on purpose:
#   * PReLU      — has learnable per-channel parameters; would need extra
#                  per-neuron state plumbing through `params_batch`.
#   * GLU / Softmax / LogSoftmax — multi-dim, not element-wise scalar.
#   * Threshold  — needs threshold + value args; expose explicitly if needed.
#   * RReLU      — stochastic, breaks reproducibility of vmap'd evaluation.
_ACTIVATIONS = {
    "identity":    nn.Identity,
    "relu":        nn.ReLU,
    "leaky_relu":  nn.LeakyReLU,   # default negative_slope = 0.01
    "elu":         nn.ELU,
    "selu":        nn.SELU,
    "celu":        nn.CELU,
    "gelu":        nn.GELU,
    "silu":        nn.SiLU,        # a.k.a. swish
    "mish":        nn.Mish,
    "tanh":        nn.Tanh,
    "tanhshrink":  nn.Tanhshrink,
    "softplus":    nn.Softplus,
    "softsign":    nn.Softsign,
    "sigmoid":     nn.Sigmoid,
    "log_sigmoid": nn.LogSigmoid,
    "hardtanh":    nn.Hardtanh,
    "hardswish":   nn.Hardswish,
    "hardsigmoid": nn.Hardsigmoid,
}


def generate_unique_pairs(
    n: int,
    max_neuron_models: int,
    seed: int | None = None,
    allow_self: bool = False,
    ordered: bool = True,
) -> list[tuple[int, int]]:
    if seed is not None:
        random.seed(seed)

    cap = _max_unique_tuples(n, 2, allow_self, ordered)
    if max_neuron_models > cap:
        logger.warning(
            "generate_unique_pairs requested %d pairs but only %d unique pairs "
            "exist for n=%d (allow_self=%s, ordered=%s); clamping.",
            max_neuron_models, cap, n, allow_self, ordered,
        )
        max_neuron_models = cap
    if max_neuron_models == 0:
        return []

    pairs = set()
    while len(pairs) < max_neuron_models:
        x1 = random.randrange(n)
        x2 = random.randrange(n)

        if not allow_self and x1 == x2:
            continue

        if not ordered:
            pair = (min(x1, x2), max(x1, x2))
        else:
            pair = (x1, x2)

        pairs.add(pair)

    return list(pairs)


def generate_unique_combinations(
    n: int,
    num_inputs: int,
    max_neuron_models: int,
    seed: int | None = None,
    allow_self: bool = False,
    ordered: bool = True,
) -> list[tuple[int, ...]]:
    """
    Generate unique k-tuples (pairs, triplets, etc.) of indices.

    Args:
        n (int): Number of available inputs.
        num_inputs (int): Size of each tuple (e.g. 2=pair, 3=triplet, etc.).
        max_neuron_models (int): Maximum number of unique combinations to generate.
        seed (int, optional): Random seed for reproducibility.
        allow_self (bool): Whether repeated indices in a tuple are allowed.
        ordered (bool): Whether order matters (permutations) or not (combinations).

    Returns:
        list[tuple[int, ...]]: List of unique tuples.
    """
    if seed is not None:
        random.seed(seed)

    cap = _max_unique_tuples(n, num_inputs, allow_self, ordered)
    if max_neuron_models > cap:
        logger.warning(
            "generate_unique_combinations requested %d %d-tuples but only %d "
            "unique tuples exist for n=%d (allow_self=%s, ordered=%s); clamping.",
            max_neuron_models, num_inputs, cap, n, allow_self, ordered,
        )
        max_neuron_models = cap
    if max_neuron_models == 0:
        return []

    pairs = set()

    while len(pairs) < max_neuron_models:
        if allow_self:
            elems = [random.randrange(n) for _ in range(num_inputs)]
        else:
            elems = random.sample(range(n), num_inputs)

        if not ordered:
            elems = tuple(sorted(elems))
        else:
            elems = tuple(elems)

        pairs.add(elems)

    return list(pairs)


# *****************************************************************************
#   Polynomial neuron class
# *****************************************************************************
class BasePolynomNeuron(SONNModule, ABC):
    num_w = -1

    # Class registry so from_checkpoint_metadata can look up subclasses by name
    # across the split modules — the old code relied on globals() of the single
    # monolithic neuron.py, which doesn't span packages.
    _registry: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BasePolynomNeuron._registry[cls.__name__] = cls

    def __init__(self,
                 num_feat: int,
                 num_src_feat: int,
                 activation: ActivationLike,
                 layer_index: int,
                 start_index: int,
                 dim: int = 2,
                 max_neuron_models: int | None = None,
                 init_method: str = "xavier") -> None:
        super().__init__()
        self.cls = self.__class__.__name__
        self.num_feat = num_feat
        self.num_src_feat = num_src_feat
        self.activation = activation
        self.layer_index = layer_index
        self.start_index = start_index
        self.dim = dim
        self.init_method = init_method
        self.max_neuron_models = max_neuron_models
        # Per-neuron output projection (multi-class, use_neuron_proj=True only).
        # Set via set_proj() after construction; None otherwise.
        self.proj_weight: nn.Parameter | None = None
        self.proj_bias:   nn.Parameter | None = None

        self.src_idxs, num_neurons = self.create_src_idxs(num_feat, max_neuron_models)
        # self.src_idxs is now a tensor with pairs of indices correspond to model input

        self.weight = torch.nn.Parameter(torch.empty((num_neurons, self.num_w)))

        self.created_neuron_idxs = torch.arange(num_neurons) + start_index
        if activation is None or activation == "":
            self.activation = nn.Identity()
        elif isinstance(activation, str):
            key = activation.lower()
            if key not in _ACTIVATIONS:
                raise ValueError(
                    f"Unknown activation {activation!r}. "
                    f"Supported names: {sorted(_ACTIVATIONS)}"
                )
            self.activation = _ACTIVATIONS[key]()
        elif callable(activation):
            # Allow passing an nn.Module instance or a plain callable for
            # ad-hoc activations (e.g. lambdas in tests, custom modules).
            self.activation = activation
        else:
            raise ValueError(
                f"activation must be None, a known name {sorted(_ACTIVATIONS)}, "
                f"or a callable; got {type(activation).__name__}"
            )

        self.reset_parameters()
        self.params_metadata_names.extend([
            # plain vars
            "cls",
            "num_feat",
            "num_src_feat",
            "activation",
            "layer_index",
            "start_index",
            "dim",
            "max_neuron_models",
            # tensors
            "src_idxs",
            "created_neuron_idxs",
        ])

    def set_proj(self, num_classes: int) -> None:
        """Attach per-neuron output projection parameters (xavier weight, zero bias).

        proj_num_classes is stored in params_metadata so from_checkpoint_metadata
        can reconstruct the right parameter shape before load_state_dict runs.
        """
        n = self.weight.shape[0]
        device, dtype = self.weight.device, self.weight.dtype
        self.proj_weight = nn.Parameter(torch.empty(n, num_classes, device=device, dtype=dtype))
        self.proj_bias   = nn.Parameter(torch.zeros(n, num_classes, device=device, dtype=dtype))
        nn.init.xavier_uniform_(self.proj_weight)
        self.proj_num_classes = num_classes
        if "proj_num_classes" not in self.params_metadata_names:
            self.params_metadata_names.append("proj_num_classes")

    @classmethod
    def from_checkpoint_metadata(cls, metadata: dict[str, Any]) -> "BasePolynomNeuron":
        neuron_cls = cls._registry[metadata["cls"]]
        obj = neuron_cls(
            num_feat=metadata["num_feat"],
            num_src_feat=metadata["num_src_feat"],
            activation=metadata["activation"],
            layer_index=metadata["layer_index"],
            start_index=metadata["start_index"],
            dim=metadata["dim"],
            max_neuron_models=metadata["src_idxs"].shape[0],
        )
        if "proj_num_classes" in metadata:
            n = metadata["src_idxs"].shape[0]
            nc = metadata["proj_num_classes"]
            obj.proj_weight = nn.Parameter(torch.zeros(n, nc))
            obj.proj_bias   = nn.Parameter(torch.zeros(n, nc))
        return obj

    def to(self, *args: Any, **kwargs: Any) -> "BasePolynomNeuron":
        self.src_idxs = self.src_idxs.to(*args, **kwargs)
        self.created_neuron_idxs = self.created_neuron_idxs.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def reset_parameters(self) -> None:
        # Bias is folded into self.weight as the constant term (w0), so there is no
        # separate self.bias to initialize.
        if self.init_method == "uniform":
            scale = 0.1  # small to prevent explosion
            nn.init.uniform_(self.weight, -scale, scale)
        elif self.init_method == "xavier":
            nn.init.xavier_uniform_(self.weight)
        else:
            raise NotImplementedError(self.init_method)

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_short_name(self) -> str:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @property
    def num_neurons(self) -> int:
        return self.weight.shape[0]

    @property
    def ensemble_size(self) -> int:
        return self.weight.shape[0]

    def need_bias_tools(self, criterion_type: CriterionType) -> bool:
        return False if criterion_type == CriterionType.cmpValidate else True

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = torch.index_select(inp, 1, self.src_idxs.view(-1)).view(inp.shape[0], -1, 2)
        x_args = self.get_args(x)
        # out = (self.weight * x_args).sum(dim=-1) + self.bias.squeeze(-1)
        out = (self.weight * x_args).sum(dim=-1)
        out = self.activation(out)
        # Inside vmap, weight is 1-D and the "num_neurons" axis is a singleton; drop
        # it so the per-sample output is (batch,), matching PolyQuadratic's contract.
        # SoftBinner / shared_proj downstream require this shape — see model.py:30.
        if self.weight.dim() == 1:
            out = out.squeeze(dim=-1)
        return out

    def create_src_idxs(
        self, num_feat: int, max_neuron_models: int | None
    ) -> tuple[torch.Tensor, int]:
        if max_neuron_models is not None:
            assert max_neuron_models > 0
            # Unordered pairs: for a polynomial neuron y = f(xi, xj), the
            # ordering (i, j) vs (j, i) produces the same closed-form fit
            # (just with the linear coefficients swapped). Requesting ordered
            # pairs would double the ensemble with mathematically equivalent
            # duplicates — wasted compute + half the chance of finding the
            # true best pair within max_neuron_models samples. Matches gmdhpy
            # which uses C(n, 2) unordered combinations.
            src_idxs = generate_unique_pairs(num_feat, max_neuron_models, ordered=False)
        else:
            # generating neuron model ensemble by enumerating all possible combinations of pairs [n*(n-1)/2]
            src_idxs = []
            for u1 in range(0, num_feat):
                for u2 in range(u1 + 1, num_feat):
                    src_idxs.append((u1, u2))

        # Derive num_neurons from the actual list length: generate_unique_pairs
        # clamps when max_neuron_models exceeds the cap, so trusting the user's
        # max_neuron_models here would leave self.weight (max_neuron_models rows)
        # and self.src_idxs (cap rows) with inconsistent leading dims and the
        # subsequent vmap would raise on mixed-size mapped dim.
        num_neurons = len(src_idxs)
        return torch.tensor(src_idxs), num_neurons

    @abstractmethod
    def get_args(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prune(self, idxs: torch.Tensor) -> None:
        self.src_idxs = self.src_idxs.index_select(0, idxs)
        self.created_neuron_idxs = self.created_neuron_idxs.index_select(0, idxs)
        self.weight = nn.Parameter(self.weight.index_select(0, idxs))
        if self.proj_weight is not None:
            self.proj_weight = nn.Parameter(self.proj_weight.index_select(0, idxs))
            self.proj_bias   = nn.Parameter(self.proj_bias.index_select(0, idxs))
