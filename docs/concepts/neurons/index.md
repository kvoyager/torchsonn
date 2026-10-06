# Neuron families

A *family* is a kind of neuron. Every neuron of a family computes the same
formula, each on its own inputs and with its own coefficients. The families
in `model.ref_functions` decide what each layer's candidates look like.

## What a neuron computes

A neuron reads a small tuple of inputs, two by default, and expands it into
a *design row* $\varphi(x_i, x_j)$, a fixed list of terms set by its family.
For `linear_cov` the terms are $1$, $x_i$, $x_j$ and $x_i x_j$; other
families add higher-order terms such as $x_i^2$ and $x_i^3$, as the weight
counts below show. Its output is the weighted sum of the design row, passed
through an optional output activation $a$:

$$
y = a\big(w \cdot \varphi(x_i, x_j)\big)
$$

The weights $w$ are the neuron's coefficients, one per term of the design
row. Their number is the family's weight count:

| Family | Inputs | Weights per neuron | Page |
|---|---|---|---|
| `linear` | 2 | 3 | [Power-basis polynomials](polynomial.md) |
| `linear_cov` | 2 | 4 | [Power-basis polynomials](polynomial.md) |
| `quadratic` | 2 | 6 | [Power-basis polynomials](polynomial.md) |
| `cubic` | 2 | 8 | [Power-basis polynomials](polynomial.md) |
| `polyquad` | `dim` | $1 + \text{dim} + \text{dim}(\text{dim}+1)/2$ with squares | [Power-basis polynomials](polynomial.md) |
| `legendre`, `chebyshev` | `dim` (default 2) | $1 + \text{dim} \cdot \text{degree} + \binom{\text{dim}}{2}$ | [Orthogonal polynomials](orthogonal.md) |
| `rbf` | `dim` (default 2) | $M + \text{dim}$, with $M$ centres | [Gaussian RBF](rbf.md) |

The `rbf` family also learns where its bumps sit and how wide they are, on
top of its weights.

## Candidates

In every layer, each family makes one candidate neuron per pair of inputs,
or per unordered tuple of `dim` inputs. With `model.max_neuron_models` set,
a family draws at most that many tuples at random from the seeded random
generator; without it, the family enumerates every tuple. `polyquad` is the
exception: it enumerates only pairs, so with a `dim` above 2 it needs
`model.max_neuron_models`, and without it building the first layer raises
`NotImplementedError`. A family that
needs more inputs than the layer has sits that layer out. The
[algorithm](../algorithm.md) page counts the inputs per layer.

## Several families in one layer

`model.ref_functions` is a list, and every entry builds one family in every
layer. An entry is a bare name, or a name with a mapping of options:

```yaml
model:
  ref_functions:
    - linear_cov
    - legendre:
        degree: 3
    - polyquad:
        dim: 4
        squares: true
```

The same family may appear more than once with different options, for
example two `polyquad` entries with different `dim`. The candidates of all
families in a layer compete together in [selection](../selection.md), so a
layer's survivors can mix families.

## Output activation

Every entry accepts an `activation:` option, the function $a$ applied to the
neuron's weighted sum. It names one of `identity`, `relu`, `leaky_relu`,
`elu`, `selu`, `celu`, `gelu`, `silu`, `mish`, `tanh`, `tanhshrink`,
`softplus`, `softsign`, `sigmoid`, `log_sigmoid`, `hardtanh`, `hardswish`
and `hardsigmoid`. An empty string or no `activation` key means the
identity:

```yaml
model:
  ref_functions:
    - linear_cov:
        activation: ""     # identity
    - quadratic:
        activation: tanh
```

A bounded activation also bounds what the neuron can predict: `tanh`, for
example, saturates, so on regression targets that range beyond its output
it can cost accuracy relative to the identity. Leave the activation at the
identity unless a bounded output is what the task wants.

## Choosing a family

The families differ in the shape of function they fit, not in a fixed
ranking — which works best depends on the data:

- **Power-basis polynomials** (`linear_cov`, `quadratic`, `cubic`,
  `polyquad`) are the simplest and the most directly readable, and a good
  first choice.
- **Orthogonal polynomials** (`legendre`, `chebyshev`) span the same
  function space as the power basis but stay better conditioned at higher
  degrees, which can make the search more stable.
- **RBF** (`rbf`) fits local structure rather than a global polynomial
  surface, at the cost of more parameters.
- **More inputs per neuron** (a higher `dim`) lets a single neuron capture
  joint interactions that pairwise neurons reach only indirectly.

`linear_cov` is the family of the [regression quickstart](../../getting-started/quickstart-regression.md)
and a sensible first model. Beyond that, the best family is an empirical
question for your data — the families compete together in a layer, so you
can also list several and let [selection](../selection.md) choose among
them.

## The knobs

`model.ref_functions` and its per-entry options, `model.max_neuron_models`,
and for the orthogonal families `model.squash_method`,
`model.squash_n_sigma` and `model.squash_core_range`. All are described in
[Configuration keys](../../reference/config.md).

<small>Checked against TorchSONN 0.1.5.</small>
