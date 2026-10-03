# Neuron families

A *family* is a kind of neuron. Every neuron of a family computes the same
formula, each on its own inputs and with its own coefficients. The families
in `model.ref_functions` decide what each layer's candidates look like.

## What a neuron computes

A neuron reads a small tuple of inputs, two by default, and expands it into
a *design row* $\varphi(x_i, x_j)$, a fixed list of terms such as $1$,
$x_i$, $x_j$ and $x_i x_j$. Its output is the weighted sum of the design
row, passed through an optional output activation $a$:

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
generator; without it, the family enumerates every tuple. A family that
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

A bounded activation also bounds what the neuron can predict. On California
housing, whose target runs up to 5, `tanh` neurons gave a test MSE of 0.2191
against 0.1877 without an activation, with a layer error of 1.87 (single
runs, recorded in `california_housing_legendre_finetune.yaml`).

## Choosing a family

What the tutorials measured:

- **Legendre is the strongest single family on California housing.** Over
  four seeds with 24 survivors, Legendre of degree 3 reaches a test MSE of
  0.1852 ± 0.0008. In the same configuration, `linear_cov` instead of
  Legendre gives 0.1990 ± 0.0021 and `quadratic` 0.2027 ± 0.0054.
- **RBF matches Legendre's accuracy with more parameters.** `rbf` with 8
  centres per neuron reaches 0.1853 ± 0.0014 in the same configuration. After
  pruning, the best Legendre model has 801 learned parameters and the
  smallest RBF model 6,485.
- **Degree and basis matter less than expected.** Legendre of degree 4 or 5,
  Chebyshev instead of Legendre, and RBF with 4 to 16 centres all land
  within the seed noise of the defaults (California housing README).
- **More inputs per neuron can matter more than a wider search.** On the
  CCPP power-plant data, adding a four-input Legendre family (`dim: 4`)
  lowers the mean absolute error from 3.31 to 3.28 MW, while widening the
  Legendre search alone gives 3.30 (CCPP README).

`linear_cov` is the family of the [regression quickstart](../../getting-started/quickstart-regression.md)
and a sensible first model; the tutorials' best configurations use
`legendre`.

## The knobs

`model.ref_functions` and its per-entry options, `model.max_neuron_models`,
and for the orthogonal families `model.squash_method`,
`model.squash_n_sigma` and `model.squash_core_range`. All are described in
[Configuration keys](../../reference/config.md).

<small>Checked against TorchSONN 0.1.5.</small>
