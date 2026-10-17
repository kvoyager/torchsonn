# MLP

The polynomial and RBF families fit a fixed kind of basis: monomials,
orthogonal polynomials or Gaussian bumps, with one weight per term. The
`mlp` family learns its own terms. Each neuron is a small neural network
over its inputs: one hidden layer of a few units, read out by a weighted
sum. The hidden units are fitted together with the output weights, so the
directions the neuron responds to are learned from the data rather than
fixed in advance.

## The neuron

A neuron reads `dim` inputs $u_1, \dots, u_{\text{dim}}$, two by default.

**Input transforms.** Two optional steps act on each input, in this order;
both are off by default, and the neuron then reads its inputs as they are.

- `standardize: true` subtracts the input's mean and divides by its
  standard deviation.
- `squash: true` maps the input into $(-1, 1)$ with the same squash the
  [orthogonal families](orthogonal.md#the-squash) use: `squash_method:
  sigma` keeps a linear core around the mean and saturates only far out,
  `tanh` applies the hyperbolic tangent.

Call the result $z$. The statistics come from the
[input pass](../algorithm.md#3-input-pass), which runs only when one of the
two steps needs them. With the `sigma` squash, standardizing first changes
nothing, because the squash standardizes on its own; with `tanh`,
`standardize` sets the scale the tangent sees.

**Expanded input.** The hidden layer reads the vector

$$
f = \big[\ z_1, \dots, z_{\text{dim}},\ \ z_i z_j \ (i < j),\ \ z_1^2, \dots, z_{\text{dim}}^2\ \big]
$$

where the pairwise products are included with `cov: true` and the squares
with `squares: true`. Without either, $f = z$. A smooth hidden unit can
approximate a product, but only with several units; handing it the product
directly is cheaper.

**Hidden layer.** $H$ units, each a weighted sum of $f$ plus a bias,
passed through the hidden activation $\sigma$ (SiLU, also called swish, by
default):

$$
h = \sigma(W_1 f + b_1)
$$

**Output.** The neuron's output is a weighted sum of the hidden units and,
with `skip: true` (the default), of the expanded input itself, passed
through the output activation $a$:

$$
y = a\big(w_0 + v \cdot f + w_h \cdot h\big)
$$

The terms $v \cdot f$ are the *skip* path. With it, the neuron contains the
polynomial over the same terms as a special case ($w_h = 0$): the `linear`
neuron by default, or the `quadratic` one with `cov` and `squares`. The
hidden layer then fits what that polynomial misses. With `skip: false` the
output reads the hidden units only.

The output weights $[w_0, v, w_h]$ are the neuron's weights, the same as
every family's, and `train.ridge_alpha` applies to them. The hidden layer's
$W_1$ and $b_1$ are fitted with them by the same candidate fit but are not
penalized.

A network with `mlp` neurons is not a polynomial (see
[the Kolmogorov-Gabor polynomial](../kolmogorov-gabor.md)).

## Hidden width

`hidden` sets $H$. The default is $2 \cdot \text{dim} + 1$: 5 for a pair,
7 for three inputs. The number comes from the Kolmogorov-Arnold
representation theorem, which writes any continuous function of $d$
variables through $2d + 1$ inner functions. That is a guide, not a
guarantee: the theorem's inner functions are not smooth units like these.

The width sets the neuron's size. With $F$ entries in the expanded input,
a neuron has $H(F + 1)$ hidden-layer parameters and $1 + F + H$ output
weights with `skip` ($1 + H$ without):

| `dim` | `cov` and `squares` | $F$ | Parameters, default `hidden` |
|---|---|---|---|
| 2 | off | 2 | 23 |
| 2 | on | 5 | 41 |
| 3 | off | 3 | 39 |
| 3 | on | 9 | 87 |

For comparison, a `quadratic` neuron has 6 weights. A layer fits many
candidates and keeps those with the lowest error on the dev split, so a
neuron with too many parameters can fit the train split well without
ranking any better on dev. Keep `hidden` small unless a larger value
measurably helps.

## Initialization

The hidden weights start from a Xavier uniform draw per neuron and the
hidden biases at zero. The output weights follow `init_method`, as in every
family. All draws come from PyTorch's random generator, so
[`Trainer.set_seed`](../../guides/training.md#seeds) before building the
model reproduces them.

## Options

| Option | Default | Meaning |
|---|---|---|
| `dim` | 2 | inputs per neuron; at least 2 |
| `hidden` | null | hidden units; null means $2 \cdot \text{dim} + 1$ |
| `cov` | false | add the pairwise products to the expanded input |
| `squares` | false | add the squared inputs to the expanded input |
| `skip` | true | also read the expanded input in the output sum |
| `hidden_activation` | `silu` | activation of the hidden units; any name `activation` accepts |
| `standardize` | false | standardize the inputs first |
| `squash` | false | squash the inputs into $(-1, 1)$ |
| `squash_method` | `model.squash_method` | `sigma` or `tanh` |
| `squash_n_sigma` | `model.squash_n_sigma` | half-width of the linear core, in standard deviations |
| `squash_core_range` | `model.squash_core_range` | squashed value at the core's edge |
| `activation` | identity | output activation |

```yaml
model:
  ref_functions:
    - mlp:
        cov: true
        standardize: true
```

## Input scale

With both transforms off, the hidden units see the inputs at whatever scale
they arrive. In the first layer that is the preprocessed features; deeper
layers read other neurons' outputs, which follow the target's scale. On
large inputs a SiLU unit is nearly flat on one side and nearly linear on
the other, so it adds little curvature, and the products and squares make
large inputs larger. If the hidden units do not help, turn on
`standardize`, or `squash` to also bound outliers.

## In the log

After selection, the trainer logs one line per `mlp` family: the number of
survivors and the mean absolute output weight on the hidden units and on
the skip path. A hidden-unit weight far below the skip weight means the
survivors use the neuron mostly as its polynomial.

The short name is `MLP<H>`, with the number of inputs when it is not 2:
`MLP5`, `MLP7x3`.

## The knobs

`model.ref_functions` with the options above, `model.squash_method`,
`model.squash_n_sigma`, `model.squash_core_range` and `train.ridge_alpha`.
See [Configuration keys](../../reference/config.md).

<small>Checked against TorchSONN 0.1.8.</small>
