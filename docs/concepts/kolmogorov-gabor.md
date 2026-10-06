# The Kolmogorov-Gabor polynomial

!!! note "Notation"
    $x_1, \dots, x_m$ are the model's $m$ inputs and $y$ the target. $d$ is
    a polynomial's degree. $a$ are the coefficients of the full polynomial,
    and $w$ the coefficients of a single neuron.

GMDH describes the relation between the inputs and the target as a
polynomial of the inputs, the Kolmogorov-Gabor polynomial:

$$
y = a_0 + \sum_{i} a_i x_i + \sum_{i \le j} a_{ij}\, x_i x_j + \sum_{i \le j \le k} a_{ijk}\, x_i x_j x_k + \dots
$$

Over a bounded range of the inputs, a polynomial of high enough degree
approximates any continuous relation as closely as needed (the Weierstrass
approximation theorem), which makes it a general-purpose model form.

The GMDH literature calls this the Kolmogorov-Gabor polynomial and
describes it as the discrete form of the Volterra functional series
([[6]](gmdh.md#ref-6), p. 19). Ivakhnenko credits D. Gabor with
introducing such polynomial decision functions around 1960
[[2]](gmdh.md#ref-2), in a learning filter and predictor
[[7]](gmdh.md#ref-7).

## Why it is not fitted directly

A polynomial of degree $d$ in $m$ inputs has $\binom{m+d}{d}$ coefficients:

| Inputs $m$ | $d = 2$ | $d = 3$ | $d = 4$ | $d = 6$ | $d = 8$ |
|---|---|---|---|---|---|
| 8 | 45 | 165 | 495 | 3,003 | 12,870 |
| 16 | 153 | 969 | 4,845 | 74,613 | 735,471 |

The [California housing tutorial](../tutorials/california-housing.md)
feeds the model 16 features and fits it on 12,384 training rows. At degree
6 the full polynomial already has six times more coefficients than there
are rows, so least squares has no unique solution, and nothing says which
of the terms the data needs. Fitting all of them fits the noise.

## Partial descriptions

GMDH builds the polynomial from small pieces instead, called partial
descriptions: low-degree polynomials of two inputs. The classic one is the
quadratic

$$
y = w_0 + w_1 x_i + w_2 x_j + w_3 x_i x_j + w_4 x_i^2 + w_5 x_j^2 ,
$$

six coefficients, a small least-squares problem. The pieces are composed layer by layer: the outputs of one layer are the
inputs of the next, so a quadratic of two quadratics has degree 4, and $L$
layers of quadratics reach degree $2^L$. At every layer the external
criterion keeps only the compositions that help on fresh data (see
[GMDH](gmdh.md)). The network grows high-degree terms only where the data
supports them, while every single fit stays a small, well-posed problem.

The [regression quickstart](../getting-started/quickstart-regression.md)
shows the effect. Its model has 10 layers of `linear_cov` neurons, each of
degree 2, so the final neuron is a polynomial of degree up to $2^{10} =
1024$ in the 8 features (leaving aside the clamp that bounds each layer's
outputs, which only acts on extreme values). A full polynomial of that degree would have about
$3 \times 10^{19}$ coefficients. After pruning, the model has 11 neurons
and 44 coefficients.

## TorchSONN's families on this picture

**Power-basis families** are truncations of the Kolmogorov-Gabor polynomial
in their inputs:

| Family | Terms over the inputs $x_i, x_j$ | Coefficients |
|---|---|---|
| `linear` | $1,\ x_i,\ x_j$ | 3 |
| `linear_cov` | $1,\ x_i,\ x_j,\ x_i x_j$ | 4 |
| `quadratic` | $1,\ x_i,\ x_j,\ x_i^2,\ x_j^2,\ x_i x_j$ (all of degree 2) | 6 |
| `cubic` | the `quadratic` terms plus $x_i^3,\ x_j^3$ (no mixed cubic terms) | 8 |
| `polyquad` | all terms up to degree 2 in `dim` inputs (the squares are optional) | $1 + \text{dim} + \text{dim}(\text{dim}+1)/2$ with the squares |

**Orthogonal families** (`legendre`, `chebyshev`) use the same building
blocks on a better-conditioned basis: for each input, the orthogonal
polynomials of degrees 1 to `degree`, plus one product $u_i u_j$ for each
pair of inputs. They keep pairwise interactions only, not every mixed
term, on the assumption that low-order interactions dominate real systems.
Before the basis is applied, each input is squashed into $[-1, 1]$. The
default squash is linear within ±2 standard deviations of the input's mean
and saturates beyond, so on the bulk of the data a network of orthogonal
neurons is still a polynomial of its inputs.

**The RBF family** is not a truncation of the polynomial at all. Its
partial description is a weighted sum of normalized Gaussian bumps plus
a linear part, with learnable bump centres and widths. A network with `rbf` survivors is a
GMDH-type network whose partial descriptions are not polynomials, and the
Kolmogorov-Gabor picture describes only its polynomial neurons.

[Neuron families](neurons/index.md) gives every family's formula and
options.

<small>Checked against TorchSONN 0.1.5.</small>
