# Orthogonal polynomials

The `legendre` and `chebyshev` families build their design rows from
orthogonal polynomials of each input instead of plain powers. On $[-1, 1]$
these polynomials are bounded by 1 and stay distinct from each other as
the degree rises, so the least-squares problems for their weights stay well
conditioned where the [power basis](polynomial.md) does not.

## The design row

A neuron over `dim` inputs first squashes each input $x_k$ into
$u_k \in (-1, 1)$ (see [The squash](#the-squash)). Its design row is then

$$
\big[\ 1,\quad P_1(u_k), \dots, P_d(u_k)\ \text{for every input } k,\quad u_k u_l\ \text{for every pair } k < l\ \big]
$$

where $P_1, \dots, P_d$ are the family's polynomials up to the degree
$d$ = `degree`. That makes $1 + \text{dim} \cdot d + \binom{\text{dim}}{2}$
weights: 8 for a pair at degree 3, and 19 for four inputs at degree 3.

Each input gets its own polynomials, and inputs interact only through the
pairwise products $u_k u_l$. A full tensor product, every mixed term
$P_a(u_k) P_b(u_l) \cdots$, would have far more columns. Keeping only the
pairwise terms assumes that low-order interactions carry most of the
signal; the products can be switched off with `cross: false`.

## The polynomials

Both families follow a three-term recurrence from $P_0 = 1$ and $P_1 = u$:

- **Legendre** (`legendre`): $P_k(u) = \frac{2k-1}{k}\, u\, P_{k-1}(u) - \frac{k-1}{k}\, P_{k-2}(u)$.
  The polynomials are orthogonal under a uniform weight on $[-1, 1]$.
- **Chebyshev of the first kind** (`chebyshev`): $T_k(u) = 2u\, T_{k-1}(u) - T_{k-2}(u)$.
  They are orthogonal under the weight $1/\sqrt{1 - u^2}$, and a truncated
  Chebyshev expansion is close to the best approximation in the maximum
  error.

On California housing, Chebyshev in place of Legendre lands within the seed
noise (California housing README).

## The squash

The polynomials are bounded only on $[-1, 1]$; outside it they grow like
$u^d$. Every input therefore passes through a squash first, chosen by
`model.squash_method` or per entry.

**`sigma`** (the default) standardizes the input with its mean and standard
deviation, maps the band within `squash_n_sigma` standard deviations
linearly onto $\pm$`squash_core_range`, and saturates smoothly beyond it,
never reaching $\pm 1$. With the defaults (2 and 0.75):

| Standard deviations from the mean | Squashed value |
|---|---|
| 0 | 0 |
| 1 | 0.375 |
| 2 | 0.75 (edge of the linear band) |
| 3 | 0.936 |
| 4 | 0.973 |
| 5 | 0.985 |
| 10 | 0.997 |

The bulk of the data passes through undistorted, and outliers keep their
order without leaving $(-1, 1)$. The mean and standard deviation are those
of the layer's actual inputs, measured on the train split in one pass before
the layer trains (`Trainer.fit_layer_inputs`), separately for every input
of every candidate.

**`tanh`** needs no statistics, but it compresses the bulk: it is already at
0.762 one standard deviation out.

**`squash: false`** feeds the inputs unchanged, for inputs already inside
$[-1, 1]$.

On the CCPP power-plant data, switching the squash on leaves the mean
error within one standard deviation, cuts the spread across folds by a third
to a half, and makes the runs 1.2 to 2 times faster (CCPP README).

## Options

| Option | Default | Meaning |
|---|---|---|
| `degree` | 3 | highest polynomial degree per input |
| `dim` | 2 | inputs per neuron |
| `cross` | true | include the pairwise products $u_k u_l$ |
| `squash` | true | squash the inputs into $(-1, 1)$ |
| `squash_method` | `model.squash_method` (`sigma`) | `sigma` or `tanh` |
| `squash_n_sigma` | `model.squash_n_sigma` (2.0) | half-width of the linear band, in standard deviations |
| `squash_core_range` | `model.squash_core_range` (0.75) | squashed value at the band's edge, in $(0, 1)$ |
| `activation` | identity | output activation |

```yaml
model:
  ref_functions:
    - legendre:
        degree: 3
    - legendre:
        degree: 3
        dim: 4
```

On California housing, degree 4 or 5 lands within the seed noise of degree
3 (California housing README). On CCPP, a four-input family (`dim: 4`) next
to the pairs is the largest single gain among the Legendre configurations
(CCPP README).

## In logs and plots

The short name joins the basis and the degree, with the number of inputs
when it is not 2: `Legendre3`, `Chebyshev4`, `Legendre3x4`.

## The knobs

`model.ref_functions` with the options above, `model.squash_method`,
`model.squash_n_sigma` and `model.squash_core_range`. See
[Configuration keys](../../reference/config.md).

<small>Checked against TorchSONN 0.1.5.</small>
