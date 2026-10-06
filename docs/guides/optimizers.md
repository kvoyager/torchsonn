# Optimizers

Each family's candidates are fitted together by a *batched* optimizer: every
candidate's parameters are one row of a batched tensor, and the optimizer
updates every row independently, with its own learning rate and history.
`train.optimizer.name` chooses it, and `train.optimizer.optimizer_params`
sets its arguments ([Optimizer parameters](../reference/config.md#optimizer-parameters)).

The head fit and the end-to-end pass use PyTorch's own optimizers instead;
see [Head fit and per-layer fine-tune](../reference/config.md#head-fit-and-per-layer-fine-tune)
and [End-to-end pass](../reference/config.md#end-to-end-pass).

## Choosing one

| Name | Method | Use it for |
|---|---|---|
| `lbfgs` | limited-memory quasi-Newton | the default choice for every family, and the tutorials' choice; the candidates are small least-squares problems it solves in tens to hundreds of steps |
| `adam` | Adam | a first-order alternative; needs more steps and a tuned learning rate |
| `sgd` | SGD with Nesterov momentum | the same, with plain momentum |

## The learning rate and the early stop

Three keys of `optimizer_params` belong to the trainer rather than the
optimizer:

- `lr`: every candidate's starting learning rate (1e-4 by default; the
  tutorials use 0.1 with LBFGS).
- `gamma`: when a candidate stops improving, its learning rate is multiplied
  by this (0.5).
- `min_lr`: once a candidate's learning rate is down to this and it still
  does not improve, it stops (1e-5 by default; the tutorials use 0.01 with
  LBFGS).

[Splits and stopping](../concepts/splits-and-stopping.md#stopping-a-candidates-fit)
describes the full rule.

## Gradient clipping

`adam`, `sgd` and `lbfgs` clip every gradient before using it: first each
entry to ±`clip_value` (1.0), then each candidate's whole gradient to a norm
of at most `clip_norm` (5.0). Set either to `null` to turn it off.

## LBFGS

LBFGS estimates each candidate's curvature from its last `history_size`
pairs of parameter and gradient changes (10 by default) and steps along the
resulting quasi-Newton direction, scaled by the learning rate. It needs
gradients from the same rows on every step, so give it one batch per split
(see [Data](data.md#batch-size)).

Two safeguards keep it stable:

- **`max_step`** (1.0) caps the norm of each candidate's update of each
  parameter tensor.
- **`curvature_eps`** (1e-8) keeps a pair only when its curvature is
  clearly positive ($y \cdot s > \varepsilon \lVert s \rVert \lVert y \rVert$),
  so the curvature estimate stays positive definite.

`null` turns either off. How often they act depends on the model. On
well-conditioned power-basis fits they rarely bind, capping only a small
fraction of updates and rejecting few curvature pairs. They matter most for
the RBF family, where a bump that loses its data has a gradient and a
curvature that vanish together: there they reject a much larger share of the
pairs, and with unbounded centres many of the survivors' centres can drift
far from the data, which the safeguards prevent. The log reports both after
every family's fit:

```text
LinearCovPolynomNeuron fit: 81 optimizer steps; 1.5% of updates capped at max_step, 0.0% of curvature pairs rejected
```

The history and the update run as batched tensor operations over all
candidates at once, so a family with many candidates costs little more than
one with few.

## Adam and SGD

`adam` takes `betas` (0.9, 0.999) and `eps` (1e-8). `sgd` takes `momentum`
(0.9), `nesterov` (true) and `weight_decay` (0.0). Both converge more slowly
than LBFGS on these small problems, so they need more steps and a learning
rate tuned to the data.

<small>Checked against TorchSONN 0.1.5.</small>
