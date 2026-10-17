# Survivor selection

Once every candidate of a layer is fitted and scored by the
[criterion](criteria.md), selection decides which ones survive. Candidates
of all families compete together. A candidate whose criterion value is NaN
is dropped first. The survivors become the layer; every other candidate is
deleted.

## How many survive

`model.nbest_neurons` sets the number of survivors per layer, GMDH's
*freedom of choice* (see [GMDH](gmdh.md)). The survivors are the inputs of
the next layer and, in a model with an output head, the inputs of the head.
More survivors give later layers more to combine, at the cost of more
candidates in every later layer: with 16 survivors and 16 features, the next
layer has 32 inputs and 496 pairs.

## The five methods

`train.neuron_selection_method` chooses how the survivors are picked:

- **`plain`**: the `nbest_neurons` candidates with the lowest criterion
  values.
- **`greedy_orth`**: walks the candidates from the lowest criterion value up.
  For each one, it takes the candidate's outputs on the dev split, centred,
  and removes the parts that the survivors kept so far already explain (a
  Gram-Schmidt step). The candidate is kept only if at least
  `train.neuron_selection_orth_threshold` (0.05) of its original norm
  remains. The walk stops at `nbest_neurons` survivors, or earlier if the
  candidates run out.
- **`max_volume`**: greedy maximum volume selection on the candidates'
  outputs. At each step it keeps the candidate whose remaining output, after
  removing the survivors' directions, has the largest norm. It ignores the
  criterion values and the targets when choosing.
- **`omp`**: orthogonal matching pursuit against the dev targets. At each
  step it keeps the candidate whose remaining output explains the most of
  the target that the survivors have not explained yet, then removes that
  candidate's direction from the target and from the other candidates. It
  ignores the criterion values. After the first survivor, it stops before
  `nbest_neurons` once the best candidate explains less than
  `train.neuron_selection_min_gain` (0.001) of the target's centered energy,
  so a layer can keep fewer survivors. A multi-class target is one-hot
  encoded, one column per class.
- **`omp_mixed`** (default): `omp` and `max_volume` in turn, over the same
  orthogonalized outputs. `omp` fills the first slots, all but
  `train.neuron_selection_reserve_frac` (0.2) of `nbest_neurons`, and stops
  early under the same `train.neuron_selection_min_gain` rule. `max_volume`
  fills the rest from what the chosen survivors leave unexplained, without
  looking at the target, so that directions which may only help a later
  layer are kept. Both phases skip a candidate when less than
  `train.neuron_selection_orth_threshold` of its original norm remains. A
  `reserve_frac` of 0.0 makes it pure `omp`, 1.0 pure `max_volume`, both
  with that guard. It ignores the criterion values.

`greedy_orth`, `max_volume`, `omp` and `omp_mixed` use the neuron's own
output, before any multi-class projection. In every method, the layer's error is computed from
the criterion values of the survivors actually kept.

## Why decorrelate

`plain` keeps the best candidates, which are often near-copies of each
other: the same strong pair of inputs, fitted by slightly different neurons.
A linear head over such survivors has little to combine, since they carry
much the same signal. `greedy_orth` keeps survivors that differ from each
other while still preferring good ones, giving the head more distinct inputs
to work with. `omp` goes further and picks each survivor for what it adds to
the prediction of the target given the survivors already kept. `omp_mixed`
keeps a share of the slots for survivors that are new rather than
predictive.

## Diagnostics

With `train.log_layer_diagnostics: true`, the trainer logs for every layer
how correlated the survivors' outputs are on the dev split (the median and
largest absolute correlation between two survivors), their top 10 singular
values, and their effective rank, the number of singular values above 1% of
the largest. It costs one extra pass over the dev split per layer.

## The knobs

`model.nbest_neurons`, `train.neuron_selection_method`,
`train.neuron_selection_orth_threshold`, `train.neuron_selection_min_gain`,
`train.neuron_selection_reserve_frac`, `train.log_layer_diagnostics`. See
[Configuration keys](../reference/config.md).

<small>Checked against TorchSONN 0.1.5.</small>
