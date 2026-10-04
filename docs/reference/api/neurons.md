# Neurons

Each entry of `model.ref_functions` builds one of these classes in every
layer; a module holds all of a family's candidates as rows of its
tensors. See [Neuron families](../../concepts/neurons/index.md) for the
formulas and [Neuron family options](../config.md#neuron-family-options)
for the options.

::: torchsonn.neurons.base.BasePolynomNeuron

::: torchsonn.neurons.base.BaseTupleNeuron

::: torchsonn.neurons.binary.LinearPolynomNeuron

::: torchsonn.neurons.binary.LinearCovPolynomNeuron

::: torchsonn.neurons.binary.QuadraticPolynomNeuron

::: torchsonn.neurons.binary.CubicPolynomNeuron

::: torchsonn.neurons.poly.PolyQuadratic

::: torchsonn.neurons.orthopoly.BaseOrthogonalNeuron

::: torchsonn.neurons.orthopoly.LegendrePolynomNeuron

::: torchsonn.neurons.orthopoly.ChebyshevPolynomNeuron

::: torchsonn.neurons.rbf.RBFNeuron

::: torchsonn.neurons.base.generate_unique_pairs

::: torchsonn.neurons.base.generate_unique_combinations
