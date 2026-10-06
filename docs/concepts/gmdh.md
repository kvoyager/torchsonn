# GMDH

The Group Method of Data Handling (GMDH) is a family of inductive
algorithms for mathematical modelling, system identification and data
analysis. It builds a model from data by trying many small candidate models
and keeping those that predict well on data they were not fitted to,
discovering the structure automatically and growing it layer by layer.
A. G. Ivakhnenko introduced it in his 1968 paper [[1]](#ref-1). Its
application to complex data-mining was later expanded by Müller and Lemke
[[2]](#ref-2). TorchSONN implements its layered form on PyTorch,
as [GmdhPy](https://github.com/kvoyager/GmdhPy) does in the scikit-learn
style.

GMDH is also recognized as an early precursor of deep learning. In his
historical survey, J. Schmidhuber credits early GMDH networks as among the
first working learning algorithms for deep, multilayer feedforward
networks, noting that Ivakhnenko's group trained networks up to eight
layers deep as early as 1971 [[3]](#ref-3).

## Inductive modelling

Most modelling starts from a structure the researcher chooses: which inputs,
which terms, how many layers. Only the coefficients are learned. GMDH
learns the structure as well. The researcher supplies the data and a set of
simple building blocks, and the method grows a model by combining them,
keeping at each step only the combinations that hold up on fresh data. The
GMDH literature calls this inductive, or self-organizing, modelling
([[4]](#ref-4), [[5]](#ref-5)).

## The external criterion

A model's error on the rows it was fitted to keeps falling as the model
gets more complex: every added term can only fit those rows better. That
error cannot choose a structure, because it always prefers the most
complex one. GMDH chooses structure with an *external criterion*, computed
on data the coefficients did not see. The GMDH literature calls this the
principle of external complement and relates it to Gödel's incompleteness
theorem: no model can be singled out using only the data it was fitted to
([[5]](#ref-5), pp. 10–11).

The data is split in two: subset $A$ fits the coefficients of each
candidate, and subset $B$ judges it. Two classic criteria use the split
this way.

**The regularity criterion** fits a candidate on $A$ and measures its
error on $B$ ([[5]](#ref-5), eq. 1.39):

$$
\Delta^2(B) = \frac{\sum_{p \in B} \big(y_p - \hat{y}^{A}_p\big)^2}{\sum_{p \in B} y_p^2}
$$

where $\hat{y}^{A}_p$ is the prediction, for row $p$, of the candidate
fitted on $A$.

**The minimum-bias criterion** fits the same candidate twice, once on $A$
and once on $B$, and measures how much the two fits disagree on all rows
$W = A \cup B$ ([[5]](#ref-5), table 1.4):

$$
\eta^2_{bs} = \sum_{p \in W} \big(\hat{y}^{A}_p - \hat{y}^{B}_p\big)^2
$$

A candidate that captures real structure predicts nearly the same whichever
half it was fitted on; one that fits noise does not. The sources normalize
this criterion in different ways.

In TorchSONN, the regularity criterion fits on the train split and judges
on the dev split. The minimum-bias criterion fits on the even and on the
odd rows of the train split. Both divide by the spread of the targets
around their mean rather than by $\sum y^2$; [Criteria](criteria.md) gives
the exact formulas.

## The multilayered algorithm

GMDH has two main algorithms. The combinatorial algorithm (COMBI) works in
a single layer and tries combinations of the input arguments of one
polynomial ([[5]](#ref-5), p. 30). The multilayered iterative algorithm
(MIA), the one TorchSONN implements, builds the model in layers of partial
descriptions [[6]](#ref-6):

1. Every pair of inputs makes a candidate *partial description*, a
   low-degree polynomial of the two inputs. The classic one is quadratic
   (see [The Kolmogorov-Gabor polynomial](kolmogorov-gabor.md)).
2. Each candidate is fitted on $A$ and scored with the external criterion.
3. The $F$ best candidates pass their outputs to the next layer as its
   inputs. Keeping several survivors per layer rather than narrowing hard
   keeps options open for later layers, since a candidate that is not the
   strongest on its own may combine well downstream. The GMDH literature
   calls $F$ the *freedom of choice* ([[5]](#ref-5), p. 30).
4. Layers are added while the criterion improves, and the model is the one
   at the criterion's minimum.

## How TorchSONN maps onto GMDH

| GMDH | TorchSONN |
|---|---|
| fitting subset $A$ | train split |
| checking subset $B$ | dev split |
| partial description | candidate neuron of a family in `model.ref_functions` |
| freedom of choice $F$ | `model.nbest_neurons` |
| regularity criterion | `train.criterion_type: validate` |
| minimum-bias criterion | `train.criterion_type: bias` |
| layer of the multilayered algorithm | layer |
| stop at the criterion's minimum | the growth rule, keeping the layers up to the best one |

[The algorithm](algorithm.md) follows one layer through the code and lists
where TorchSONN departs from classical GMDH.

For a longer introduction to GMDH, see Farlow's article [[7]](#ref-7) and
the books edited by Farlow [[4]](#ref-4) and written by Madala and
Ivakhnenko [[5]](#ref-5).

## References

1. <a id="ref-1"></a>A. G. Ivakhnenko, "The group method of data handling,
   a rival of the method of stochastic approximation," *Soviet Automatic
   Control*, no. 3, pp. 43–55, 1968. English translation of *Avtomatika*,
   1968, no. 3.
2. <a id="ref-2"></a>J.-A. Müller and F. Lemke, *Self-Organising Data
   Mining: An Intelligent Analysis Tool for Finding Knowledge*. Hamburg:
   Libri / KnowledgeMiner Software, 2000. ISBN 3-89811-861-4.
3. <a id="ref-3"></a>J. Schmidhuber, "Deep learning in neural networks: an
   overview," *Neural Networks*, vol. 61, pp. 85–117, 2015.
   [doi:10.1016/j.neunet.2014.09.003](https://doi.org/10.1016/j.neunet.2014.09.003)
4. <a id="ref-4"></a>S. J. Farlow (ed.), *Self-Organizing Methods in
   Modeling: GMDH Type Algorithms*. New York: Marcel Dekker, 1984.
   ISBN 0-8247-7161-3.
5. <a id="ref-5"></a>H. R. Madala and A. G. Ivakhnenko, *Inductive Learning
   Algorithms for Complex Systems Modeling*. Boca Raton, FL: CRC Press, 1994.
   ISBN 0-8493-4438-7.
6. <a id="ref-6"></a>A. G. Ivakhnenko, "Polynomial theory of complex
   systems," *IEEE Transactions on Systems, Man, and Cybernetics*,
   vol. SMC-1, no. 4, pp. 364–378, 1971.
   [doi:10.1109/TSMC.1971.4308320](https://doi.org/10.1109/TSMC.1971.4308320)
7. <a id="ref-7"></a>S. J. Farlow, "The GMDH algorithm of Ivakhnenko,"
   *The American Statistician*, vol. 35, no. 4, pp. 210–215, 1981.
   [doi:10.1080/00031305.1981.10479358](https://doi.org/10.1080/00031305.1981.10479358)

<small>Checked against TorchSONN 0.1.5.</small>
