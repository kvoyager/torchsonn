# Combined Cycle Power Plant (CCPP) regression

A TorchSONN tutorial that reproduces the canonical **UCI Combined Cycle Power
Plant** benchmark: predict a plant's net hourly electrical output `PE` (MW) from
four hourly-average ambient measurements, and evaluate with the **5×2
cross-validation** protocol that every published CCPP result is measured
against.

The plant runs a gas turbine and a steam turbine in tandem; four ambient
variables set the operating point, and the model predicts the net output:

| Feature | Meaning | Unit | Effect on output |
|---------|---------|------|------------------|
| `AT` | Ambient Temperature | °C | strong negative driver (r ≈ −0.95) |
| `V`  | Exhaust Vacuum | cm Hg | strong negative driver (r ≈ −0.87) |
| `AP` | Ambient Pressure | millibar | mild positive driver (r ≈ +0.52) |
| `RH` | Relative Humidity | % | mild positive driver (r ≈ +0.39) |
| `PE` | **Net electrical output** (target) | MW | range 420–496 MW |

The relationship is predominantly linear — a plain linear fit already reaches
R² ≈ 0.93 — so this is a clean demonstration of SONN *discovering* that
structure with a `linear_cov` neuron and then squeezing out the residual
curvature and interactions (the AT×V collinearity, humidity's
temperature-dependent effect) with `quadratic` and a full four-input `polyquad`
neuron. Because TorchSONN is itself a self-organizing **polynomial** network, it
lands squarely in the model class that has been shown to be state-of-the-art
competitive on this dataset (see [Comparison](#comparison-with-published-results)).

- **Dataset:** <https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant>
  (9568 samples, 4 features; the tutorial auto-downloads a byte-identical CSV
  mirror and caches it under `data/ccpp.csv`).
- **Entry point:** [`ccpp.py`](ccpp.py)
- **Configs:** [`ccpp.yaml`](ccpp.yaml) (light, default) · [`ccpp_heavy.yaml`](ccpp_heavy.yaml) (heavy) · [`ccpp_legendre.yaml`](ccpp_legendre.yaml) · [`ccpp_legendre_heavy.yaml`](ccpp_legendre_heavy.yaml) · [`ccpp_legendre_poly.yaml`](ccpp_legendre_poly.yaml) (Legendre-basis variants)

## Why 5×2 cross-validation?

A single train/test split yields one number whose third decimal is noise — a
different random split can swing R² by ±0.005. Tüfekci & Kaya (2014), the paper
every CCPP result is measured against, use **5×2 CV**: shuffle the data and cut
it in half, train on half A / test on half B, then swap (train B / test A) —
that is one "2-fold" repeat; do it for five independent shuffles → **5 × 2 = 10
train/test measurements**, and report the mean ± standard deviation. The spread
across the 10 tells you how stable the number actually is. The tutorial trains
the model from scratch on all 10 folds and reports mean ± std of RMSE / MAE / R².

## Running it

From the repo root:

```bash
# Default (light) config — small survivor pool / candidate cap, CPU.
python -m tutorials.ccpp.ccpp

# Heavy config — larger pools + cubic/poly5/poly10 neurons, CUDA.
python -m tutorials.ccpp.ccpp --config-name=ccpp_heavy
```

Every config key is overridable on the command line via Hydra:

```bash
python -m tutorials.ccpp.ccpp train.device=cpu
python -m tutorials.ccpp.ccpp model.ref_functions='[linear_cov]' train.max_layer_count=3
python -m tutorials.ccpp.ccpp -m train.ridge_alpha=0.001,0.01,0.05   # multirun sweep
```

Each run writes a self-contained folder `checkpoints/YYYY-MM-DD-HH-MM/` holding
one `train.log` plus a `fold_NN/` checkpoint subfolder per fold. After the CV
loop, the discovered network from the final fold is rendered to
[`ccpp_model.svg`](ccpp_model.svg) (full) and
[`ccpp_pruned_model.svg`](ccpp_pruned_model.svg) (pruned to the single
best-error path — a readable "discovered formula"). Rendering needs the `[viz]`
extra (`pip install "torchsonn[viz]"` + system Graphviz).

## The two configs

| Setting | `ccpp.yaml` (default) | `ccpp_heavy.yaml` |
|---|---|---|
| Reference functions | `linear_cov`, `quadratic`, `polyquad`(dim 4) | + `cubic`, `polyquad`(dim 5), `polyquad`(dim 10) |
| `nbest_neurons` (survivors/layer) | 8 | 120 |
| `max_neuron_models` (candidates/family) | 60 | 1200 |
| Device | CPU | CUDA |
| Wall-clock ([hardware](#hardware)) | ≈ 19 min | ≈ 11.3 h |

The light config is deliberately cheap so the full 10-fold run is tractable; the
heavy config is the larger search used while exploring the model. As the results
below show, the heavy config buys only ~0.08 MW of MAE for ~36× the compute.

### Hardware

The wall-clock figures above are hardware-dependent and quoted only to
illustrate the cost/accuracy trade-off. Both runs were measured on:

- **CPU** (default config, `device: cpu`) — 12th-gen Intel Core i9-12900
  (16 cores: 8 P + 8 E) → ≈ 19 min.
- **GPU** (heavy config, `device: cuda`) — NVIDIA GeForce RTX 5080 → ≈ 11.3 h.

## Results

Each block below is the final summary printed by `ccpp.py` (the `====` section
at the end of that run's `train.log`), reproduced verbatim.

### Default — `ccpp.yaml`
<sub>run `checkpoints/2026-07-22-19-59/train.log` — light config, CPU</sub>

```
==================================================================
5×2 cross-validation summary — 10 train/test measurements
==================================================================
repeat   fold   RMSE(MW)    MAE(MW)        R²
     1    A→B     4.2246     3.3675    0.9390
     1    B→A     4.3290     3.3655    0.9353
     2    A→B     4.2797     3.3309    0.9374
     2    B→A     4.2232     3.3129    0.9385
     3    A→B     4.2330     3.3369    0.9385
     3    B→A     4.2966     3.3621    0.9366
     4    A→B     4.2662     3.3432    0.9380
     4    B→A     4.2445     3.3497    0.9376
     5    A→B     4.1670     3.2897    0.9393
     5    B→A     4.3290     3.3643    0.9367
------------------------------------------------------------------
  RMSE = 4.2593 ± 0.0511 MW
  MAE  = 3.3423 ± 0.0256 MW
  R²   = 0.9377 ± 0.0012
```

### Heavy — `ccpp_heavy.yaml`
<sub>run `checkpoints/2026-07-22-23-11/train.log` — heavy config, CUDA</sub>

```
==================================================================
5×2 cross-validation summary — 10 train/test measurements
==================================================================
repeat   fold   RMSE(MW)    MAE(MW)        R²
     1    A→B     4.2135     3.3002    0.9394
     1    B→A     4.1894     3.2276    0.9394
     2    A→B     4.1846     3.2324    0.9401
     2    B→A     4.1349     3.2461    0.9411
     3    A→B     4.1307     3.2188    0.9414
     3    B→A     4.2051     3.2855    0.9393
     4    A→B     4.2126     3.2992    0.9396
     4    B→A     4.1656     3.2581    0.9399
     5    A→B     4.1070     3.2219    0.9411
     5    B→A     4.2713     3.3092    0.9384
------------------------------------------------------------------
  RMSE = 4.1815 ± 0.0486 MW
  MAE  = 3.2599 ± 0.0356 MW
  R²   = 0.9400 ± 0.0010
```

The `ccpp_legendre.yaml` and `ccpp_legendre_heavy.yaml` variants replace the
default's monomial neurons (`linear_cov` / `quadratic` / `polyquad`) with a
**Legendre orthogonal-polynomial** basis over each input pair — a
better-conditioned stand-in for the raw cubic terms, since the Legendre Gram
matrix stays well-behaved where the monomial one is a near-singular Hilbert
matrix. The light variant uses a single degree-3 Legendre family (light pools,
CPU); the heavy variant adds a degree-2 family alongside it and widens the
search (`nbest_neurons` 60, `max_neuron_models` 600, CUDA). Run them with
`python -m tutorials.ccpp.ccpp --config-name=ccpp_legendre` (or
`ccpp_legendre_heavy`).

A third variant, `ccpp_legendre_poly.yaml`, keeps the light config's degree-3
Legendre pair family and adds a **multi-input** degree-3 Legendre neuron over all
four inputs at once (`dim: 4`) — the orthogonal-basis counterpart of the
default's four-input `polyquad`. That neuron carries the univariate Legendre
curvature for each input plus one bilinear term per input pair in a single
19-coefficient (`1 + 4·3 + 6`) fit, letting a layer model the joint AT×V×AP×RH
interaction directly rather than only through pairwise Legendre combinations. Run
it with `python -m tutorials.ccpp.ccpp --config-name=ccpp_legendre_poly`.

### Legendre — `ccpp_legendre.yaml`
<sub>run `checkpoints/2026-07-23-20-58/train.log` — degree-3 Legendre basis, light pools, CPU</sub>

```
==================================================================
5×2 cross-validation summary — 10 train/test measurements
==================================================================
repeat   fold   RMSE(MW)    MAE(MW)        R²
     1    A→B     4.1734     3.2878    0.9405
     1    B→A     4.2851     3.3223    0.9366
     2    A→B     4.2502     3.2959    0.9382
     2    B→A     4.1802     3.2906    0.9398
     3    A→B     4.1783     3.2678    0.9401
     3    B→A     4.2769     3.3385    0.9372
     4    A→B     4.2720     3.3489    0.9379
     4    B→A     4.1874     3.2659    0.9393
     5    A→B     4.1890     3.3100    0.9387
     5    B→A     4.3167     3.3669    0.9371
------------------------------------------------------------------
  RMSE = 4.2309 ± 0.0545 MW
  MAE  = 3.3094 ± 0.0342 MW
  R²   = 0.9385 ± 0.0013
```

### Legendre (heavy) — `ccpp_legendre_heavy.yaml`
<sub>run `checkpoints/2026-07-23-21-19-2/train.log` — degree-3 + degree-2 Legendre basis, wide pools, CUDA</sub>

```
==================================================================
5×2 cross-validation summary — 10 train/test measurements
==================================================================
repeat   fold   RMSE(MW)    MAE(MW)        R²
     1    A→B     4.1867     3.2796    0.9401
     1    B→A     4.2825     3.3206    0.9367
     2    A→B     4.2411     3.2911    0.9385
     2    B→A     4.1682     3.2682    0.9401
     3    A→B     4.1863     3.2701    0.9398
     3    B→A     4.2332     3.2977    0.9385
     4    A→B     4.2169     3.2844    0.9395
     4    B→A     4.1723     3.2653    0.9397
     5    A→B     4.1483     3.2830    0.9399
     5    B→A     4.3168     3.3593    0.9371
------------------------------------------------------------------
  RMSE = 4.2152 ± 0.0538 MW
  MAE  = 3.2919 ± 0.0287 MW
  R²   = 0.9390 ± 0.0013
```

### Legendre (multi-input) — `ccpp_legendre_poly.yaml`
<sub>run `checkpoints/2026-07-24-12-59/train.log` — degree-3 pair + four-input (dim 4) degree-3 Legendre, light pools, CPU</sub>

```
==================================================================
5×2 cross-validation summary — 10 train/test measurements
==================================================================
repeat   fold   RMSE(MW)    MAE(MW)        R²
     1    A→B     4.1223     3.2328    0.9420
     1    B→A     4.2406     3.2893    0.9379
     2    A→B     4.2021     3.2462    0.9396
     2    B→A     4.1687     3.2746    0.9401
     3    A→B     4.1556     3.2446    0.9407
     3    B→A     4.2354     3.3139    0.9384
     4    A→B     4.1985     3.2696    0.9400
     4    B→A     4.1755     3.2654    0.9396
     5    A→B     4.1526     3.2708    0.9398
     5    B→A     4.2808     3.3196    0.9381
------------------------------------------------------------------
  RMSE = 4.1932 ± 0.0481 MW
  MAE  = 3.2727 ± 0.0285 MW
  R²   = 0.9396 ± 0.0012
```

On MAE, the Legendre light config (3.31 ± 0.03) slightly improves on the
monomial default (3.34 ± 0.03), and the Legendre heavy config (3.29 ± 0.03)
improves a little further. Adding the four-input Legendre neuron
(`ccpp_legendre_poly.yaml`, **3.27 ± 0.03**) is the strongest orthogonal-basis
variant: it edges past the heavy Legendre search while still on light pools /
CPU, and closes to within ~0.01 MW of the heavy monomial search (3.26 ± 0.04) —
effectively matching that result at a small fraction of its cost (~30 min CPU vs
11.3 h CUDA). The joint four-input interaction the `dim: 4` neuron captures is
doing real work the pairwise-only Legendre families leave on the table.

Every fold selects all four ambient variables (`features=AT, V, AP, RH`),
confirming each input carries signal.

## Comparison with published results

The canonical CCPP metric under 5×2 CV is **MAE**. On that apples-to-apples
protocol, TorchSONN's heavy config lands right in the published range:

| Study | Method | Protocol | MAE (MW) | RMSE (MW) |
|-------|--------|----------|:--------:|:---------:|
| Tüfekci (2014) | Bagging REP-tree | 5×2 CV | **3.22** | — |
| Torre et al. (2019) | data-driven PCE (polynomial) | 5×2 CV (same splits) | **3.11 ± 0.03** | — |
| **TorchSONN — heavy** (`ccpp_heavy.yaml`) | self-organizing polynomial net | 5×2 CV | **3.26 ± 0.04** | 4.18 ± 0.05 |
| **TorchSONN — Legendre poly** (`ccpp_legendre_poly.yaml`) | self-organizing polynomial net | 5×2 CV | **3.27 ± 0.03** | 4.19 ± 0.05 |
| **TorchSONN — default** (`ccpp.yaml`) | self-organizing polynomial net | 5×2 CV | **3.34 ± 0.03** | 4.26 ± 0.05 |
| Siddiqui et al. (2021) | GBRT (450 trees) | single 90/10 split | — | 2.58 † |

**Reading the table:**

- **TorchSONN (heavy) MAE 3.26 ± 0.04** matches the canonical Tüfekci (2014)
  tree-ensemble benchmark (3.22) to within one standard deviation of the CV
  spread, and comes within ~0.15 MW of Torre et al.'s polynomial-chaos result
  (3.11) — the current CV state of the art. The **default** config trails the
  heavy one by only ~0.08 MW while running in minutes on a CPU.
- **TorchSONN (Legendre poly) MAE 3.27 ± 0.03** reaches the heavy config's
  accuracy (3.26) within noise, but on light pools and CPU (~30 min) — a single
  four-input Legendre neuron (`dim: 4`) over the raw ambient variables recovers
  most of what the heavy monomial search buys with its far larger pools, a GPU,
  and ~36× the compute.
- **Torre et al.** run their polynomial method on Tüfekci's *exact* CV splits
  and beat the tree ensemble with lower variance — direct evidence that a
  polynomial model is SOTA-competitive on CCPP. TorchSONN, being a
  self-organizing polynomial network, lands in the same regime, a good sanity
  check that its discovered polynomial structure is doing real work.
- **† Siddiqui et al.'s RMSE 2.58 is not comparable.** It comes from a single
  90/10 train/test split with a 450-tree gradient-boosted ensemble — a
  different, more optimistic protocol than 5×2 CV, and reported as RMSE rather
  than MAE. It should not be read against the cross-validated numbers above.

## References

- **Tüfekci, P. (2014).** *Prediction of full load electrical power output of a
  base load operated combined cycle power plant using machine learning methods.*
  International Journal of Electrical Power & Energy Systems, 60, 126–140. — the
  canonical CCPP benchmark; establishes the 5×2 CV protocol and the raw 4-feature
  setup; best model (Bagging REP-tree) reaches MAE ≈ 3.22.
  <https://www.sciencedirect.com/science/article/abs/pii/S0142061514000908>
- **Torre, E., Marelli, S., Embrechts, P., & Sudret, B. (2019).** *Data-driven
  polynomial chaos expansion for machine learning regression.* Journal of
  Computational Physics, 388, 601–623. — runs a polynomial method on Tüfekci's
  exact CV splits: MAE 3.11, beating the tree ensemble with far lower variance.
  arXiv:1808.03216 · <https://arxiv.org/abs/1808.03216>
- **Siddiqui, R., et al. (2021).** *Power Prediction of Combined Cycle Power
  Plant (CCPP) Using Machine Learning Algorithm-Based Paradigm.* Wireless
  Communications and Mobile Computing (Wiley). — the RMSE 2.58 "SOTA" claim, but
  on a 90/10 single split with a 450-tree GBRT, not cross-validation; not
  comparable to CV numbers.
  <https://onlinelibrary.wiley.com/doi/10.1155/2021/9966395>
- **Dataset:** UCI Machine Learning Repository — Combined Cycle Power Plant.
  <https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant>
