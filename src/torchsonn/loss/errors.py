import torch
import torch.nn.functional as F

LN2 = 0.6931471805599453   # ln(2), the maximum of JS in nats


def regularity_error(y_hat_B: torch.Tensor,
                     y_B:     torch.Tensor,
                     eps:     float = 1e-12) -> torch.Tensor:
    """
    Classical GMDH regularity criterion (regression) — squared error
    on subset B, normalized by the squared L2 norm of the targets on B.

        Δ²(B) = Σ_i (y_i - ŷ_i)² / Σ_i y_i²

    ŷ comes from a candidate whose coefficients were fit on subset A
    and evaluated on subset B. Lower is better; Δ² → 0 for a perfect
    fit, Δ² ≈ 1 for predictions no better than the zero predictor,
    Δ² > 1 means the model is worse than zero. If your targets aren't
    centered around zero, consider subtracting the mean first, or
    normalize by Σ_i (y_i - ȳ)² instead to get a proper R²-like
    "fraction of variance unexplained."

    Parameters
    ----------
    y_hat_B : (..., N) tensor
        Predictions on subset B from each candidate. Leading dims
        carry the candidate batch.
    y_B : (N,) tensor
        Ground-truth targets on subset B.
    eps : float
        Floor on the denominator, guarding against degenerate targets
        (all zeros).

    Returns
    -------
    delta_sq : (...) tensor
        Regularity error per candidate. Lower is better.
    """
    num   = ((y_hat_B - y_B) ** 2).sum(dim=-1)
    denom = (y_B ** 2).sum().clamp_min(eps)
    return num / denom


def bias_error(y_hat_A: torch.Tensor,
               y_hat_B: torch.Tensor,
               y:       torch.Tensor,
               eps:     float = 1e-12) -> torch.Tensor:
    """
    Classical GMDH bias criterion (regression) — squared difference
    between predictions from the candidate fit on A and on B, evaluated
    on a common set (typically A ∪ B), normalized by squared L2 norm
    of the targets on that set:

        n_B = Σ_i (ŷ_iA - ŷ_iB)² / Σ_i y_i²

    Measures parameter stability: small n_B means the model's
    coefficients are insensitive to which half of the data they were
    fit on (= captures real structure); large n_B means the model is
    fitting noise.

    Equivalent in ranking (and cheaper) for linear-in-coefficients
    models: n_B = ||a_A - a_B||² up to a constant scale.

    Sample alignment: y_hat_A[..., i], y_hat_B[..., i], and y[i] all
    refer to the same row of input. Both forward passes must be on a
    fixed evaluation set in the same order — never via a shuffled
    dataloader.

    Parameters
    ----------
    y_hat_A, y_hat_B : (..., N) tensors
        Predictions on the common evaluation set from candidate
        coefficients fit on subset A and B respectively. Leading
        dims carry the candidate batch.
    y : (N,) tensor
        Ground-truth targets on the common evaluation set, used only
        for the denominator's normalization.
    eps : float
        Floor on the denominator.

    Returns
    -------
    n_B : (...) tensor
        Bias error per candidate. Lower is better; identical fits
        give 0.
    """
    num   = ((y_hat_A - y_hat_B) ** 2).sum(dim=-1)
    denom = (y ** 2).sum().clamp_min(eps)
    return num / denom


def regularity_error_ce(logits_B: torch.Tensor,
                     y_B: torch.Tensor,
                     eps: float = 1e-12) -> torch.Tensor:
    """
    CE-GMDH regularity criterion — Normalized Cross-Entropy on subset B.

        NCE = mean_i [ -log p_model(y_i | x_i) ] / H(Y_B)

    H(Y_B) is the Shannon entropy of the empirical class frequencies on B,
    which equals the cross-entropy of the constant-marginal predictor and
    serves as the principled "no-skill" baseline. NCE → 0 for a perfect
    model; NCE ≈ 1 means no better than predicting marginal frequencies;
    NCE > 1 means actively worse than the trivial classifier.

    Parameters
    ----------
    logits_B : (..., N, K) float tensor
        Projection-layer outputs (W · z) on subset B for each candidate.
        Leading dims are treated as candidate-batch dimensions.
    y_B : (N,) long tensor
        Integer class labels on B. For one-hot labels, pass y_B.argmax(-1).
    eps : float
        Floor for H(Y_B), guarding against absent classes on near-degenerate
        validation splits.

    Returns
    -------
    nce : (...) tensor
        Regularity error per candidate.  Lower is better.
    """
    N, K = logits_B.shape[-2], logits_B.shape[-1]

    # Convert raw logits to log-probabilities along the class axis.
    # F.log_softmax is numerically stable (subtracts max internally),
    # avoiding the overflow that plain log(softmax(...)) would produce
    # when logits are large. Shape unchanged: (..., N, K).
    log_p = F.log_softmax(logits_B, dim=-1)

    # Reshape y_B from (N,) so it can index log_p's last (class) axis
    # via torch.gather, broadcasting across any leading candidate dims.
    #
    # .view(*([1]*(log_p.dim()-2)), N, 1):
    #     Insert size-1 dimensions for every leading candidate axis,
    #     then append a trailing size-1 axis for gather.
    #     Example: if log_p is (C1, C2, N, K), y_B (N,) → (1, 1, N, 1).
    #
    # .expand(*log_p.shape[:-1], 1):
    #     Broadcast the size-1 leading dims to match log_p's leading
    #     shape, e.g. (1, 1, N, 1) → (C1, C2, N, 1). This is a view
    #     (zero memory copy), not an actual replication.
    y_idx = y_B.view(*([1] * (log_p.dim() - 2)), N, 1) \
                .expand(*log_p.shape[:-1], 1)

    # Per-sample cross-entropy, averaged over N to give per-candidate CE.
    #
    # .gather(-1, y_idx):
    #     For each sample i, picks log_p[..., i, y_B[i]] — the log-
    #     probability the model assigns to the true class. Output
    #     shape (..., N, 1).
    # .squeeze(-1):
    #     Drops the trailing size-1 class axis → (..., N).
    # -... :
    #     Negate so it's NLL (cross-entropy per sample), not log-prob.
    # .mean(dim=-1):
    #     Average over the N validation samples → (...) per candidate.
    ce_model = -log_p.gather(-1, y_idx).squeeze(-1).mean(dim=-1)

    # Empirical class frequencies on B: count occurrences and normalize.
    # minlength=K guarantees a length-K vector even if some classes are
    # absent from B (their entries become 0). .float() because bincount
    # returns int64 and we need float arithmetic below.
    counts = torch.bincount(y_B, minlength=K).float()
    p_marg = counts / counts.sum()

    # Shannon entropy of the marginal class distribution, in nats.
    #
    # torch.special.entr(p) computes -p * log(p) elementwise, with the
    # convention 0 * log(0) := 0 baked in — so absent classes contribute
    # 0 cleanly, no NaN risk, no need for an eps inside the log.
    # .sum() aggregates over the K classes.
    # .clamp_min(eps) is a seatbelt: prevents division-by-zero in the
    # final NCE on pathological splits where a single class swallows
    # all samples and H_Y is exactly 0.
    H_Y = torch.special.entr(p_marg).sum().clamp_min(eps)

    # NCE: ratio of model CE to baseline (marginal-frequency predictor)
    # CE. The baseline is the constant classifier that emits p_marg for
    # every input; its expected CE equals H(Y_B). Lower NCE is better.
    return ce_model / H_Y


def bias_error_l2(z_A: torch.Tensor,
                  z_B: torch.Tensor,
                  y:   torch.Tensor = None) -> torch.Tensor:
    """
    CE-GMDH bias criterion — L2 disagreement on scalar neuron outputs,
    normalized by label energy. Direct port of Ivakhnenko's formula:

        n_B = Σ_i (z_A(x_i) - z_B(x_i))^2  /  Σ_i ||y_i||^2

    For one-hot labels, Σ_i ||y_i||^2 = N regardless of N, so the
    default (y = None) uses N as the denominator — the structurally
    correct value for standard classification encoding. Pass y
    explicitly for ordinal encodings or soft/weighted labels.

    Sample alignment: z_A[..., i] and z_B[..., i] MUST refer to the
    same input x_i.

    Parameters
    ----------
    z_A, z_B : (..., N) tensors
        Scalar neuron outputs on the common evaluation set, from the
        same neuron structure with parameters fit on subset A vs B.
        Leading dims carry the candidate batch.
    y : tensor, optional
        Labels for explicit denominator. (N,) integers for ordinal
        encoding, or (N, K) one-hot / soft labels. If None, denominator
        defaults to N — the one-hot case.

    Returns
    -------
    bias : (...) tensor
        Lower is better; identical fits give 0.
    """
    sq_diff = ((z_A - z_B) ** 2).sum(dim=-1)
    if y is None:
        denom = float(z_A.shape[-1])                    # one-hot: ||y_i||² = 1
    else:
        denom = (y.float() ** 2).sum().clamp_min(1e-12)
    return sq_diff / denom


def bias_error_js(logits_A: torch.Tensor,
                  logits_B: torch.Tensor) -> torch.Tensor:
    """
    CE-GMDH bias criterion — Normalized Jensen–Shannon divergence in
    predicted-probability space.

        n_B = Σ_i JS(p_A(x_i) || p_B(x_i))  /  (N · ln 2)

    The denominator is the analog of Ivakhnenko's Σ_i ||y_i||²: the
    maximum total disagreement attainable on N samples, which for JS is
    N · ln 2 (since JS is bounded by ln 2 per sample regardless of K).
    Result is in [0, 1]: 0 = identical predictions, 1 = disjoint-support
    disagreement on every sample.

    Use when you want bounded, portable thresholds across layers / K /
    datasets, or when behavioral agreement matters more than parameter
    stability. JS underweights minority-class disagreements on
    imbalanced data — prefer bias_error_l2 there.

    Sample alignment: logits_A[..., i, :] and logits_B[..., i, :] MUST
    refer to the same input x_i.

    Parameters
    ----------
    logits_A, logits_B : (..., N, K) tensors of identical shape
        Projection-layer outputs (W · z) from the two candidate fits on
        the common evaluation set. Leading dims carry the candidate
        batch.

    Returns
    -------
    bias : (...) tensor in [0, 1]
    """
    N = logits_A.shape[-2]

    log_pA = F.log_softmax(logits_A, dim=-1)
    log_pB = F.log_softmax(logits_B, dim=-1)

    # log M where M = (pA + pB)/2, computed stably via logsumexp.
    log_M = torch.logsumexp(
        torch.stack([log_pA, log_pB], dim=0), dim=0
    ) - LN2                                                       # (..., N, K)

    pA, pB = log_pA.exp(), log_pB.exp()
    kl_AM  = (pA * (log_pA - log_M)).sum(dim=-1)                  # (..., N)
    kl_BM  = (pB * (log_pB - log_M)).sum(dim=-1)
    js_sum = (0.5 * (kl_AM + kl_BM)).sum(dim=-1)                  # (...,)

    return js_sum / (N * LN2)