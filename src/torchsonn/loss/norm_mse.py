import torch
import torch.nn as nn


class NormMSE(nn.Module):
    """Sum-of-squares error normalized by the spread of the targets.

        loss = Σ(y - ŷ)² / (n · scale)          when `scale` is set
        loss = Σ(y - ŷ)² / Σ(y - ȳ)²           centered, per call
        loss = Σ(y - ŷ)² / Σ y²                 centered=False (gmdhpy parity)

    `scale` is the per-sample target variance measured once over the whole
    training split (`Trainer._fit_target_scale`) and is the form the trainer
    uses. Fixing the denominator rather than recomputing it per batch keeps the
    loss batch-size-independent and safe on a degenerate batch — with the
    schema default `batch_size: 1` a per-batch centered denominator would be
    exactly zero on every batch — and puts the training loss on the same scale
    as the dev-set regularity criterion, which is what makes the absolute
    `early_stop_patience` / `divergence_threshold` thresholds mean the same
    thing across datasets.

    The per-call fallback (`scale=None`) applies when the loss is used outside a
    `Trainer.train`, e.g. a standalone `train_out_proj` after
    `load_model_checkpoint`.

    `scale` is a plain attribute rather than a registered buffer on purpose:
    `loss_fn` is a SONN submodule, so a buffer would add a `state_dict` key and
    invalidate existing checkpoints. It is a property of the data, not of the
    fitted model, and is re-measured on every `train()` including resume.

    `censor_at` handles a right-censored target (e.g. California housing
    prices recorded as 5.0 for every house worth 5.0 or more). A target at or
    above the cap only says "at least this much", so for those rows the
    prediction is clipped to the cap before the squared error: predicting
    above the cap costs nothing and pushes no gradient, predicting below it is
    penalized as usual. Uncensored rows are untouched. This is censored least
    squares - the Tobit likelihood with the noise scale taken to zero - and it
    keeps the objective a convex piecewise quadratic. Predictions should be
    clipped to the cap at inference time as well; the model is now free to
    predict above it. The cap is in the units the loss sees (log price when
    the target is log-transformed). None (default) disables it.
    """

    def __init__(self, eps: float = 1e-8, centered: bool = True,
                 scale: float | None = None, censor_at: float | None = None) -> None:
        super(NormMSE, self).__init__()
        self.eps = eps
        self.centered = centered
        self.scale = scale
        self.censor_at = censor_at

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Normalized squared error of `y_pred` against `y_true`.

        Parameters
        ----------
        y_pred : tensor
            Predictions. Clipped to `censor_at` on the censored rows when it is
            set.
        y_true : tensor
            Targets, same shape as `y_pred`.

        Returns
        -------
        loss : scalar tensor
            sum((y - y_hat)^2) divided by n*`scale` when `scale` is set,
            otherwise by the (centered) sum of squares of `y_true`; `eps` is
            added to the denominator.
        """
        if self.censor_at is not None:
            # torch.minimum's gradient is 1 below the cap and 0 above it, which
            # is exactly the one-sided penalty; no soft clip is needed.
            y_pred = torch.where(y_true >= self.censor_at,
                                 torch.minimum(y_pred, torch.full_like(y_pred, self.censor_at)),
                                 y_pred)
        num = torch.sum((y_true - y_pred) ** 2)
        if self.scale is not None:
            return num / (y_true.numel() * self.scale + self.eps)
        # Adding a small epsilon prevents division by zero on a degenerate
        # target (constant under `centered`, all-zero otherwise).
        spread = y_true - y_true.mean() if self.centered else y_true
        return num / (torch.sum(spread ** 2) + self.eps)