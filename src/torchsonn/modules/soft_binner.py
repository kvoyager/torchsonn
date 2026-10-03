import torch
from torch import nn


class SoftBinner(nn.Module):
    """Soft one-hot encoding of a scalar prediction over `n_bins` fixed centers.

    Produces logits (pre-softmax) — the SONN multi-class loss path applies
    log_softmax over the bin axis. Used by SONN when
    `param.model.soft_binner=True` as an alternative to a learned linear head.
    """

    def __init__(self, n_bins: int = 10, scale: float = 100.0) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.centers = nn.Parameter(torch.linspace(0.05, 0.95, n_bins), requires_grad=False)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Turn a scalar prediction into logits over the bin centers.

        Parameters
        ----------
        x : (B,) or (B, T) tensor
            Prediction(s) on the scale of the centers (0.05 to 0.95).

        Returns
        -------
        logits : (B, n_bins) or (B, n_bins, T) tensor
            -scale*(x - center)^2, highest at the nearest center.
        """
        # if x.dim() == 1:
        x = x.unsqueeze(-1)  # [B, 1] or [B, 1, T]

        # Compute squared distance to bin centers
        dists = (x - self.centers)**2  # [B, T, ..., n_bins]
        if dists.dim() > 2:
            dists = dists.transpose(1, -1)  # [B, n_bins, T, ...]
        logits = -self.scale * dists
        return logits