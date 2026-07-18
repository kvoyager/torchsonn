import torch
import torch.nn as nn


class NormMSE(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super(NormMSE, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Adding a small epsilon (1e-8) prevents division by zero if y_true is all zeros
        return torch.sum((y_true - y_pred) ** 2) / (torch.sum(y_true ** 2) + self.eps)