"""MLP architecture for tract-feature -> cognitive score regression.

Architecture is unchanged from the original monolithic script.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """Feed-forward regressor: ``input_dim -> hidden_dim -> hidden_dim//2 -> 1``."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
