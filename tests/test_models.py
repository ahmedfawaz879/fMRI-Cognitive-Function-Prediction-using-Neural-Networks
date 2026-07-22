"""Tests for fmri_cognition.models (MLP forward pass)."""
import torch

from fmri_cognition.models import MLPRegressor


def test_mlp_forward_pass_shape():
    model = MLPRegressor(input_dim=24, hidden_dim=64)
    x = torch.randn(5, 24)
    out = model(x)
    assert out.shape == (5, 1)


def test_mlp_forward_pass_single_sample():
    model = MLPRegressor(input_dim=24, hidden_dim=64)
    x = torch.randn(1, 24)
    out = model(x)
    assert out.shape == (1, 1)


def test_mlp_forward_pass_small_hidden_dim():
    # hidden_dim // 2 must still be a valid positive layer width
    model = MLPRegressor(input_dim=8, hidden_dim=4)
    x = torch.randn(3, 8)
    out = model(x)
    assert out.shape == (3, 1)
