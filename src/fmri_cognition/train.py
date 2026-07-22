"""Training loop and 5-fold CV orchestration for the tract-feature MLP."""
from __future__ import annotations

import logging
import os
import random
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .data import build_feature_matrix
from .evaluate import compute_metrics
from .models import MLPRegressor

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python's ``random``, NumPy, and PyTorch (CPU + CUDA) RNGs.

    The original script only passed ``random_state=42`` to the KFold
    splitter; model weight initialization and any other stochastic
    operation were left unseeded, so two runs over identical data could
    still produce different models. This seeds every RNG the pipeline
    touches so a given seed reproduces a given run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    log_every: int = 50,
) -> nn.Module:
    """Full-batch gradient descent training loop.

    This matches the original script's real behavior: it defined a
    ``BATCH_SIZE`` constant but never used it to mini-batch the data --
    every "epoch" was a single full-batch forward/backward pass over the
    whole training fold. That is preserved here rather than introducing
    real mini-batching, which would be new modeling behavior.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % log_every == 0:
            logger.info("Epoch %d/%d, Loss: %.4f", epoch + 1, epochs, loss.item())
    return model


def _save_scatter_plot(
    y_true: np.ndarray, y_pred: np.ndarray, r2: float, output_dir: str, show: bool = False
) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe backend; avoids blocking automated/CI runs
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title(f"Tract-based NN Prediction (R²={r2:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_scatter_nn.png"))
    if show:
        plt.show()
    plt.close()


def run_cross_validation(
    df: pd.DataFrame,
    config: Dict,
    output_dir: str,
    show_plot: bool = False,
) -> Dict[str, float]:
    """Run the full pipeline: feature extraction, standardization,
    5-fold CV, a final model fit on all data, and artifact saving.

    Returns the metrics dict computed from the actual out-of-fold
    predictions produced during cross-validation (never hardcoded).
    """
    os.makedirs(output_dir, exist_ok=True)
    seed = config["seed"]
    set_seed(seed)
    device = get_device()
    logger.info("Using device: %s", device)

    tract_names = config["tract_names"]
    hidden_dim = config["model"]["hidden_dim"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    n_splits = config["cv"]["n_splits"]
    shuffle = config["cv"].get("shuffle", True)

    input_dim = len(tract_names) * config["features_per_tract"]

    X = build_feature_matrix(df, tract_names)
    y = df["label"].values.astype(float).reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    y_true_all, y_pred_all = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor)):
        logger.info("Fold %d/%d", fold + 1, n_splits)
        X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
        y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

        model = MLPRegressor(input_dim, hidden_dim).to(device)
        model = train_model(model, X_train, y_train, epochs, lr)

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).cpu().numpy()
            y_true = y_test.cpu().numpy()
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

    y_true_all = np.vstack(y_true_all).flatten()
    y_pred_all = np.vstack(y_pred_all).flatten()

    metrics = compute_metrics(y_true_all, y_pred_all)

    pred_df = pd.DataFrame(
        {
            "subject_id": df["subject_id"],
            "y_true": y_true_all,
            "y_pred": y_pred_all,
        }
    )
    pred_csv = os.path.join(output_dir, "predictions_nn.csv")
    pred_df.to_csv(pred_csv, index=False)
    logger.info("Saved predictions CSV: %s", pred_csv)

    # Final model trained on the full dataset (as in the original script)
    final_model = MLPRegressor(input_dim, hidden_dim).to(device)
    final_model = train_model(final_model, X_tensor, y_tensor, epochs, lr)
    torch.save(final_model.state_dict(), os.path.join(output_dir, "mlp_tract_model.pth"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler_nn.joblib"))

    _save_scatter_plot(y_true_all, y_pred_all, metrics["r2"], output_dir, show=show_plot)

    return metrics
