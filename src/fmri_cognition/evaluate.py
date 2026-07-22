"""Evaluation metrics for the fMRI cognition regression pipeline.

Every value returned by this module is computed directly from the
``y_true``/``y_pred`` arrays passed in by the caller -- nothing here is
hardcoded or simulated. If this function has not been called on real
model output, no metric value exists.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R^2 and MAE for predicted vs. true cognitive scores.

    Parameters
    ----------
    y_true, y_pred:
        Equal-length arrays of ground-truth and predicted values produced
        by an actual model run (e.g. the out-of-fold predictions from
        :func:`fmri_cognition.train.run_cross_validation`).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} vs {y_pred.shape}"
        )
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    logger.info("Computed metrics: R^2=%.3f, MAE=%.3f", r2, mae)
    return {"r2": r2, "mae": mae}
