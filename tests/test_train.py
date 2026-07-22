"""End-to-end test of the training/CV orchestration on synthetic data.

This confirms the pipeline runs without error and produces the expected
artifacts -- it is NOT a performance/benchmark claim. Metric values
computed here are on small random synthetic tract volumes with random
labels and are not meaningful as accuracy numbers.
"""
import torch

from fmri_cognition.train import run_cross_validation, set_seed


def _small_config():
    return {
        "seed": 0,
        "tract_names": [
            "or_l",
            "or_r",
            "af_l",
            "af_r",
            "atr_l",
            "atr_r",
            "slf2_l",
            "slf2_r",
        ],
        "features_per_tract": 3,
        "model": {"hidden_dim": 8},
        "training": {"epochs": 5, "learning_rate": 0.01},
        "cv": {"n_splits": 3, "shuffle": True},
    }


def test_run_cross_validation_end_to_end(tmp_path, synthetic_manifest):
    _, df = synthetic_manifest
    output_dir = tmp_path / "results"

    metrics = run_cross_validation(df, _small_config(), str(output_dir))

    assert set(metrics.keys()) == {"r2", "mae"}
    assert isinstance(metrics["r2"], float)
    assert isinstance(metrics["mae"], float)
    assert (output_dir / "predictions_nn.csv").exists()
    assert (output_dir / "mlp_tract_model.pth").exists()
    assert (output_dir / "scaler_nn.joblib").exists()
    assert (output_dir / "prediction_scatter_nn.png").exists()


def test_set_seed_makes_model_init_reproducible():
    set_seed(123)
    a = torch.randn(3)
    set_seed(123)
    b = torch.randn(3)
    assert torch.equal(a, b)
