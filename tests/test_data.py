"""Tests for fmri_cognition.data -- run against synthetic NIfTI volumes only."""
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_cognition.data import (
    FEATURES_PER_TRACT,
    TRACT_NAMES,
    build_feature_matrix,
    extract_tract_features,
    load_manifest,
)


def test_extract_tract_features_on_synthetic_nifti(tmp_path):
    data = np.zeros((3, 3, 3), dtype=np.float32)
    data[0, 0, 0] = 2.0
    data[1, 1, 1] = 4.0
    # remaining voxels are zero and must be excluded from the summary stats
    path = tmp_path / "tract.nii.gz"
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(path))

    mean, max_, count = extract_tract_features(str(path))

    assert count == 2
    assert max_ == 4.0
    assert mean == 3.0  # (2 + 4) / 2


def test_extract_tract_features_all_zero_returns_zeros(tmp_path):
    data = np.zeros((2, 2, 2), dtype=np.float32)
    path = tmp_path / "empty_tract.nii.gz"
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(path))

    feats = extract_tract_features(str(path))

    assert feats == [0.0, 0.0, 0]


def test_build_feature_matrix_shape(synthetic_manifest):
    _, df = synthetic_manifest
    X = build_feature_matrix(df)
    assert X.shape == (len(df), len(TRACT_NAMES) * FEATURES_PER_TRACT)


def test_load_manifest_round_trip(synthetic_manifest):
    manifest_path, df = synthetic_manifest
    loaded = load_manifest(str(manifest_path))
    assert list(loaded["subject_id"]) == list(df["subject_id"])


def test_load_manifest_rejects_missing_tract_columns(tmp_path):
    bad = pd.DataFrame({"subject_id": ["a"], "label": [1.0]})
    path = tmp_path / "bad_manifest.csv"
    bad.to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_manifest(str(path))
