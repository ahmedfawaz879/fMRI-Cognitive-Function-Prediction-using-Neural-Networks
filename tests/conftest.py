"""Shared pytest fixtures.

Synthetic manifest / NIfTI generation lives here (test-only code), not in
the package itself: production code (``fmri_cognition.data``) must never
fabricate subject data or ship a manifest of "fake" subjects.
"""
from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_cognition.data import TRACT_NAMES


def _make_synthetic_tract_nifti(path, rng, shape=(4, 4, 4), nonzero_frac=0.5):
    data = np.zeros(shape, dtype=np.float32)
    mask = rng.random(shape) < nonzero_frac
    n_nonzero = int(mask.sum())
    if n_nonzero:
        data[mask] = rng.uniform(0.1, 5.0, size=n_nonzero).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(path))


@pytest.fixture
def synthetic_manifest(tmp_path):
    """Build a small synthetic manifest CSV plus matching in-memory-generated
    NIfTI tract volumes on disk. Purely synthetic random data -- not derived
    from or resembling any real HCP subject -- used only to exercise the
    pipeline's plumbing (feature extraction, training loop, CV loop) without
    requiring real imaging data.
    """
    rng = np.random.default_rng(0)
    n_subjects = 12
    rows = []
    for i in range(n_subjects):
        subj_id = f"synthetic_{i:03d}"
        row = {"subject_id": subj_id, "label": float(rng.uniform(80, 120))}
        for tract in TRACT_NAMES:
            nifti_path = tmp_path / f"{subj_id}_{tract}.nii.gz"
            _make_synthetic_tract_nifti(nifti_path, rng)
            row[f"{tract}_path"] = str(nifti_path)
        rows.append(row)
    df = pd.DataFrame(rows)
    manifest_path = tmp_path / "synthetic_manifest.csv"
    df.to_csv(manifest_path, index=False)
    return manifest_path, df
