"""Tract NIfTI loading and per-tract summary-statistic feature extraction.

Expected manifest schema (CSV)
-------------------------------
The manifest passed to the CLI's ``--manifest`` argument must be a CSV
with the following columns. This mirrors the schema the original
``predict_cognition_nn.py`` script assumed (via ``row.get(f"{tract}_path")``
and ``df["label"]``) but never documented anywhere:

* ``subject_id``    -- unique subject identifier
* ``label``          -- the cognitive score to predict (numeric)
* ``<tract>_path``   -- one column per tract in :data:`TRACT_NAMES`, each
  holding a filesystem path to that subject's tract NIfTI volume:
  ``or_l_path``, ``or_r_path``, ``af_l_path``, ``af_r_path``,
  ``atr_l_path``, ``atr_r_path``, ``slf2_l_path``, ``slf2_r_path``.

No manifest and no HCP imaging data are bundled with this repository.
See README.md for how to obtain HCP data under its Data Use Agreement
and build a manifest that follows this schema. Tests in ``tests/``
exercise this module against small synthetic NIfTI volumes only.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRACT_NAMES: List[str] = [
    "or_l",
    "or_r",
    "af_l",
    "af_r",
    "atr_l",
    "atr_r",
    "slf2_l",
    "slf2_r",
]
FEATURES_PER_TRACT = 3  # mean, max, non-zero voxel count

REQUIRED_MANIFEST_COLUMNS: List[str] = ["subject_id", "label"] + [
    f"{tract}_path" for tract in TRACT_NAMES
]


def load_manifest(manifest_csv: str) -> pd.DataFrame:
    """Load and validate the tract manifest CSV.

    Raises ``ValueError`` listing any missing required column(s) so
    schema problems surface immediately, rather than mid-way through
    feature extraction.
    """
    df = pd.read_csv(manifest_csv)
    missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Manifest {manifest_csv!r} is missing required column(s): {missing}. "
            f"Expected columns: {REQUIRED_MANIFEST_COLUMNS}"
        )
    logger.info("Loaded manifest %s with %d subjects", manifest_csv, len(df))
    return df


def extract_tract_features(nifti_path: str) -> List[float]:
    """Extract (mean, max, non-zero voxel count) from one tract NIfTI volume.

    Only strictly positive voxel values are included in the summary
    statistics (matching the original script's ``vox[vox > 0]`` filter).
    If a tract volume has no positive voxels, all three features are 0.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    vox = data.flatten()
    vox_nonzero = vox[vox > 0]
    if vox_nonzero.size == 0:
        return [0.0, 0.0, 0]
    return [float(vox_nonzero.mean()), float(vox_nonzero.max()), int(vox_nonzero.size)]


def build_feature_matrix(
    df: pd.DataFrame, tract_names: Sequence[str] = TRACT_NAMES
) -> np.ndarray:
    """Build the (n_subjects, n_tracts * FEATURES_PER_TRACT) feature matrix
    by extracting per-tract summary statistics for every subject/row.
    """
    features_list = []
    for _, row in df.iterrows():
        subj_feats: List[float] = []
        for tract in tract_names:
            path = row.get(f"{tract}_path")
            if path is None:
                raise ValueError(f"Missing column {tract}_path in manifest")
            subj_feats.extend(extract_tract_features(path))
        features_list.append(subj_feats)
    return np.vstack(features_list)
