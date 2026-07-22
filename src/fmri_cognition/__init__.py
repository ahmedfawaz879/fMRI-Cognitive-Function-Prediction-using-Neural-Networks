"""fmri_cognition: tract-feature MLP pipeline for predicting cognitive
function from fMRI/dMRI-derived white matter tract summary statistics.

This package is a reorganization of the original single-file
predict_cognition_nn.py script into importable, testable modules:

* ``data``     -- manifest loading and per-tract feature extraction
* ``models``   -- MLP architecture
* ``train``    -- seeding, training loop, and 5-fold CV orchestration
* ``evaluate`` -- R^2 / MAE computation
* ``cli``      -- command-line entrypoint

No results shipped with this package are computed from real HCP data --
see the repository README for dataset access and current status.
"""

__version__ = "0.1.0"
