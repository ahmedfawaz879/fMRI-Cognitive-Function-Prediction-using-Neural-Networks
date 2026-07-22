# fMRI Cognitive Function Prediction using Neural Networks

Predicts a subject's cognitive score from per-tract summary statistics of
diffusion/structural MRI-derived white matter tract volumes, using a
compact MLP evaluated with 5-fold cross-validation.

## Why this matters

White matter tract microstructure (e.g. arcuate fasciculus, anterior
thalamic radiation, superior longitudinal fasciculus) is an established
structural correlate of cognitive function. A lightweight, reproducible
pipeline that maps tract-level summary features to a cognitive score is
relevant groundwork for research on neurodevelopmental trajectories (e.g.
tracking white matter maturation against cognitive milestones) and
neurodegenerative assessment (e.g. flagging tract-level changes that
precede or accompany cognitive decline). This repository is an
**implementation of that pipeline**, not a validated clinical or research
result — see [Results](#results) and [Limitations](#limitations) below.

## Dataset: Human Connectome Project (HCP)

This pipeline is designed around tract-level features derived from
**Human Connectome Project (HCP)** diffusion/structural MRI data (e.g. the
HCP Young Adult release). **HCP data is not bundled with this repository
and cannot be redistributed.** Access requires registering for and
accepting the HCP Data Use Agreement:

* https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms

You must obtain tract volumes (e.g. from a tractography/parcellation
pipeline such as TractSeg or a similar tool) and a cognitive outcome
measure for each subject yourself, then build a manifest file in the
schema below. Any other tract-level dMRI dataset providing the same
columns can be substituted, since nothing in the code is HCP-specific.

### Manifest schema

`--manifest` must point to a CSV with these columns (enforced by
[`fmri_cognition.data.load_manifest`](src/fmri_cognition/data.py), which
validates them before any feature extraction runs):

| Column | Meaning |
|---|---|
| `subject_id` | Unique subject identifier |
| `label` | Cognitive score to predict (numeric) |
| `or_l_path`, `or_r_path` | Path to left/right optic radiation NIfTI volume |
| `af_l_path`, `af_r_path` | Path to left/right arcuate fasciculus NIfTI volume |
| `atr_l_path`, `atr_r_path` | Path to left/right anterior thalamic radiation NIfTI volume |
| `slf2_l_path`, `slf2_r_path` | Path to left/right superior longitudinal fasciculus II NIfTI volume |

No manifest is committed to this repository. `tests/conftest.py` contains
a fixture that generates a small **synthetic** manifest and matching
synthetic NIfTI volumes (random voxel intensities, random labels) purely
to exercise the code path in tests — it does not represent, and should
never be mistaken for, real subject data.

## Method

1. **Feature extraction** (`fmri_cognition/data.py`): for each of the 8
   tracts above, load the NIfTI volume and compute 3 summary statistics
   over strictly-positive voxels — mean, max, and non-zero voxel count —
   giving a 24-dimensional feature vector per subject.
2. **Standardization**: features are z-scored with `sklearn.StandardScaler`.
3. **Model** (`fmri_cognition/models.py`): a feed-forward MLP,
   `24 -> 64 -> 32 -> 1`, ReLU activations, trained with Adam + MSE loss.
4. **Evaluation** (`fmri_cognition/train.py`, `fmri_cognition/evaluate.py`):
   5-fold cross-validation (`sklearn.model_selection.KFold`,
   `shuffle=True`); R² and MAE are computed once on the pooled
   out-of-fold predictions across all 5 folds. A final model is then
   refit on all subjects and saved for later inference.

## Results

**Implementation only; not yet evaluated on benchmark data.** This
repository has never been run against real HCP subjects. No R², MAE, or
correlation figure anywhere in this repo (including in tests) should be
read as a performance claim — the only numbers produced so far come from
unit/integration tests on small synthetic, randomly generated feature
arrays, which exist purely to confirm the code runs correctly end to end,
not to demonstrate predictive accuracy.

## Limitations

* **No real-data evaluation.** The pipeline has not been run on real HCP
  (or any other) subject data; there is no benchmark result to compare
  against.
* **No baseline comparison.** The MLP has not been compared against
  simpler alternatives (e.g. ridge/lasso regression, gradient-boosted
  trees) or other architectures on the same features, so it is unknown
  whether the added model capacity is warranted for a ~24-dimensional
  feature vector.
* **No external cohort validation.** Even once evaluated on HCP data,
  generalization to other scanners, acquisition protocols, or
  populations would remain unverified.
* **Coarse feature set.** Three summary statistics per tract (mean, max,
  non-zero voxel count) discard most of the spatial and microstructural
  information in the tract volume (e.g. along-tract profiles, FA/MD
  distributions, tract shape) and may not capture clinically relevant
  microstructural variation.
* **Small-sample instability.** With only ~24 features, the model's
  effective train-set size per fold and its sensitivity to CV-fold
  composition have not been characterized.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

Dependencies are exactly pinned in [`requirements.txt`](requirements.txt)
(runtime) and [`requirements-dev.txt`](requirements-dev.txt) (adds
`pytest`).

## Reproduce (synthetic fixtures)

Since no real manifest/data is bundled, "reproduce" here means: install,
then run the test suite, which exercises the entire pipeline (feature
extraction -> MLP -> 5-fold CV -> metrics -> artifact saving) against
synthetic data generated on the fly.

```bash
pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
pytest tests/ -v
python main.py --help
```

To run the CLI itself against your own real manifest once you have HCP
(or equivalent) data and a manifest matching the schema above:

```bash
python main.py --manifest tract_manifest.csv --output-dir results_nn
```

`--manifest` (default `tract_manifest.csv`) and `--output-dir` (default
`results_nn`) match the paths the original single-script version of this
repo hardcoded; they're now overridable rather than fixed. See
`python main.py --help` for all options, including `--config` (default
`configs/default.yaml`), `--seed`, and `--show-plot` (the original script
always called `plt.show()`; that's now opt-in so headless/automated runs
don't block on a GUI window).

## Configuration

Hyperparameters live in [`configs/default.yaml`](configs/default.yaml)
rather than as hardcoded constants: seed, tract list, hidden layer width,
epochs, learning rate, and number of CV folds.

## Reproducibility

`fmri_cognition.train.set_seed()` seeds Python's `random`, NumPy, and
PyTorch (CPU + CUDA) before model construction and training, in addition
to the `random_state` already used for the KFold split. In the original
script only the KFold split was seeded, so model weight initialization
was not reproducible between runs; that gap is closed here.

## Repository structure

```
.
├── main.py                       # CLI entrypoint (python main.py --help)
├── configs/
│   └── default.yaml              # hyperparameters (seed, model, training, CV)
├── src/fmri_cognition/
│   ├── data.py                   # manifest loading + per-tract feature extraction
│   ├── models.py                 # MLPRegressor architecture
│   ├── train.py                  # seeding, training loop, 5-fold CV orchestration
│   ├── evaluate.py               # R²/MAE computation
│   └── cli.py                    # argparse entrypoint
├── tests/                        # pytest suite against synthetic data only
├── requirements.txt               # exactly-pinned runtime dependencies
├── requirements-dev.txt           # + pytest
├── pyproject.toml                 # packaging (src layout, `pip install -e .`)
└── LICENSE
```

## Citation

If you use this code, please cite:

```bibtex
@software{fawaz2026fmricognition,
  author  = {Fawaz, Ahmed},
  title   = {fMRI Cognitive Function Prediction using Neural Networks},
  year    = {2026},
  url     = {https://github.com/ahmedfawaz879/fMRI-Cognitive-Function-Prediction-using-Neural-Networks}
}
```

If you use HCP data with this pipeline, please cite the HCP dataset per
its own terms, e.g.:

```bibtex
@article{vanessen2013hcp,
  author  = {Van Essen, David C. and Smith, Stephen M. and Barch, Deanna M.
             and Behrens, Timothy E. J. and Yacoub, Essa and Ugurbil, Kamil},
  title   = {The WU-Minn Human Connectome Project: An overview},
  journal = {NeuroImage},
  volume  = {80},
  pages   = {62--79},
  year    = {2013},
  doi     = {10.1016/j.neuroimage.2013.05.041}
}
```

## References

* Glasser MF, et al. (2016). A multi-modal parcellation of human cerebral
  cortex. *Nature*.
* Van Essen DC, et al. (2013). The WU-Minn Human Connectome Project: An
  overview. *NeuroImage*.

## License

MIT — see [LICENSE](LICENSE).
