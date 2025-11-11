# fMRI Cognitive Function Prediction using Neural Networks

## Overview

This repository provides a pipeline to predict cognitive function from fMRI-derived white matter tract features using a feed-forward neural network (MLP). The approach extracts summary statistics (mean, max, non-zero voxel count) from major tracts and predicts cognitive scores across subjects.

The method is suitable for small feature matrices (~24 features per subject), making a compact MLP sufficient for predictions.

## Repository Structure

```
.
├── predict_cognition_nn.py     # Main Python script for feature extraction, model training, and evaluation
├── tract_manifest.csv          # Manifest file containing subject IDs and paths to tract NIfTI files
├── results_nn/                # Output folder for trained models, predictions, and figures
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

* numpy
* pandas
* nibabel
* matplotlib
* scikit-learn
* torch
* joblib

## Usage

1. **Prepare the manifest file** (`tract_manifest.csv`) containing:

   * `subject_id` column
   * `label` column (cognitive score)
   * Paths to tract NIfTI files for each tract (columns: `or_l_path`, `or_r_path`, `af_l_path`, `af_r_path`, `atr_l_path`, `atr_r_path`, `slf2_l_path`, `slf2_r_path`)

2. **Run the script**:

```bash
python predict_cognition_nn.py
```

This script will:

* Extract tract-level features
* Train an MLP using 5-fold cross-validation
* Save predictions, trained model, scaler, and scatter plot visualization

## Neural Network Architecture

* **Input:** 24 features (8 tracts × 3 features per tract)
* **Hidden layers:** 64 → 32 neurons with ReLU activation
* **Output:** 1 (predicted cognitive score)
* **Optimizer:** Adam
* **Loss function:** MSE
* **Cross-validation:** 5-fold

## Outputs

Files saved in `results_nn/`:

* `mlp_tract_model.pth` – Trained PyTorch model
* `scaler_nn.joblib` – Feature standardizer
* `predictions_nn.csv` – True vs predicted cognitive scores
* `prediction_scatter_nn.png` – Scatter plot visualization

## Notes

* This pipeline is optimized for small feature sets (~24 features per subject).
* For larger voxel-level or hundreds of tracts, consider GPU acceleration and batch training.
* Scatter plots display the relationship between predicted and true scores, with the red dashed line indicating perfect prediction.

## References

* Glasser MF, et al. (2016). A multi-modal parcellation of human cerebral cortex. *Nature*.
* Human Connectome Project (HCP) datasets for structural and functional connectivity.
