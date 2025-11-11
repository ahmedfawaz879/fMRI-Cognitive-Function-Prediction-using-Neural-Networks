import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

# ---------------- CONFIG ----------------
MANIFEST_CSV = "tract_manifest.csv"
OUTPUT_DIR = "results_nn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRACT_NAMES = ["or_l", "or_r", "af_l", "af_r", "atr_l", "atr_r", "slf2_l", "slf2_r"]
FEATURES_PER_TRACT = 3  # mean, max, non-zero voxel count
INPUT_DIM = len(TRACT_NAMES) * FEATURES_PER_TRACT
HIDDEN_DIM = 64
EPOCHS = 200
BATCH_SIZE = 32
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- HELPER FUNCTIONS ----------------
def extract_tract_features(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    vox = data.flatten()
    vox_nonzero = vox[vox > 0]
    if vox_nonzero.size == 0:
        return [0.0, 0.0, 0]
    return [vox_nonzero.mean(), vox_nonzero.max(), vox_nonzero.size]

def build_feature_matrix(df):
    features_list = []
    for idx, row in df.iterrows():
        subj_feats = []
        for tract in TRACT_NAMES:
            path = row.get(f"{tract}_path")
            if path is None:
                raise ValueError(f"Missing column {tract}_path in manifest")
            feats = extract_tract_features(path)
            subj_feats.extend(feats)
        features_list.append(subj_feats)
    return np.vstack(features_list)

# ---------------- NEURAL NETWORK ----------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------- MAIN ----------------
def main():
    # Load manifest
    df = pd.read_csv(MANIFEST_CSV)
    X = build_feature_matrix(df)
    y = df["label"].values.astype(float).reshape(-1,1)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    # K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor)):
        print(f"Fold {fold+1}")

        X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
        y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

        model = MLPRegressor(INPUT_DIM, HIDDEN_DIM).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).cpu().numpy()
            y_true = y_test.cpu().numpy()
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

    y_true_all = np.vstack(y_true_all).flatten()
    y_pred_all = np.vstack(y_pred_all).flatten()

    # Metrics
    r2 = r2_score(y_true_all, y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    print(f"\nFinal Metrics: R² = {r2:.3f}, MAE = {mae:.3f}")

    # Save predictions
    pred_df = pd.DataFrame({
        "subject_id": df["subject_id"],
        "y_true": y_true_all,
        "y_pred": y_pred_all
    })
    pred_csv = os.path.join(OUTPUT_DIR, "predictions_nn.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"Saved predictions CSV: {pred_csv}")

    # Save scaler & model (final model on full data)
    final_model = MLPRegressor(INPUT_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(final_model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    # train on all data
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = final_model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    torch.save(final_model.state_dict(), os.path.join(OUTPUT_DIR, "mlp_tract_model.pth"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler_nn.joblib"))

    # Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_true_all, y_pred_all, alpha=0.6)
    plt.plot([y_true_all.min(), y_true_all.max()],
             [y_true_all.min(), y_true_all.max()], 'r--')
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title(f"Tract-based NN Prediction (R²={r2:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_scatter_nn.png"))
    plt.show()

if __name__ == "__main__":
    main()
