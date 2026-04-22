"""
Build Similarity Index — One-time script
=========================================
Processes the 899K SBA dataset into:
  1. data/sba_features.pkl  — compact feature matrix (11 cols) + outcome
  2. data/sba_knn.pkl       — fitted NearestNeighbors model

Run once:  python scripts/build_similarity_index.py
"""
import os, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Paths
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT, "SBAnational.csv")
DATA_DIR = os.path.join(PROJECT, "data")
MODELS_DIR = os.path.join(PROJECT, "models")

FEATURE_COLS = ["Term", "NoEmp", "NewExist", "CreateJob", "RetainedJob",
                "DisbursementGross", "UrbanRural", "RevLineCr", "LowDoc",
                "SBA_Appv", "GrAppv"]


def clean_currency(val):
    """Convert '$60,000.00 ' to float."""
    if isinstance(val, str):
        return float(val.replace("$", "").replace(",", "").strip())
    return float(val)


def main():
    print(f"[1/5] Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"       Loaded {len(df):,} records, {df.shape[1]} columns")

    # Clean currency columns
    print("[2/5] Cleaning currency columns...")
    for col in ["DisbursementGross", "SBA_Appv", "GrAppv"]:
        df[col] = df[col].apply(clean_currency)

    # Encode binary columns
    df["RevLineCr"] = df["RevLineCr"].map({"Y": 1, "N": 0, "T": 1, "0": 0}).fillna(0).astype(int)
    df["LowDoc"] = df["LowDoc"].map({"Y": 1, "N": 0, "S": 0, "C": 0, "A": 0, "R": 0}).fillna(0).astype(int)
    df["NewExist"] = pd.to_numeric(df["NewExist"], errors="coerce").fillna(1).astype(int)
    df["UrbanRural"] = pd.to_numeric(df["UrbanRural"], errors="coerce").fillna(0).astype(int)

    # Parse outcome
    print("[3/5] Parsing outcomes...")
    df["MIS_Status"] = df["MIS_Status"].str.strip()
    df = df[df["MIS_Status"].isin(["P I F", "CHGOFF"])].copy()
    df["Outcome"] = (df["MIS_Status"] == "P I F").astype(int)  # 1 = success, 0 = default

    # Extract features
    print("[4/5] Extracting features and fitting KNN...")
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    features = df[FEATURE_COLS].values.astype(np.float32)
    outcomes = df["Outcome"].values.astype(np.int8)

    # Also keep business name, state, and industry for context
    meta_cols = ["Name", "State", "NAICS", "MIS_Status"]
    meta = df[meta_cols].reset_index(drop=True)

    # Scale features for KNN
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features).astype(np.float32)

    # Fit KNN
    knn = NearestNeighbors(n_neighbors=50, metric="euclidean", algorithm="ball_tree",
                           n_jobs=-1)
    knn.fit(features_scaled)

    # Save
    print("[5/5] Saving to data/...")
    os.makedirs(DATA_DIR, exist_ok=True)

    save_data = {
        "features": features,          # original unscaled
        "features_scaled": features_scaled,
        "outcomes": outcomes,
        "meta": meta,
        "feature_cols": FEATURE_COLS,
    }

    joblib.dump(save_data, os.path.join(DATA_DIR, "sba_features.pkl"), compress=3)
    joblib.dump(knn, os.path.join(DATA_DIR, "sba_knn.pkl"), compress=3)
    joblib.dump(scaler, os.path.join(DATA_DIR, "sba_knn_scaler.pkl"), compress=3)

    # Stats
    total = len(outcomes)
    success = int(outcomes.sum())
    default = total - success
    feat_size = os.path.getsize(os.path.join(DATA_DIR, "sba_features.pkl")) / 1e6
    knn_size = os.path.getsize(os.path.join(DATA_DIR, "sba_knn.pkl")) / 1e6

    print(f"\n✅ Done!")
    print(f"   Records:  {total:,} ({success:,} success, {default:,} default)")
    print(f"   Features: {feat_size:.1f} MB")
    print(f"   KNN:      {knn_size:.1f} MB")


if __name__ == "__main__":
    main()
