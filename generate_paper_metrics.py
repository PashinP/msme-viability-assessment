"""
Generate all missing metrics for the research paper.
Outputs: confusion matrix, ROC-AUC, global SHAP importance,
counterfactual feasibility stats, and inference latency.
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    accuracy_score
)
import shap

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
LABEL_NAMES = {0: "Critical", 1: "At-Risk", 2: "Stable", 3: "Growing", 4: "Thriving"}
FEATURE_NAMES = [
    "Term", "NoEmp", "NewExist", "CreateJob", "RetainedJob",
    "DisbursementGross", "UrbanRural", "RevLineCr", "LowDoc",
    "SBA_Appv", "GrAppv"
]
FEATURES_TO_VARY = ["Term", "DisbursementGross", "SBA_Appv", "GrAppv",
                     "CreateJob", "RetainedJob"]

print("=" * 70)
print("  PAPER METRICS GENERATOR")
print("=" * 70)

# ── Step 1: Load data and reproduce the exact preprocessing pipeline ──
print("\n[1/7] Loading dataset...")
df = pd.read_csv("SBAnational.csv", encoding="latin1")

# Clean currency columns
for col in ["DisbursementGross", "SBA_Appv", "GrAppv"]:
    if df[col].dtype == object:
        df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(float)

# Select features + target
features_plus_target = FEATURE_NAMES + ["MIS_Status"]
df_model = df[features_plus_target].dropna()

# Clean MIS_Status
df_model = df_model[df_model["MIS_Status"].isin(["P I F", "CHGOFF"])]
df_model["Target"] = (df_model["MIS_Status"] == "P I F").astype(int)

# Encode categoricals same way as notebook
for col in ["NewExist", "UrbanRural"]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))

for col in ["RevLineCr", "LowDoc"]:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0).astype(int)
    df_model[col] = df_model[col].clip(0, 1)

# ── Step 2: Build health score ──
print("[2/7] Building health scores...")
outcome_score = df_model["Target"] * 40
term_score = (df_model["Term"].clip(upper=240) / 240 * 20)
job_score = ((df_model["CreateJob"] + df_model["RetainedJob"]).clip(upper=50) / 50 * 20)
loan_score = (df_model["GrAppv"].clip(upper=500000) / 500000 * 20)
health = outcome_score + term_score + job_score + loan_score

df_model["health_label"] = pd.cut(
    health, bins=[-0.01, 25, 40, 60, 75, 100.01],
    labels=[0, 1, 2, 3, 4]
).astype(int)

print(f"   Class distribution:")
for c in range(5):
    cnt = (df_model["health_label"] == c).sum()
    print(f"   {LABEL_NAMES[c]:10s}: {cnt:>8,d} ({cnt/len(df_model)*100:.2f}%)")

# ── Step 3: Split and scale ──
print("[3/7] Splitting and scaling...")
from sklearn.model_selection import train_test_split

X = df_model[FEATURE_NAMES].values
y = df_model["health_label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_mc.pkl"))
X_test_scaled = scaler.transform(X_test)
X_train_scaled = scaler.transform(X_train)

print(f"   Train: {len(X_train):,d}  |  Test: {len(X_test):,d}")

# ── Step 4: Load models and generate confusion matrices + ROC-AUC ──
print("[4/7] Loading models and computing metrics...")
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_mc.pkl"))
lgbm_model = joblib.load(os.path.join(MODELS_DIR, "lgbm_mc.pkl"))

results = {}
for name, model in [("XGBoost", xgb_model), ("LightGBM", lgbm_model)]:
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # One-vs-Rest ROC-AUC
    try:
        auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception as e:
        auc_macro = auc_weighted = f"Error: {e}"
    
    results[name] = {
        "accuracy": acc,
        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    
    print(f"\n   {name}:")
    print(f"   Accuracy:     {acc:.4f}")
    print(f"   AUC (macro):  {auc_macro:.4f}" if isinstance(auc_macro, float) else f"   AUC: {auc_macro}")
    print(f"   AUC (weighted): {auc_weighted:.4f}" if isinstance(auc_weighted, float) else "")
    print(f"   Confusion Matrix:")
    class_labels = [LABEL_NAMES[i] for i in range(5)]
    print(f"   {'':>10s}  " + "  ".join(f"{c:>8s}" for c in class_labels))
    for i in range(5):
        print(f"   {class_labels[i]:>10s}  " + "  ".join(f"{cm[i,j]:>8d}" for j in range(5)))

# ── Step 5: Naive baseline ──
print("\n[5/7] Computing baselines...")
# Majority class baseline
majority_class = np.argmax(np.bincount(y_test))
naive_acc = (y_test == majority_class).mean()
print(f"   Naive majority-class baseline (always predict {LABEL_NAMES[majority_class]}): {naive_acc:.4f} ({naive_acc*100:.2f}%)")

# Binary logistic regression baseline on same data
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"   Logistic Regression (5-class): {lr_acc:.4f} ({lr_acc*100:.2f}%)")

# ── Step 6: Global SHAP analysis ──
print("\n[6/7] Computing global SHAP values (1000 sample)...")
np.random.seed(42)
shap_idx = np.random.choice(len(X_test_scaled), size=1000, replace=False)
X_shap = X_test_scaled[shap_idx]

explainer = shap.TreeExplainer(xgb_model)
sv_raw = explainer.shap_values(X_shap)

# Handle different SHAP output formats
if isinstance(sv_raw, list):
    # List of arrays, one per class. Stack to (n_classes, n_samples, n_features)
    stacked = np.array(sv_raw)
    # Mean absolute across all classes and samples
    global_importance = np.abs(stacked).mean(axis=(0, 1))
elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
    # (n_samples, n_features, n_classes)
    global_importance = np.abs(sv_raw).mean(axis=(0, 2))
else:
    global_importance = np.abs(sv_raw).mean(axis=0)

sorted_idx = np.argsort(global_importance)[::-1]
print(f"\n   Global Feature Importance (mean |SHAP| across 1000 samples, all classes):")
print(f"   {'Feature':>20s}  {'Mean |SHAP|':>12s}")
print(f"   {'─'*20}  {'─'*12}")
for i in sorted_idx:
    print(f"   {FEATURE_NAMES[i]:>20s}  {global_importance[i]:>12.4f}")

# ── Step 7: Counterfactual feasibility evaluation ──
print("\n[7/7] Evaluating counterfactual engine on 200 Critical/At-Risk samples...")

perturbations = {
    "Term": [12, 24, 36, 48, 60, 84, 120, 180, 240],
    "CreateJob": list(range(1, 25)),
    "RetainedJob": list(range(1, 25)),
    "DisbursementGross": [0.9, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
    "SBA_Appv": [0.9, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
    "GrAppv": [0.9, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
}

def run_counterfactual(app_dict, target_class):
    """Simplified counterfactual matching engine.py logic."""
    X = scaler.transform(np.array([[app_dict[f] for f in FEATURE_NAMES]]))
    current_class = int(xgb_model.predict(X)[0])
    
    if current_class >= target_class:
        return {"feasible": True, "changes": 0, "phase": 0}
    
    # Phase 1: single feature
    best_changes = None
    for feat in FEATURES_TO_VARY:
        for delta in perturbations[feat]:
            candidate = app_dict.copy()
            if feat in ["Term", "CreateJob", "RetainedJob"]:
                candidate[feat] = app_dict[feat] + delta
            else:
                candidate[feat] = app_dict[feat] * delta
            X_cf = scaler.transform(np.array([[candidate[f] for f in FEATURE_NAMES]]))
            pred = int(xgb_model.predict(X_cf)[0])
            if pred >= target_class:
                return {"feasible": True, "changes": 1, "phase": 1, "feature": feat}
    
    # Phase 2: pairwise
    for i, f1 in enumerate(FEATURES_TO_VARY):
        for f2 in FEATURES_TO_VARY[i+1:]:
            for d1 in perturbations[f1][:5]:
                for d2 in perturbations[f2][:5]:
                    candidate = app_dict.copy()
                    for feat, delta in [(f1, d1), (f2, d2)]:
                        if feat in ["Term", "CreateJob", "RetainedJob"]:
                            candidate[feat] = app_dict[feat] + delta
                        else:
                            candidate[feat] = app_dict[feat] * delta
                    X_cf = scaler.transform(np.array([[candidate[f] for f in FEATURE_NAMES]]))
                    pred = int(xgb_model.predict(X_cf)[0])
                    if pred >= target_class:
                        return {"feasible": True, "changes": 2, "phase": 2}
    
    return {"feasible": False, "changes": 0, "phase": -1}

# Select samples from Critical and At-Risk classes
critical_mask = y_test == 0
atrisk_mask = y_test == 1

np.random.seed(42)
n_sample = 100
critical_idx = np.random.choice(np.where(critical_mask)[0], size=min(n_sample, critical_mask.sum()), replace=False)
atrisk_idx = np.random.choice(np.where(atrisk_mask)[0], size=min(n_sample, atrisk_mask.sum()), replace=False)

cf_stats = {"Critical→At-Risk": [], "At-Risk→Stable": []}
feature_freq = {}

# Measure inference latency
latencies = []

print("   Running Critical → At-Risk counterfactuals...")
for idx in critical_idx:
    app = {f: float(X_test[idx, j]) for j, f in enumerate(FEATURE_NAMES)}
    start = time.perf_counter()
    result = run_counterfactual(app, target_class=1)
    elapsed = time.perf_counter() - start
    latencies.append(elapsed)
    cf_stats["Critical→At-Risk"].append(result)
    if result.get("feature"):
        feature_freq[result["feature"]] = feature_freq.get(result["feature"], 0) + 1

print("   Running At-Risk → Stable counterfactuals...")
for idx in atrisk_idx:
    app = {f: float(X_test[idx, j]) for j, f in enumerate(FEATURE_NAMES)}
    start = time.perf_counter()
    result = run_counterfactual(app, target_class=2)
    elapsed = time.perf_counter() - start
    latencies.append(elapsed)
    cf_stats["At-Risk→Stable"].append(result)
    if result.get("feature"):
        feature_freq[result["feature"]] = feature_freq.get(result["feature"], 0) + 1

# Also measure pure prediction latency
print("   Measuring prediction latency (1000 calls)...")
pred_latencies = []
for i in range(1000):
    x_single = X_test_scaled[i:i+1]
    start = time.perf_counter()
    _ = xgb_model.predict(x_single)
    _ = xgb_model.predict_proba(x_single)
    elapsed = time.perf_counter() - start
    pred_latencies.append(elapsed)

print("\n" + "=" * 70)
print("  COUNTERFACTUAL EVALUATION RESULTS")
print("=" * 70)

for transition, stats in cf_stats.items():
    total = len(stats)
    feasible = sum(1 for s in stats if s["feasible"])
    phase1 = sum(1 for s in stats if s.get("phase") == 1)
    phase2 = sum(1 for s in stats if s.get("phase") == 2)
    avg_changes = np.mean([s["changes"] for s in stats if s["feasible"]]) if feasible else 0
    
    print(f"\n   {transition} (n={total}):")
    print(f"   Feasibility rate:       {feasible}/{total} ({feasible/total*100:.1f}%)")
    print(f"   Phase 1 success:        {phase1}/{total} ({phase1/total*100:.1f}%)")
    print(f"   Phase 2 success:        {phase2}/{total} ({phase2/total*100:.1f}%)")
    print(f"   Avg features changed:   {avg_changes:.2f}")

print(f"\n   Most frequently recommended features:")
for feat, count in sorted(feature_freq.items(), key=lambda x: -x[1]):
    print(f"   {feat:>20s}: {count} times")

print(f"\n   Latency Statistics:")
print(f"   Prediction (mean):           {np.mean(pred_latencies)*1000:.2f} ms")
print(f"   Prediction (p95):            {np.percentile(pred_latencies, 95)*1000:.2f} ms")
print(f"   Counterfactual (mean):       {np.mean(latencies)*1000:.2f} ms")
print(f"   Counterfactual (p95):        {np.percentile(latencies, 95)*1000:.2f} ms")

print("\n" + "=" * 70)
print("  ALL METRICS GENERATED SUCCESSFULLY")
print("=" * 70)
