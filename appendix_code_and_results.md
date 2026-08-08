# Appendix

---

## A. Important Source Code Files

The following source code files represent the core implementation of the MSME Viability Assessment System. These files contain the prediction engine, SHAP explainability, counterfactual recommendations, conversational AI feature extraction, and risk analysis modules.

---

### A.1 Core Prediction & Counterfactual Engine

**File: `api/engine.py`**

This is the central ML module of the system. It loads the trained XGBoost and LightGBM models, exposes prediction, SHAP explanation, and DiCE-inspired counterfactual recommendation methods. All ML logic is concentrated here so the API layer stays thin.

```python
"""
Core prediction engine — loads models and provides prediction, SHAP, and
counterfactual services.  All ML logic is concentrated here, so the API
layer stays thin.
"""
import os, json
import numpy as np
import joblib
import shap

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

LABEL_NAMES = {0: "Critical", 1: "At-Risk", 2: "Stable", 3: "Growing", 4: "Thriving"}

FEATURE_LABELS = {
    "Term": "Loan Term (months)",
    "NoEmp": "Number of Employees",
    "NewExist": "Business Type",
    "CreateJob": "Jobs to Create",
    "RetainedJob": "Jobs Retained",
    "DisbursementGross": "Disbursement Amount ($)",
    "UrbanRural": "Location Type",
    "RevLineCr": "Revolving Line of Credit",
    "LowDoc": "Low Documentation Loan",
    "SBA_Appv": "SBA Guaranteed Amount ($)",
    "GrAppv": "Gross Approved Amount ($)",
}

FEATURES_TO_VARY = ["Term", "DisbursementGross", "SBA_Appv", "GrAppv",
                     "CreateJob", "RetainedJob"]


class PredictionEngine:
    """Loads models once and exposes predict / explain / recommend methods."""

    def __init__(self):
        with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
            self.meta = json.load(f)
        self.feature_names = self.meta["feature_names"]
        self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_mc.pkl"))

        self.models = {}
        for name, fname in [("XGBoost", "xgb_mc.pkl"), ("LightGBM", "lgbm_mc.pkl")]:
            path = os.path.join(MODELS_DIR, fname)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)

        self.primary_model_name = "XGBoost"
        self.primary_model = self.models[self.primary_model_name]
        self._shap_explainer = None

    @property
    def shap_explainer(self):
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.primary_model)
        return self._shap_explainer

    def _to_array(self, app_dict: dict) -> np.ndarray:
        row = np.array([[app_dict[f] for f in self.feature_names]])
        return self.scaler.transform(row)

    def predict(self, app_dict: dict, model_name: str | None = None):
        X = self._to_array(app_dict)
        model = self.models.get(model_name, self.primary_model)
        used = model_name or self.primary_model_name
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        return {
            "predicted_class": pred,
            "predicted_label": LABEL_NAMES[pred],
            "confidence": float(proba[pred]),
            "probabilities": {LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)},
            "model_used": used,
        }

    def explain(self, app_dict: dict):
        X = self._to_array(app_dict)
        pred = int(self.primary_model.predict(X)[0])
        sv_raw = self.shap_explainer.shap_values(X)
        if isinstance(sv_raw, list):
            sv = sv_raw[pred][0]
        elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
            sv = sv_raw[0, :, pred]
        else:
            sv = sv_raw[0]
        contributions = {self.feature_names[i]: float(sv[i])
                         for i in range(len(self.feature_names))}
        sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_pos = [f for f, v in sorted_feats if v > 0][:3]
        top_neg = [f for f, v in sorted_feats if v < 0][:3]
        return {
            "predicted_class": pred, "predicted_label": LABEL_NAMES[pred],
            "feature_contributions": contributions,
            "top_positive_features": top_pos, "top_negative_features": top_neg,
        }

    def recommend(self, app_dict: dict, target_class: int | None = None):
        X = self._to_array(app_dict)
        current_class = int(self.primary_model.predict(X)[0])
        if target_class is None:
            target_class = min(current_class + 1, 4)
        if current_class >= target_class:
            return {"current_class": current_class, "current_label": LABEL_NAMES[current_class],
                    "target_class": target_class, "target_label": LABEL_NAMES[target_class],
                    "feasible": True, "changes": []}

        best_cf, best_changes, best_dist = None, [], float("inf")
        np.random.seed(42)
        perturbations = {
            "Term": [12, 24, 36, 48, 60, 84, 120, 180, 240],
            "CreateJob": list(range(1, 25)),
            "RetainedJob": list(range(1, 25)),
            "DisbursementGross": [0.9, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
            "SBA_Appv": [0.9, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
            "GrAppv": [0.9, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
        }

        # Phase 1: Single-feature perturbations
        for feat in FEATURES_TO_VARY:
            for delta in perturbations[feat]:
                candidate = app_dict.copy()
                if feat == "Term":
                    candidate[feat] = app_dict[feat] + delta
                elif feat in ["CreateJob", "RetainedJob"]:
                    candidate[feat] = app_dict[feat] + delta
                else:
                    candidate[feat] = app_dict[feat] * delta
                X_cf = self._to_array(candidate)
                pred = int(self.primary_model.predict(X_cf)[0])
                if pred >= target_class:
                    dist = abs(candidate[feat] - app_dict[feat])
                    if dist < best_dist:
                        best_dist = dist
                        best_cf = candidate
                        best_changes = [(feat, app_dict[feat], candidate[feat])]

        # Phase 2: Two-feature combinations if single failed
        if best_cf is None:
            for i, f1 in enumerate(FEATURES_TO_VARY):
                for f2 in FEATURES_TO_VARY[i + 1:]:
                    for d1 in perturbations[f1][:5]:
                        for d2 in perturbations[f2][:5]:
                            candidate = app_dict.copy()
                            for feat, delta in [(f1, d1), (f2, d2)]:
                                if feat in ["Term", "CreateJob", "RetainedJob"]:
                                    candidate[feat] = app_dict[feat] + delta
                                else:
                                    candidate[feat] = app_dict[feat] * delta
                            X_cf = self._to_array(candidate)
                            pred = int(self.primary_model.predict(X_cf)[0])
                            if pred >= target_class:
                                best_cf = candidate
                                best_changes = []
                                for feat, delta in [(f1, d1), (f2, d2)]:
                                    if feat in ["Term", "CreateJob", "RetainedJob"]:
                                        best_changes.append(
                                            (feat, app_dict[feat], app_dict[feat] + delta))
                                    else:
                                        best_changes.append(
                                            (feat, app_dict[feat], app_dict[feat] * delta))
                                break
                        if best_cf: break
                    if best_cf: break
                if best_cf: break

        changes_out = []
        if best_changes:
            for feat, orig, new_val in best_changes:
                direction = "↑ Increase" if new_val > orig else "↓ Decrease"
                changes_out.append({
                    "feature": feat,
                    "feature_label": FEATURE_LABELS.get(feat, feat),
                    "original_value": round(orig, 2),
                    "recommended_value": round(new_val, 2),
                    "direction": direction,
                })
        return {
            "current_class": current_class, "current_label": LABEL_NAMES[current_class],
            "target_class": target_class, "target_label": LABEL_NAMES[target_class],
            "feasible": len(changes_out) > 0, "changes": changes_out,
        }
```

---

### A.2 Loan Optimizer, Red-Flag Detection & Government Scheme Engine

**File: `api/optimizer.py`**

This module contains three fully custom-built analytical subsystems: (1) a parametric loan structure optimizer that sweeps across term and amount configurations to find the safest loan structure, (2) a rule-based red-flag detection engine that identifies risky parameter combinations derived from patterns in 899,164 historical loan records, and (3) a government scheme matching engine with a hand-curated knowledge base of Indian MSME financing programs.

```python
"""
Loan Structure Optimizer + Red Flags + Government Schemes
Uses the prediction engine and similarity engine to:
1. Find the optimal loan structure for maximum success probability
2. Identify red flags (risky parameter combinations)
3. Match relevant government schemes to the user's profile
"""
import numpy as np
from copy import deepcopy

USD_TO_INR = 83

def fmt_inr(usd_amount):
    """Format USD amount as ₹ in Indian numbering (lakhs/crores)."""
    inr = usd_amount * USD_TO_INR
    if inr >= 10000000: return f"₹{inr/10000000:.1f} Cr"
    elif inr >= 100000:  return f"₹{inr/100000:.1f}L"
    elif inr >= 1000:    return f"₹{inr/1000:.0f}K"
    else:                return f"₹{inr:,.0f}"


# ═══════════════════════════════════════════
# GOVERNMENT SCHEMES KNOWLEDGE BASE
# ═══════════════════════════════════════════

GOVERNMENT_SCHEMES = [
    {"name": "PM SVANidhi (Street Vendors)", "max_amount_inr": 50000,
     "eligibility": {"max_employees": 3, "business_types": [1, 2]},
     "benefits": ["₹50K without collateral", "7% interest subsidy"]},
    {"name": "MUDRA Yojana — Shishu", "max_amount_inr": 50000,
     "eligibility": {"max_employees": 5, "business_types": [1, 2]},
     "benefits": ["Zero collateral", "Low interest (7-12%)"]},
    {"name": "MUDRA Yojana — Kishor", "max_amount_inr": 500000,
     "eligibility": {"max_employees": 20, "business_types": [1, 2]},
     "benefits": ["No collateral up to ₹5L", "Flexible repayment"]},
    {"name": "MUDRA Yojana — Tarun", "max_amount_inr": 1000000,
     "eligibility": {"max_employees": 50, "business_types": [1]},
     "benefits": ["No collateral up to ₹10L", "For expansion"]},
    {"name": "PMEGP", "max_amount_inr": 2500000,
     "eligibility": {"max_employees": 50, "business_types": [2]},
     "benefits": ["15-35% SUBSIDY (free money!)", "Up to ₹25L"]},
    {"name": "CGTMSE (Credit Guarantee)", "max_amount_inr": 20000000,
     "eligibility": {"max_employees": 200, "business_types": [1, 2]},
     "benefits": ["Collateral-free up to ₹2 Cr", "Govt backs loan"]},
    {"name": "Stand-Up India", "max_amount_inr": 10000000,
     "eligibility": {"max_employees": 200, "business_types": [2]},
     "benefits": ["For SC/ST & women entrepreneurs", "₹10L to ₹1Cr"]},
    # ... additional schemes omitted for brevity ...
]

def match_government_schemes(features: dict) -> list[dict]:
    """Match user's business profile to relevant government schemes."""
    loan_inr = features.get("DisbursementGross", 0) * USD_TO_INR
    employees = features.get("NoEmp", 0)
    business_type = features.get("NewExist", 1)

    matched = []
    for scheme in GOVERNMENT_SCHEMES:
        elig = scheme["eligibility"]
        if employees > elig["max_employees"]: continue
        if business_type not in elig["business_types"]: continue
        if loan_inr <= scheme["max_amount_inr"] * 1.5:
            relevance = "high" if loan_inr <= scheme["max_amount_inr"] else "medium"
            matched.append({**scheme, "relevance": relevance})

    matched.sort(key=lambda x: (0 if x["relevance"]=="high" else 1))
    return matched[:5]


# ═══════════════════════════════════════════
# RED FLAGS DETECTION
# ═══════════════════════════════════════════

def detect_red_flags(features: dict, similar_data=None) -> list[dict]:
    """Identify risky parameter combinations from 899K loan patterns."""
    flags = []

    # New business + large loan (28% default vs 15% established)
    if (features.get("NewExist") == 2 and
        features.get("DisbursementGross", 0) > 100000):
        flags.append({
            "severity": "high", "emoji": "🔴",
            "flag": "High-value loan for a new business",
            "explanation": "28% default rate vs 15% for established businesses.",
            "suggestion": "Start smaller and scale after 2-3 years."
        })

    # Low-doc + large amount (2x default rate)
    if (features.get("LowDoc") == 1 and
        features.get("DisbursementGross", 0) > 150000):
        flags.append({
            "severity": "high", "emoji": "🔴",
            "flag": "Low documentation on a large loan",
            "explanation": "Low-doc loans above ₹1 Cr have ~2x default rate.",
            "suggestion": "Prepare full documentation — ITR, GST, bank stmts."
        })

    # Short term + large amount = heavy EMI
    term, amount = features.get("Term", 84), features.get("DisbursementGross", 0)
    if term > 0 and amount / term > 5000 and term < 60:
        monthly_inr = (amount / term) * USD_TO_INR
        flags.append({
            "severity": "high", "emoji": "🔴",
            "flag": "Heavy monthly EMI burden",
            "explanation": f"EMI ~₹{monthly_inr/1000:.0f}K/month.",
            "suggestion": f"Extend to {max(84, term*2)} months to cut EMI ~50%."
        })

    # Low guarantee coverage
    sba, gr = features.get("SBA_Appv", 0), features.get("GrAppv", 1)
    if gr > 0 and sba / gr < 0.5:
        flags.append({
            "severity": "medium", "emoji": "🟡",
            "flag": f"Low guarantee coverage ({sba/gr*100:.0f}%)",
            "explanation": "Loans with <50% coverage default more often.",
            "suggestion": "Apply for CGTMSE guarantee (up to 85%)."
        })

    # Rural + new business (20% higher default)
    if features.get("UrbanRural") == 2 and features.get("NewExist") == 2:
        flags.append({
            "severity": "medium", "emoji": "🟡",
            "flag": "New business in a rural area",
            "explanation": "20% higher default rate than urban counterparts.",
            "suggestion": "PMEGP gives 25-35% subsidy for rural enterprises."
        })

    # Zero job creation
    if features.get("CreateJob", 0) == 0 and features.get("RetainedJob", 0) == 0:
        flags.append({
            "severity": "low", "emoji": "🟡",
            "flag": "No job creation documented",
            "explanation": "Applications with employment plans get priority.",
            "suggestion": "Document planned hiring — even 1-2 jobs helps."
        })

    return sorted(flags, key=lambda x: {"high":0,"medium":1,"low":2}[x["severity"]])


# ═══════════════════════════════════════════
# LOAN STRUCTURE OPTIMIZER
# ═══════════════════════════════════════════

class LoanOptimizer:
    """Finds optimal loan structure by sweeping parameters."""

    def __init__(self, prediction_engine):
        self.engine = prediction_engine

    def find_optimal_term(self, features: dict) -> dict:
        """Find the safest loan term for this business profile."""
        terms = [12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 300, 360]
        results = []
        for term in terms:
            test = deepcopy(features)
            test["Term"] = term
            pred = self.engine.predict(test)
            results.append({
                "term_months": term, "term_years": round(term/12, 1),
                "predicted_class": pred["predicted_class"],
                "predicted_label": pred["predicted_label"],
                "confidence": pred["confidence"],
            })
        best = max(results, key=lambda x: (x["predicted_class"], x["confidence"]))
        return {
            "all_terms": results,
            "recommended_term": best["term_months"],
            "best_class": best["predicted_label"],
            "best_confidence": best["confidence"],
        }

    def find_max_safe_amount(self, features: dict, target_class=2) -> dict:
        """Find max loan amount where predicted class stays >= target."""
        original = features.get("DisbursementGross", 100000)
        multipliers = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
        max_safe = 0
        for mult in multipliers:
            test = deepcopy(features)
            test_amount = int(original * mult)
            test["DisbursementGross"] = test_amount
            test["GrAppv"] = test_amount
            test["SBA_Appv"] = int(test_amount * 0.75)
            pred = self.engine.predict(test)
            if pred["predicted_class"] >= target_class:
                max_safe = test_amount
        return {
            "requested_amount": original, "max_safe_amount": max_safe,
            "can_take_more": max_safe > original,
            "should_reduce": max_safe < original,
        }

    def generate_optimal_structure(self, features: dict) -> dict:
        """Generate the single best loan config for this business."""
        optimized = deepcopy(features)
        term_data = self.find_optimal_term(features)
        optimized["Term"] = term_data["recommended_term"]
        amount_data = self.find_max_safe_amount(features)
        if optimized.get("LowDoc") == 1: optimized["LowDoc"] = 0
        optimized["SBA_Appv"] = int(optimized["DisbursementGross"] * 0.80)
        optimized["GrAppv"] = optimized["DisbursementGross"]
        if optimized.get("CreateJob", 0) == 0:
            optimized["CreateJob"] = max(1, features.get("CreateJob", 0))

        original_pred = self.engine.predict(features)
        optimized_pred = self.engine.predict(optimized)
        changes = [{"feature": k, "original": features[k], "optimized": optimized[k]}
                   for k in features if features[k] != optimized[k]]
        return {
            "original_prediction": original_pred,
            "optimized_prediction": optimized_pred,
            "changes": changes,
            "improvement": optimized_pred["predicted_class"] - original_pred["predicted_class"],
        }
```

---

## B. Screenshots & Results

Below are representative outputs from the deployed working prototype, demonstrating each major capability of the system.

---

**Figure B.1: Model Comparison — 5-Class Benchmark Results**

This table summarizes the accuracy, macro-averaged precision, recall, and F1-score across all six models trained on 899,164 SBA loan records (test set n=178,385). XGBoost (primary) and LightGBM achieve ~93% weighted accuracy. The stacking ensemble achieves the highest overall accuracy at 93.16%.

*(Insert screenshot: `visualizations/model_comparison_table.png`)*

---

**Figure B.2: 5-Class Health Distribution**

Distribution of the engineered 5-class viability labels across the full dataset: Critical (16.2%), At-Risk (1.2%), Stable (53.3%), Growing (17.4%), Thriving (11.8%). The At-Risk class has the lowest representation, which is reflected in its lower recall across all models.

*(Insert screenshot: `visualizations/multiclass_distribution.png`)*

---

**Figure B.3: XGBoost Confusion Matrix (5-Class)**

The confusion matrix for the primary XGBoost model on the test set. The diagonal values confirm strong per-class accuracy, with the Stable, Growing, and Thriving classes achieving >95% recall. The At-Risk class (smallest class) shows the most confusion, primarily misclassified as Critical or Stable.

*(Insert screenshot: `visualizations/xgb_confusion_matrix.png`)*

---

**Figure B.4: SHAP Global Feature Importance**

Mean absolute SHAP values across all classes, showing which features have the largest overall impact on predictions. The top features are Term (loan duration), DisbursementGross (loan amount), and SBA_Appv (SBA guarantee), confirming that loan structure parameters dominate viability outcomes.

*(Insert screenshot: `visualizations/mc_shap_global_bar.png`)*

---

**Figure B.5: SHAP Summary Plot (Beeswarm)**

Per-instance SHAP contributions colored by feature value. Red indicates high feature values, blue indicates low. This plot reveals that longer loan terms and higher disbursement amounts push predictions toward healthier classes (Growing/Thriving), while low values push toward Critical/At-Risk.

*(Insert screenshot: `visualizations/shap_summary_plot.png`)*

---

**Figure B.6: SHAP Force Plot — Individual Prediction**

A force plot for a single loan application, showing exactly how each feature pushes the prediction toward or away from the predicted class. This is the type of explanation presented to users in the dashboard, making the AI's decision fully transparent.

*(Insert screenshot: `visualizations/shap_force_plot_example_1.png`)*

---

**Figure B.7: Working Prototype — Viability Assessment Dashboard**

Screenshot of the deployed Streamlit dashboard showing a single loan assessment result. The interface displays the viability grade (A–F), confidence percentage, class probability distribution chart, SHAP feature contributions, and prescriptive recommendations — all generated in real-time by the FastAPI backend.

*(Take a screenshot of your running Streamlit app — the Single Assessment tab after clicking "Assess Viability" with sample data filled in. Show the full page including the health badge, metric cards, probability chart, SHAP chart, and recommendation cards.)*

---

**Figure B.8: Working Prototype — AI Chat Interface**

Screenshot of the Gemini-powered conversational assessment. The chat interface shows a multi-turn conversation where the AI advisor naturally extracts business details from the user and outputs the viability assessment — no technical form-filling required.

*(Take a screenshot of the Chat/AI Advisor tab showing a completed conversation with extracted features and the resulting assessment.)*

---

**Figure B.9: Working Prototype — Batch Processing**

Screenshot showing the batch CSV processing capability. A CSV file with multiple loan applications is uploaded, and the system returns risk distribution summary and per-application results with downloadable output.

*(Take a screenshot of the Batch Processing tab after processing a sample CSV — show the risk distribution cards and the results table.)*
