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

        # Load only lightweight models (XGBoost=8MB, LightGBM=2.6MB)
        self.models = {}
        for name, fname in [("XGBoost", "xgb_mc.pkl"), ("LightGBM", "lgbm_mc.pkl")]:
            path = os.path.join(MODELS_DIR, fname)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)

        # Primary model for single predictions
        self.primary_model_name = "XGBoost"
        self.primary_model = self.models[self.primary_model_name]

        # SHAP explainer (lazy init)
        self._shap_explainer = None

    @property
    def shap_explainer(self):
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.primary_model)
        return self._shap_explainer

    def _to_array(self, app_dict: dict) -> np.ndarray:
        """Convert a flat feature dict to a scaled numpy array."""
        row = np.array([[app_dict[f] for f in self.feature_names]])
        return self.scaler.transform(row)

    # ── Predict ──
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

    def predict_all_models(self, app_dict: dict):
        results = {}
        for name in self.models:
            results[name] = self.predict(app_dict, model_name=name)
        return results

    # ── SHAP Explain ──
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
            "predicted_class": pred,
            "predicted_label": LABEL_NAMES[pred],
            "feature_contributions": contributions,
            "top_positive_features": top_pos,
            "top_negative_features": top_neg,
        }

    # ── Counterfactual Recommendations ──
    def recommend(self, app_dict: dict, target_class: int | None = None):
        X = self._to_array(app_dict)
        current_class = int(self.primary_model.predict(X)[0])

        if target_class is None:
            target_class = min(current_class + 1, 4)

        if current_class >= target_class:
            return {
                "current_class": current_class,
                "current_label": LABEL_NAMES[current_class],
                "target_class": target_class,
                "target_label": LABEL_NAMES[target_class],
                "feasible": True,
                "changes": [],
            }

        # Systematic grid search over modifiable features
        best_cf = None
        best_changes = []
        best_dist = float("inf")

        np.random.seed(42)

        # Define perturbation ranges for each modifiable feature
        perturbations = {
            "Term": [12, 24, 36, 48, 60, 84, 120, 180, 240],
            "CreateJob": list(range(1, 25)),
            "RetainedJob": list(range(1, 25)),
            "DisbursementGross": [0.9, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
            "SBA_Appv": [0.9, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
            "GrAppv": [0.9, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
        }

        # Try single-feature changes first (simplest recommendations)
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

        # Try two-feature changes if single didn't work
        if best_cf is None:
            for i, f1 in enumerate(FEATURES_TO_VARY):
                for f2 in FEATURES_TO_VARY[i + 1:]:
                    for d1 in perturbations[f1][:5]:
                        for d2 in perturbations[f2][:5]:
                            candidate = app_dict.copy()
                            for feat, delta in [(f1, d1), (f2, d2)]:
                                if feat == "Term":
                                    candidate[feat] = app_dict[feat] + delta
                                elif feat in ["CreateJob", "RetainedJob"]:
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
                                            (feat, app_dict[feat], app_dict[feat] + delta)
                                        )
                                    else:
                                        best_changes.append(
                                            (feat, app_dict[feat], app_dict[feat] * delta)
                                        )
                                break
                        if best_cf:
                            break
                    if best_cf:
                        break
                if best_cf:
                    break

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
            "current_class": current_class,
            "current_label": LABEL_NAMES[current_class],
            "target_class": target_class,
            "target_label": LABEL_NAMES[target_class],
            "feasible": len(changes_out) > 0,
            "changes": changes_out,
        }
