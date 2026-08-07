"""
Similar Business Engine — KNN-based historical matching
=========================================================
Finds the K most similar historical SBA loans and reports
their outcomes, giving users data-driven context for their
loan application.
"""
import os
import numpy as np
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

FEATURE_COLS = ["Term", "NoEmp", "NewExist", "CreateJob", "RetainedJob",
                "DisbursementGross", "UrbanRural", "RevLineCr", "LowDoc",
                "SBA_Appv", "GrAppv"]


class SimilarBusinessEngine:
    """Finds similar historical SBA loans using KNN."""

    def __init__(self):
        data = joblib.load(os.path.join(DATA_DIR, "sba_features.pkl"))
        self.features = data["features"]           # unscaled originals
        self.features_scaled = data["features_scaled"]
        self.outcomes = data["outcomes"]            # 1=success, 0=default
        self.meta = data["meta"]                    # Name, State, NAICS, MIS_Status
        self.knn = joblib.load(os.path.join(DATA_DIR, "sba_knn.pkl"))
        self.scaler = joblib.load(os.path.join(DATA_DIR, "sba_knn_scaler.pkl"))
        self.total_records = len(self.outcomes)
        self.total_success = int(self.outcomes.sum())
        self.total_default = self.total_records - self.total_success

    def find_similar(self, app_dict: dict, k: int = 50) -> dict:
        """
        Find K most similar historical loans.

        Returns:
            {
                "total_similar": 50,
                "success_count": 35,
                "default_count": 15,
                "success_rate": 0.70,
                "baseline_success_rate": 0.82,
                "risk_vs_baseline": "below_average",
                "similar_businesses": [...top 5 details...],
                "insight": "Your profile matches 50 historical loans..."
            }
        """
        # Build input vector
        row = np.array([[app_dict[f] for f in FEATURE_COLS]], dtype=np.float32)
        row_scaled = self.scaler.transform(row)

        # KNN query
        distances, indices = self.knn.kneighbors(row_scaled, n_neighbors=k)
        indices = indices[0]
        distances = distances[0]

        # Outcome stats
        neighbor_outcomes = self.outcomes[indices]
        success_count = int(neighbor_outcomes.sum())
        default_count = k - success_count
        success_rate = success_count / k

        baseline_rate = self.total_success / self.total_records

        if success_rate >= baseline_rate + 0.05:
            risk_level = "above_average"
            risk_emoji = "🟢"
        elif success_rate >= baseline_rate - 0.05:
            risk_level = "average"
            risk_emoji = "🟡"
        else:
            risk_level = "below_average"
            risk_emoji = "🔴"

        # Top 5 most similar businesses (for display)
        top_businesses = []
        for i in range(min(5, k)):
            idx = indices[i]
            biz = {
                "rank": i + 1,
                "name": str(self.meta.iloc[idx]["Name"]).strip()[:30],
                "state": str(self.meta.iloc[idx]["State"]).strip(),
                "outcome": "Paid in Full" if self.outcomes[idx] == 1 else "Defaulted",
                "outcome_emoji": "✅" if self.outcomes[idx] == 1 else "❌",
                "similarity_score": round(1.0 / (1.0 + float(distances[i])), 4),
                "term": float(self.features[idx][0]),
                "employees": int(self.features[idx][1]),
                "disbursement": float(self.features[idx][5]),
            }
            top_businesses.append(biz)

        # Generate human-readable insight
        pct = round(success_rate * 100)
        insight = (
            f"{risk_emoji} Out of {k} businesses with a similar profile, "
            f"{success_count} ({pct}%) successfully repaid their loans and "
            f"{default_count} ({100-pct}%) defaulted. "
        )
        if risk_level == "above_average":
            insight += "This is better than the overall average — a positive signal."
        elif risk_level == "average":
            insight += "This is in line with the overall average."
        else:
            insight += "This is below the overall average — additional scrutiny may be warranted."

        return {
            "total_similar": k,
            "success_count": success_count,
            "default_count": default_count,
            "success_rate": round(success_rate, 4),
            "baseline_success_rate": round(baseline_rate, 4),
            "risk_vs_baseline": risk_level,
            "similar_businesses": top_businesses,
            "insight": insight,
            "dataset_size": self.total_records,
        }
