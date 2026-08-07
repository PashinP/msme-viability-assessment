import random

class DummySimilarBusinessEngine:
    def __init__(self):
        self.total_records = 897000

    def find_similar(self, app_dict: dict, k: int = 50) -> dict:
        # Generate dummy data that looks real for the demo
        success_count = random.randint(30, 45)
        default_count = k - success_count
        success_rate = success_count / k
        
        mock_businesses = [
            {"rank": 1, "name": "Standard Enterprises LLC", "state": "GJ", "outcome": "Paid in Full", "outcome_emoji": "✅", "similarity_score": 0.98, "term": app_dict.get("Term", 84), "employees": app_dict.get("NoEmp", 5), "disbursement": app_dict.get("DisbursementGross", 50000)},
            {"rank": 2, "name": "Local Retail Co", "state": "GJ", "outcome": "Paid in Full", "outcome_emoji": "✅", "similarity_score": 0.95, "term": app_dict.get("Term", 84), "employees": app_dict.get("NoEmp", 5)+2, "disbursement": app_dict.get("DisbursementGross", 50000)*1.1},
            {"rank": 3, "name": "City Logistics", "state": "MH", "outcome": "Defaulted", "outcome_emoji": "❌", "similarity_score": 0.91, "term": app_dict.get("Term", 84)-12, "employees": app_dict.get("NoEmp", 5), "disbursement": app_dict.get("DisbursementGross", 50000)*0.8},
            {"rank": 4, "name": "National Traders", "state": "MH", "outcome": "Paid in Full", "outcome_emoji": "✅", "similarity_score": 0.89, "term": app_dict.get("Term", 84), "employees": app_dict.get("NoEmp", 5)-1, "disbursement": app_dict.get("DisbursementGross", 50000)},
            {"rank": 5, "name": "Western Suppliers", "state": "RJ", "outcome": "Paid in Full", "outcome_emoji": "✅", "similarity_score": 0.88, "term": app_dict.get("Term", 84)+12, "employees": app_dict.get("NoEmp", 5)+5, "disbursement": app_dict.get("DisbursementGross", 50000)*1.5},
        ]
        
        return {
            "total_similar": k,
            "success_count": success_count,
            "default_count": default_count,
            "success_rate": round(success_rate, 4),
            "baseline_success_rate": 0.73,
            "risk_vs_baseline": "above_average" if success_rate > 0.73 else "average",
            "similar_businesses": mock_businesses,
            "insight": f"🟢 Out of {k} businesses with a similar profile, {success_count} ({round(success_rate*100)}%) successfully repaid their loans. This is better than the overall SBA average.",
            "dataset_size": self.total_records
        }
