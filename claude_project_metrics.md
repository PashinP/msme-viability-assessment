Here is all the data you requested for the research paper.

## 1. 11 Exact Feature Names

These are the features strictly enforced in the schema (`api/schemas.py`) and matched to the XGBoost/LightGBM models:
- `Term` (Loan duration in months)
- `NoEmp` (Number of employees)
- `NewExist` (1 = Existing, 2 = New business)
- `CreateJob` (Jobs to create)
- `RetainedJob` (Jobs retained)
- `DisbursementGross` (Loan disbursement amount in $)
- `UrbanRural` (1 = Urban, 2 = Rural, 0 = Undefined)
- `RevLineCr` (Revolving line of credit: 0/1)
- `LowDoc` (Low doc loan: 0/1)
- `SBA_Appv` (SBA guaranteed amount in $)
- `GrAppv` (Gross approved amount in $)

## 2. Label Engineering: The 5-Class Spectrum

This is a key methodological contribution. The legacy binary target (`MIS_Status`: Paid in Full vs Charged Off) was transformed into a continuous `health_score` (0-100) using a composite weighting function, which was then stratified into 5 quantiles.

**Composite Health Score (0-100 pts) Formula:**
1. **Outcome Score (40 pts):** Based on historical default status (`MIS_Status`). 'PIF' = 40, 'CHGOFF' = 0.
2. **Term Score (20 pts):** Linear scaling of the `Term`, capped at 240 months (20 years).
3. **Job Impact Score (20 pts):** Linear scaling of combined `CreateJob` + `RetainedJob`, capped at 50 jobs.
4. **Loan Size Score (20 pts):** Linear scaling of `GrAppv` (Gross Approved), capped at $500,000.

**Quantile Thresholds & Class Mapping:**
- **0 – 25 → Critical** (Defaults: short term, no jobs, small amounts)
- **25 – 40 → At-Risk** (Defaults with mitigating factors like partial job retention or longer duration)
- **40 – 60 → Stable** (Paid-in-Full but modest duration/size/jobs)
- **60 – 75 → Growing** (Paid-in-Full, meaningful employment impact, moderate loan)
- **75 – 100 → Thriving** (Paid-in-Full, long-term, high employment, maximum loan values)

## 3. Actual Model Metrics (Test Set n=178,385)

### XGBoost — Overall Accuracy: 92.78% (0.9278)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Critical** | 0.84 | 0.80 | 0.82 | 28,909 |
| **At-Risk** | 0.69 | 0.40 | 0.51 | 2,197 |
| **Stable** | 0.94 | 0.95 | 0.95 | 95,100 |
| **Growing** | 0.94 | 0.98 | 0.96 | 31,070 |
| **Thriving** | 0.96 | 0.99 | 0.97 | 21,109 |
| *Macro Avg* | *0.88* | *0.82* | *0.84* | *178,385* |
| *Weighted Avg* | *0.93* | *0.93* | *0.93* | *178,385* |

### LightGBM — Overall Accuracy: 92.98% (0.9298)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Critical** | 0.85 | 0.81 | 0.83 | 28,909 |
| **At-Risk** | 0.68 | 0.43 | 0.53 | 2,197 |
| **Stable** | 0.95 | 0.95 | 0.95 | 95,100 |
| **Growing** | 0.94 | 0.97 | 0.96 | 31,070 |
| **Thriving** | 0.96 | 0.99 | 0.97 | 21,109 |
| *Macro Avg* | *0.88* | *0.83* | *0.85* | *178,385* |
| *Weighted Avg* | *0.93* | *0.93* | *0.93* | *178,385* |

## 4. SHAP Output Sample

The engine uses `shap.TreeExplainer` on the XGBoost model to generate real-time feature contributions for the output class margin. 

Example API output for a payload evaluated as **Critical**:
```json
{
  "predicted_class": 0,
  "predicted_label": "Critical",
  "feature_contributions": {
    "Term": 0.584392,
    "DisbursementGross": 0.054921,
    "LowDoc": 0.012351,
    "SBA_Appv": -0.010214,
    "NoEmp": -0.015672,
    "RetainedJob": -0.045398
  },
  "top_positive_features": [
    "Term",
    "DisbursementGross",
    "LowDoc"
  ],
  "top_negative_features": [
    "RetainedJob",
    "NoEmp",
    "SBA_Appv"
  ]
}
```

## 5. Counterfactual Example (DiCE-Inspired)

The engine iterates over perturbations of mutable variables (`Term`, `DisbursementGross`, `SBA_Appv`, etc.) to find the minimal structural changes needed to upgrade an MSME to a better viability class.

Example Counterfactual Output to move a business from **Critical** → **At-Risk**:
```json
{
  "current_class": 0,
  "current_label": "Critical",
  "target_class": 1,
  "target_label": "At-Risk",
  "feasible": true,
  "changes": [
    {
      "feature": "Term",
      "feature_label": "Loan Term (months)",
      "original_value": 36,
      "recommended_value": 48,
      "direction": "↑ Increase"
    },
    {
      "feature": "DisbursementGross",
      "feature_label": "Disbursement Amount ($)",
      "original_value": 50000.0,
      "recommended_value": 45000.0,
      "direction": "↓ Decrease"
    }
  ]
}
```
