# MSME Viability Assessment Ecosystem — Project Context

This document contains the complete technical context of the current state of the MSME (Micro, Small & Medium Enterprise) Viability Assessment project. Use this for continuing development.

---

## 1. Project Background & Paradigm Shift
This project recently pivoted from a legacy predictive approach using deep learning (LSTM) to a state-of-the-art, **Enterprise-Grade Prescriptive System**. 
- **Legacy Architecture:** Standard monolithic predictive modeling.
- **New Architecture:** Decoupled ecosystem (FastAPI Backend + Streamlit Frontend). 
- **Core Value Proposition:** Moves beyond *predicting* default/risk towards *prescriptive interventions*, utilizing SHAP for explainability and a DiCE-inspired algorithmic approach to generate counterfactual recommendations.

---

## 2. Decoupled Architecture

The system is fully decoupled to allow for robust scaling, cloud deployments, and separation of ML logic from the UI.

### A. FastAPI Backend (`/api`)
- **Port:** `8000`
- **Framework:** FastAPI, Uvicorn (ASGI), Pydantic (data validation/schemas).
- **Authentication:** Token-based API key security (`X-API-Key`).
- **Core Endpoints:**
  - `POST /predict`: Evaluates a single loan application.
  - `POST /predict/batch`: Processes bulk CSV requests.
  - `POST /explain`: Generates SHAP values for the prediction instance.
  - `POST /recommend`: Conducts counterfactual perturbations.
  - `GET /health` & `GET /analytics`: Telemetry and persistence aggregation.

### B. Streamlit Frontend (`app.py`)
- **Port:** `8501`
- **Design Aesthetic:** Premium enterprise dashboard injected with custom CSS (Inter font, glassmorphism gradients, unified hex colors, responsive metric cards).
- **Core Functionality:** Queries the FastAPI backend for all heavy lifting. Provides modular tabs for: Single Assessment, Batch Processing, and Database Analytics. 

---

## 3. Data & Persistence Layer
- **Storage System:** SQLite via SQLAlchemy ORM.
- **Schema (`api/database.py`):** Automatically logs **every single API request**, recording exactly what feature values were sent, the model used, the output probabilities, confidence thresholds, and the generated DiCE recommendations.
- **Audit Logging:** Allows full audit trails of ML operations (e.g., historical predictions chart parsing `PredictionRecord`).

---

## 4. Machine Learning & The "Engine" (`api/engine.py`)

The ML logic is concentrated in `api/engine.py` (serving as a Singleton during backend lifespan). All models are lightweight and pickled. 

### Prediction Matrix
- Trained on **899,164 U.S. SBA loan records**.
- Uses an 11-feature input array (e.g., Term, DisbursementGross, CreateJob, SBA_Appv).
- **Model Output:** 5-Class Viability Spectrum (Label 0: Critical → 1: At-Risk → 2: Stable → 3: Growing → 4: Thriving).
- **Live Models:** Primary relies on robust Gradient Boosting architectures (`XGBoost` and `LightGBM`) scaling to 92% base accuracy. Random Forest/Stacking are relegated to notebooks due to payload size. 

### Explainability (SHAP Integration)
- The system instantiates `shap.TreeExplainer(self.primary_model)` explicitly inside the engine.
- Calculates exact SHAP feature contributions arrayed across the 11 feature inputs in real-time, identifying the top 3 Positive Risk Factors and Top 3 Restricting Factors.

### Prescriptive Logic (The Counterfactual Recommender)
Instead of relying on heavy adversarial DL networks, the project uses a custom, highly optimized deterministic grid-search algorithmic approach inspired by DiCE (Diverse Counterfactual Explanations).
- **How it works:** 
  1. The engine checks the current class. 
  2. If the user doesn't meet the `target_class` (defaulting to +1 of their current state), it iterates over `FEATURES_TO_VARY`.
  3. It performs single-feature, and subsequently two-feature combinations of standard scaling perturbations (e.g., reducing `Disbursement` by 10%, extending `Term` to 120 months) mapped via `np.random.seed`.
  4. Once it finds an input vector that bumps the ML model output to the desired target class, it outputs actionable, real-world deltas back to the JSON payload.

---

## 5. Directory Structure & Files

```text
Practicum_Project/
├── README.md               # Quickstart and config notes
├── requirements.txt        # fastapi, shap, xgboost, lightgbm, streamlit, sqlalchemy, etc.
├── app.py                  # Streamlit Premium UI Dashboard
├── msme_viability.db       # SQLite Database for persisting ML history
├── models/
│   ├── metadata.json       # Array lists mapping inputs
│   ├── scaler_mc.pkl       # Scikit-learn normalizers
│   ├── xgb_mc.pkl          # Pickled XGBoost Model
│   └── lgbm_mc.pkl         # Pickled LightGBM Model
└── api/                    # FastAPI Module
    ├── __init__.py
    ├── server.py           # App routing, endpoint controllers, lifespan definitions
    ├── engine.py           # Core ML Singleton class, prediction, shap, and counterfactuals
    ├── database.py         # SQLAlchemy configuration & ORM tables
    └── schemas.py          # Pydantic typing and models
```

## 6. How To Continue Development

When suggesting new code or debugging:
1. **Never break the decoupling:** Ensure frontend requests utilize HTTP via the `api_call` wrapper in `app.py`. Do not load ML tools directly in the frontend. 
2. **Feature Matching:** Ensure any new UI sliders directly match the 11 schema keys strictly defined in `api/schemas.py`.
3. **Keep it Enterprise:** Any UI changes must map to the CSS payload style (maintaining `card`, `metric-box` aesthetics). No raw tables or generic aesthetics.
4. **Target Deployment:** Keep in mind the project targets Render (API) and Streamlit Community Cloud (Frontend). Ensure dependencies remain pip-installable linux standards.
