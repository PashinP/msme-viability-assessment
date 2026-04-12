# MSME Viability Assessment System

An enterprise AI platform for MSME (Micro, Small & Medium Enterprise) loan risk stratification, powered by XGBoost, SHAP explainability, and counterfactual recommendations.

## Architecture

```
┌───────────────┐        HTTP        ┌──────────────┐        SQL        ┌───────────┐
│   Streamlit   │ ──────────────────→│   FastAPI    │ ─────────────────→│  SQLite   │
│   Frontend    │   JSON/CSV/SHAP   │   Backend    │   Persist every   │  Database  │
│  (port 8501)  │ ←──────────────── │  (port 8000) │   prediction      │           │
└───────────────┘                    └──────────────┘                    └───────────┘
                                       ↕ Models
                                    XGBoost + LightGBM
```

## Features

- **5-Class Viability Assessment**: Critical → At-Risk → Stable → Growing → Thriving
- **SHAP Explainability**: Feature-level contribution analysis for every prediction
- **Counterfactual Recommendations**: AI-generated prescriptive interventions (DiCE-inspired)
- **Batch Processing**: Upload CSV files with 100+ loan applications for bulk scoring
- **Database Persistence**: Every prediction stored in SQLite for audit trail
- **API Key Security**: All endpoints protected with authentication
- **Interactive API Docs**: Swagger UI at `/docs`

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Backend
```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### 3. Start the Streamlit Frontend
```bash
streamlit run app.py --server.port 8501
```

### 4. Open the Dashboard
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | System health check |
| `/predict` | POST | Single loan viability assessment |
| `/predict/batch` | POST | Bulk CSV processing |
| `/explain` | POST | SHAP feature contributions |
| `/recommend` | POST | Counterfactual recommendations |
| `/analytics` | GET | Historical prediction analytics |

## Models

Trained on 899,164 U.S. SBA loan records with 11 financial features:
- **XGBoost** (primary) — 92% accuracy
- **LightGBM** — 92% accuracy
- **Random Forest** — 92% accuracy (notebook only, too large for deployment)
- **Stacking Ensemble** — 92% accuracy (notebook only)

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM, SHAP
- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Streamlit
- **Deployment**: Render (API) + Streamlit Community Cloud (Frontend)
