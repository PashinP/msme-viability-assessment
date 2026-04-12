"""
MSME Viability Assessment — FastAPI Production Server
======================================================
Endpoints:
  POST /predict          — Single loan assessment
  POST /predict/batch    — CSV batch processing
  POST /explain          — SHAP explanation
  POST /recommend        — Counterfactual recommendations
  GET  /health           — System health check
  GET  /analytics        — Historical analytics from DB
"""
import os, uuid, io, csv
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np

from api.database import init_db, get_db, PredictionRecord
from api.schemas import (
    LoanApplication, PredictionResult, ShapExplanation,
    RecommendationRequest, RecommendationResult, RecommendationChange,
    BatchResult, HealthCheck
)
from api.engine import PredictionEngine, LABEL_NAMES

# ── Config ──
API_KEY = os.getenv("MSME_API_KEY", "msme-dev-key-2024")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str | None = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


# ── Lifespan ──
engine_instance: PredictionEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_instance
    init_db()
    engine_instance = PredictionEngine()
    print(f"[Server] Models loaded: {list(engine_instance.models.keys())}")
    print(f"[Server] Database initialized at: {os.path.abspath('msme_viability.db')}")
    yield
    engine_instance = None


# ── App ──
app = FastAPI(
    title="MSME Viability Assessment API",
    description="AI-powered loan risk stratification with SHAP explainability and counterfactual recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _app_to_dict(app: LoanApplication) -> dict:
    return app.model_dump()


def _save_prediction(db: Session, app_dict: dict, result: dict,
                     batch_id: str | None = None,
                     recommendation: dict | None = None) -> int:
    record = PredictionRecord(
        term=app_dict["Term"],
        no_emp=app_dict["NoEmp"],
        new_exist=app_dict["NewExist"],
        create_job=app_dict["CreateJob"],
        retained_job=app_dict["RetainedJob"],
        disbursement_gross=app_dict["DisbursementGross"],
        urban_rural=app_dict["UrbanRural"],
        rev_line_cr=app_dict["RevLineCr"],
        low_doc=app_dict["LowDoc"],
        sba_appv=app_dict["SBA_Appv"],
        gr_appv=app_dict["GrAppv"],
        predicted_class=result["predicted_class"],
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        model_used=result["model_used"],
        all_probabilities=result["probabilities"],
        recommendation=recommendation,
        batch_id=batch_id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record.id


# ── Endpoints ──

@app.get("/health", response_model=HealthCheck)
def health_check(db: Session = Depends(get_db)):
    total = db.query(func.count(PredictionRecord.id)).scalar()
    return HealthCheck(
        status="healthy",
        models_loaded=list(engine_instance.models.keys()) if engine_instance else [],
        database="connected",
        total_predictions=total or 0,
    )


@app.post("/predict", response_model=PredictionResult)
def predict(application: LoanApplication, db: Session = Depends(get_db),
            _key: str = Depends(verify_api_key)):
    app_dict = _app_to_dict(application)
    result = engine_instance.predict(app_dict)
    pid = _save_prediction(db, app_dict, result)
    return PredictionResult(**result, prediction_id=pid)


@app.post("/predict/batch", response_model=BatchResult)
async def predict_batch(file: UploadFile = File(...), db: Session = Depends(get_db),
                        _key: str = Depends(verify_api_key)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    content = await file.read()
    text = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))

    batch_id = str(uuid.uuid4())[:8]
    results = []
    summary = {name: 0 for name in LABEL_NAMES.values()}

    for row in reader:
        try:
            app_dict = {
                "Term": float(row.get("Term", 0)),
                "NoEmp": int(float(row.get("NoEmp", 0))),
                "NewExist": int(float(row.get("NewExist", 1))),
                "CreateJob": int(float(row.get("CreateJob", 0))),
                "RetainedJob": int(float(row.get("RetainedJob", 0))),
                "DisbursementGross": float(row.get("DisbursementGross", 0)),
                "UrbanRural": int(float(row.get("UrbanRural", 0))),
                "RevLineCr": int(float(row.get("RevLineCr", 0))),
                "LowDoc": int(float(row.get("LowDoc", 0))),
                "SBA_Appv": float(row.get("SBA_Appv", 0)),
                "GrAppv": float(row.get("GrAppv", 0)),
            }
            result = engine_instance.predict(app_dict)
            pid = _save_prediction(db, app_dict, result, batch_id=batch_id)
            summary[result["predicted_label"]] += 1
            results.append(PredictionResult(**result, prediction_id=pid))
        except Exception:
            continue

    return BatchResult(
        total_processed=len(results),
        batch_id=batch_id,
        results=results,
        summary=summary,
    )


@app.post("/explain", response_model=ShapExplanation)
def explain(application: LoanApplication, _key: str = Depends(verify_api_key)):
    app_dict = _app_to_dict(application)
    return ShapExplanation(**engine_instance.explain(app_dict))


@app.post("/recommend", response_model=RecommendationResult)
def recommend(req: RecommendationRequest, db: Session = Depends(get_db),
              _key: str = Depends(verify_api_key)):
    app_dict = _app_to_dict(req.application)
    result = engine_instance.recommend(app_dict, target_class=req.target_class)

    # Also save the base prediction with recommendation
    pred = engine_instance.predict(app_dict)
    pid = _save_prediction(db, app_dict, pred, recommendation=result)

    changes = [RecommendationChange(**c) for c in result["changes"]]
    return RecommendationResult(
        current_class=result["current_class"],
        current_label=result["current_label"],
        target_class=result["target_class"],
        target_label=result["target_label"],
        feasible=result["feasible"],
        changes=changes,
        prediction_id=pid,
    )


@app.get("/analytics")
def analytics(db: Session = Depends(get_db), _key: str = Depends(verify_api_key)):
    total = db.query(func.count(PredictionRecord.id)).scalar() or 0

    # Class distribution
    class_dist = {}
    rows = db.query(
        PredictionRecord.predicted_label,
        func.count(PredictionRecord.id)
    ).group_by(PredictionRecord.predicted_label).all()
    for label, count in rows:
        class_dist[label] = count

    # Average confidence
    avg_conf = db.query(func.avg(PredictionRecord.confidence)).scalar() or 0.0

    # Recent predictions
    recent = db.query(PredictionRecord).order_by(
        PredictionRecord.id.desc()
    ).limit(10).all()
    recent_list = [
        {
            "id": r.id,
            "timestamp": str(r.timestamp),
            "predicted_label": r.predicted_label,
            "confidence": round(r.confidence, 4),
            "term": r.term,
            "disbursement": r.disbursement_gross,
        }
        for r in recent
    ]

    return {
        "total_predictions": total,
        "class_distribution": class_dist,
        "average_confidence": round(avg_conf, 4),
        "recent_predictions": recent_list,
    }
