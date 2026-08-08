from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from api.dependencies import get_db, verify_api_key
from api.models.prediction import PredictionRecord
from api.services.ml.engine import get_prediction_engine

router = APIRouter()

@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    engine_instance = get_prediction_engine()
    total = db.query(func.count(PredictionRecord.id)).scalar()
    return {
        "status": "healthy",
        "models_loaded": list(engine_instance.models.keys()) if engine_instance else [],
        "database": "connected",
        "total_predictions": total or 0,
    }

@router.get("/analytics")
def analytics(db: Session = Depends(get_db), _key: str = Depends(verify_api_key)):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    last_week = now - timedelta(days=7)
    
    total = db.query(func.count(PredictionRecord.id)).scalar()
    recent = db.query(func.count(PredictionRecord.id)).filter(PredictionRecord.timestamp >= last_week).scalar()
    
    label_counts = db.query(PredictionRecord.predicted_label, func.count(PredictionRecord.id)).group_by(PredictionRecord.predicted_label).all()
    distribution = {lbl: int(cnt) for lbl, cnt in label_counts}
    
    return {
        "total_predictions": total,
        "recent_predictions": recent,
        "class_distribution": distribution
    }
