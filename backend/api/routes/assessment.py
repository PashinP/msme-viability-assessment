from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.schemas.assessment import AssessmentRequest
from backend.api.dependencies import get_db, verify_api_key
from backend.services.ml.engine import get_prediction_engine
from backend.services.ml.scoring import generate_readiness_assessment
from backend.services.ml.prescription import generate_prescriptions
from backend.services.ml.optimizer import match_government_schemes
from backend.models.prediction import PredictionRecord

router = APIRouter()

def _save_prediction(db: Session, app_dict: dict, result: dict):
    from datetime import datetime, timezone
    record = PredictionRecord(
        term=app_dict["Term"], no_emp=app_dict["NoEmp"], new_exist=app_dict["NewExist"],
        create_job=app_dict["CreateJob"], retained_job=app_dict["RetainedJob"],
        disbursement_gross=app_dict["DisbursementGross"], urban_rural=app_dict["UrbanRural"],
        rev_line_cr=app_dict["RevLineCr"], low_doc=app_dict["LowDoc"],
        sba_appv=app_dict["SBA_Appv"], gr_appv=app_dict["GrAppv"],
        predicted_class=result["predicted_class"], predicted_label=result["predicted_label"],
        confidence=result["confidence"], model_used=result["model_used"],
        all_probabilities=result["probabilities"]
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record.id

@router.post("/assess")
def full_assessment(req: AssessmentRequest, db: Session = Depends(get_db), _key: str = Depends(verify_api_key)):
    engine_instance = get_prediction_engine()
    if not engine_instance:
        raise HTTPException(503, "Scoring engine not available")
        
    pred = engine_instance.predict(req.features)
    pid = _save_prediction(db, req.features, pred)
    assessment = generate_readiness_assessment(req.features, req.context, pred)
    prescriptions = generate_prescriptions(assessment, req.features, req.context)
    schemes = match_government_schemes(req.features)

    return {
        "prediction_id": pid, "prediction": pred, "assessment": assessment,
        "prescriptions": prescriptions, "schemes": schemes
    }

@router.post("/explain")
def explain_prediction(features: dict, _key: str = Depends(verify_api_key)):
    engine_instance = get_prediction_engine()
    if not engine_instance:
        raise HTTPException(503, "Scoring engine not available")
    return engine_instance.explain(features)

@router.post("/similar")
def similar_profiles(features: dict, _key: str = Depends(verify_api_key)):
    try:
        from backend.services.ml.similarity import get_similar_engine
        sim = get_similar_engine()
        return sim.find_similar(features)
    except Exception as e:
        raise HTTPException(503, str(e))
