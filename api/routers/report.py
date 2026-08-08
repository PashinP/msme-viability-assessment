from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import io
import traceback
from api.schemas.assessment import AssessmentRequest
from api.dependencies import get_db, verify_api_key
from api.services.ml.engine import get_prediction_engine
from api.services.pdf.report_generator import generate_report
from api.services.ml.optimizer import get_loan_optimizer, match_government_schemes
from api.services.ml.scoring import generate_readiness_assessment
from api.services.ml.prescription import generate_prescriptions
from api.services.ml.similarity import get_similar_engine
from api.routers.assessment import _save_prediction

router = APIRouter()

@router.post("/report")
def generate_pdf_report(req: AssessmentRequest, db: Session = Depends(get_db), _key: str = Depends(verify_api_key)):
    engine_instance = get_prediction_engine()
    pred = engine_instance.predict(req.features)
    pid = _save_prediction(db, req.features, pred)
    shap_data = engine_instance.explain(req.features)

    try:
        sim = get_similar_engine()
        similar = sim.find_similar(req.features)
    except Exception:
        similar = None

    assessment = generate_readiness_assessment(req.features, req.context, pred)
    prescriptions = generate_prescriptions(assessment, req.features, req.context)
    
    optimizer = get_loan_optimizer()
    optimizer_data = optimizer.generate_optimal_structure(req.features)
    schemes = match_government_schemes(req.features)
    schemes_out = {"schemes": schemes, "total_matched": len(schemes)}

    try:
        pdf_bytes = generate_report(
            features=req.features, context=req.context, pred=pred,
            shap_data=shap_data, similar=similar, assessment=assessment,
            prescriptions=prescriptions, optimizer=optimizer_data, schemes=schemes_out
        )
    except Exception as e:
        error_msg = traceback.format_exc()
        raise HTTPException(status_code=500, detail=error_msg)

    return StreamingResponse(
        io.BytesIO(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=MSME_Loan_Report.pdf"}
    )
