"""
Pydantic schemas — request/response validation for the API.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Request Schemas ──

class LoanApplication(BaseModel):
    """A single MSME loan application."""
    Term: float = Field(..., ge=0, le=480, description="Loan duration in months")
    NoEmp: int = Field(..., ge=0, le=10000, description="Number of employees")
    NewExist: int = Field(..., ge=0, le=2, description="1=Existing, 2=New business")
    CreateJob: int = Field(..., ge=0, le=5000, description="Jobs to create")
    RetainedJob: int = Field(..., ge=0, le=5000, description="Jobs retained")
    DisbursementGross: float = Field(..., ge=0, description="Loan disbursement amount ($)")
    UrbanRural: int = Field(..., ge=0, le=2, description="1=Urban, 2=Rural, 0=Undefined")
    RevLineCr: int = Field(..., ge=0, le=1, description="Revolving line of credit (0/1)")
    LowDoc: int = Field(..., ge=0, le=1, description="Low doc loan (0/1)")
    SBA_Appv: float = Field(..., ge=0, description="SBA guaranteed amount ($)")
    GrAppv: float = Field(..., ge=0, description="Gross approved amount ($)")


class RecommendationRequest(BaseModel):
    """Request for counterfactual recommendations."""
    application: LoanApplication
    target_class: Optional[int] = Field(
        None, ge=0, le=4,
        description="Desired health class (0-4). If None, targets one class above current."
    )


# ── Response Schemas ──

class PredictionResult(BaseModel):
    """Prediction response for a single application."""
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    model_used: str
    prediction_id: int


class ShapExplanation(BaseModel):
    """SHAP explanation for one prediction."""
    predicted_class: int
    predicted_label: str
    feature_contributions: dict[str, float]
    top_positive_features: list[str]
    top_negative_features: list[str]


class RecommendationChange(BaseModel):
    """A single recommended change."""
    feature: str
    feature_label: str
    original_value: float
    recommended_value: float
    direction: str


class RecommendationResult(BaseModel):
    """Full recommendation response."""
    current_class: int
    current_label: str
    target_class: int
    target_label: str
    feasible: bool
    changes: list[RecommendationChange]
    prediction_id: int


class BatchResult(BaseModel):
    """Response for batch predictions."""
    total_processed: int
    batch_id: str
    results: list[PredictionResult]
    summary: dict[str, int]


class HealthCheck(BaseModel):
    """API health check response."""
    status: str
    models_loaded: list[str]
    database: str
    total_predictions: int
