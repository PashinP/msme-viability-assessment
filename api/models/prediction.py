from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from datetime import datetime, timezone
from api.core.database import Base

class PredictionRecord(Base):
    """Stores every prediction the system makes for audit trail."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Input features
    term = Column(Float)
    no_emp = Column(Integer)
    new_exist = Column(Integer)
    create_job = Column(Integer)
    retained_job = Column(Integer)
    disbursement_gross = Column(Float)
    urban_rural = Column(Integer)
    rev_line_cr = Column(Integer)
    low_doc = Column(Integer)
    sba_appv = Column(Float)
    gr_appv = Column(Float)

    # Prediction outputs
    predicted_class = Column(Integer)
    predicted_label = Column(String)
    confidence = Column(Float)
    model_used = Column(String)
    all_probabilities = Column(JSON)

    # Recommendations (if generated)
    recommendation = Column(JSON, nullable=True)

    # Batch tracking
    batch_id = Column(String, nullable=True)
