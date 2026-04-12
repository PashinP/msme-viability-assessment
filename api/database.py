"""
Database layer — SQLAlchemy ORM with SQLite.
Every prediction is persisted for historical audit and analytics.
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "msme_viability.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


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


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for FastAPI — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
