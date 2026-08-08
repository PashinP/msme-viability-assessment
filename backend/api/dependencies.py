from fastapi import Header, HTTPException, status
from backend.core.database import SessionLocal

API_KEYS = {
    "msme-dev-key-2024",
    "msme-prod-key-2025"
}

def get_db():
    """Dependency for FastAPI — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_api_key(x_api_key: str = Header(...)):
    """Simple API Key validation."""
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return x_api_key
