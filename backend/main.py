from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.core.database import init_db
from backend.services.ml.engine import init_prediction_engine
from backend.api.routes import assessment, report, chat, health

app = FastAPI(
    title="MSME Viability Assessment API",
    description="Enterprise-grade loan viability assessment engine.",
    version="2.0.0"
)

@app.on_event("startup")
def startup_event():
    print("Starting up database...")
    init_db()
    print("Database started. Loading ML models...")
    init_prediction_engine()
    print("ML models loaded. Application is ready!")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assessment.router, tags=["Assessment"])
app.include_router(report.router, tags=["Reporting"])
app.include_router(chat.router, tags=["NLP"])
app.include_router(health.router, tags=["System"])
