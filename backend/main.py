"""FastAPI entrypoint. Run from the project root:

    uvicorn backend.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml.config import TARGETS

app = FastAPI(title="CardioLens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "targets": list(TARGETS)}


# Task #6: POST /api/predict -> predictions + probabilities + SHAP breakdown for all 4 models.
