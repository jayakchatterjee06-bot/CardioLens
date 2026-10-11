"""FastAPI entrypoint. Run from the project root:

    uvicorn backend.main:app --reload --port 8000

Interactive API docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import ModelsNotTrainedError, PredictionService, Predictor
from backend.samples import load_sample_patients
from backend.schemas import (
    HealthResponse,
    MetadataResponse,
    PatientInput,
    PredictionResponse,
    SamplePatient,
    patient_to_record,
)
from backend.settings import ALLOWED_ORIGINS
from ml.config import TARGETS

logger = logging.getLogger("cardiolens")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models, explainers and sample patients once per process."""
    app.state.samples = load_sample_patients()
    try:
        app.state.service = PredictionService.from_artifacts()
    except ModelsNotTrainedError as error:
        logger.warning("%s The API will answer 503 on prediction routes.", error)
        app.state.service = None
    else:
        # The first SHAP call pays a one-off JIT/initialisation cost (~5 s); pay it at startup, not on a user's click.
        app.state.service.predict(app.state.samples[0]["patient"])
    yield


app = FastAPI(title="CardioLens API", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service(request: Request) -> Predictor:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Models are not trained yet. Run `python -m ml.train`.")
    return service


ServiceDep = Annotated[Predictor, Depends(get_service)]


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "models_loaded": getattr(request.app.state, "service", None) is not None,
        "targets": list(TARGETS),
    }


@app.get("/api/metadata", response_model=MetadataResponse)
def metadata(service: ServiceDep) -> dict:
    """Feature schema (labels, units, allowed values, bounds) and per-target model performance."""
    return service.metadata()


@app.get("/api/samples", response_model=list[SamplePatient])
def samples(request: Request) -> list[dict]:
    """Real dataset patients for demo buttons; each `patient` object can be posted to /api/predict as-is."""
    return request.app.state.samples


@app.post("/api/predict", response_model=PredictionResponse)
def predict(patient: PatientInput, service: ServiceDep) -> dict:  # sync: SHAP is CPU-bound, runs in the threadpool
    """Calibrated CAD / LAD / LCX / RCA probabilities with a per-feature SHAP breakdown for each."""
    return service.predict(patient_to_record(patient))
