"""API contract.

`PatientInput` is generated from the feature registry (ml/features.py), so the API accepts exactly
the 54 inputs the models were trained on: JSON keys are the dataset column names, categorical
fields only accept their declared levels, continuous fields must sit inside their clinical bounds,
and any extra key (including the target columns Cath/LAD/LCX/RCA) is rejected with a 422.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from ml.features import FEATURES, Feature, FeatureKind

FeatureValue = int | float | str


def _python_name(column: str) -> str:
    return re.sub(r"\W+", "_", column).strip("_").lower()


def _field_definition(feature: Feature) -> tuple[Any, Any]:
    description = feature.label + (f" ({feature.unit})" if feature.unit else "")
    if feature.kind is FeatureKind.CONTINUOUS:
        low, high = feature.bounds
        return float, Field(alias=feature.name, ge=low, le=high, description=description)
    return Literal[tuple(feature.categories)], Field(alias=feature.name, description=description)


def _build_patient_model() -> type[BaseModel]:
    fields = {_python_name(f.name): _field_definition(f) for f in FEATURES}
    if len(fields) != len(FEATURES):
        raise ValueError("Two feature names map to the same Python field name")
    return create_model(
        "PatientInput",
        __config__=ConfigDict(extra="forbid"),
        __doc__="The 54 clinical inputs of one patient, keyed by dataset column name.",
        **fields,
    )


PatientInput = _build_patient_model()


def patient_to_record(patient: BaseModel) -> dict[str, FeatureValue]:
    """Validated request -> {dataset column name: value}."""
    return patient.model_dump(by_alias=True)


# ----- prediction -----------------------------------------------------------------------------

class FeatureContributionOut(BaseModel):
    feature: str
    label: str
    group: str
    unit: str
    value: FeatureValue
    contribution: float = Field(description="Change in predicted probability attributed to this feature")


class TargetPredictionOut(BaseModel):
    target: str
    display_name: str
    model: str
    probability: float = Field(ge=0, le=1, description="Calibrated probability of the positive class")
    predicted_positive: bool
    threshold: float
    base_value: float = Field(description="Average predicted probability over the reference population")
    contributions: list[FeatureContributionOut] = Field(description="All 54 features, largest |contribution| first")


class PredictionResponse(BaseModel):
    predictions: dict[str, TargetPredictionOut]
    disclaimer: str


# ----- metadata -------------------------------------------------------------------------------

class FeatureInfo(BaseModel):
    name: str
    label: str
    kind: str
    group: str
    unit: str
    categories: list[FeatureValue]
    min: float | None
    max: float | None


class MetricSummaryOut(BaseModel):
    mean: float
    std: float


class ImportanceOut(BaseModel):
    feature: str
    label: str
    mean_abs_shap: float


class TargetInfo(BaseModel):
    key: str
    display_name: str
    column: str
    positive_label: str
    model: str
    cv_metrics: dict[str, MetricSummaryOut]
    oof_metrics_calibrated: dict[str, float]
    global_importance: list[ImportanceOut]


class MetadataResponse(BaseModel):
    trained_at: str
    dataset: dict[str, Any]
    validation: dict[str, Any]
    decision_threshold: float
    targets: list[TargetInfo]
    features: list[FeatureInfo]
    disclaimer: str


# ----- samples & health -----------------------------------------------------------------------

class SamplePatient(BaseModel):
    id: str
    title: str
    description: str
    note: str
    ground_truth: dict[str, bool]
    patient: dict[str, FeatureValue]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    targets: list[str]
