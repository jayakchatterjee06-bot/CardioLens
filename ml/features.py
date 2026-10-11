"""Declarative registry of the 54 model input features.

Preprocessing, API validation and dashboard grouping are all derived from this list,
so adding a clinical feature means adding one entry here.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ml.config import LEAKAGE_COLUMNS


class FeatureKind(str, Enum):
    CONTINUOUS = "continuous"  # standardised measurement
    BINARY = "binary"          # categories = (negative, positive) -> 0 / 1
    ORDINAL = "ordinal"        # ordered categories -> 0..k-1, then standardised
    NOMINAL = "nominal"        # unordered categories -> one-hot


class FeatureGroup(str, Enum):
    DEMOGRAPHIC = "demographic"
    HISTORY = "history"
    EXAMINATION = "examination"
    ECG = "ecg"
    LABORATORY = "laboratory"
    ECHO = "echocardiography"


@dataclass(frozen=True)
class Feature:
    name: str  # column name in the dataset and key in the API payload
    kind: FeatureKind
    group: FeatureGroup
    label: str
    unit: str = ""
    categories: tuple = ()

    def __post_init__(self) -> None:
        if self.kind is FeatureKind.CONTINUOUS and self.categories:
            raise ValueError(f"{self.name}: continuous features take no categories")
        if self.kind is FeatureKind.BINARY and len(self.categories) != 2:
            raise ValueError(f"{self.name}: binary features need exactly (negative, positive)")
        if self.kind in (FeatureKind.ORDINAL, FeatureKind.NOMINAL) and len(self.categories) < 2:
            raise ValueError(f"{self.name}: {self.kind.value} features need at least two categories")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind.value,
            "group": self.group.value,
            "unit": self.unit,
            "categories": list(self.categories),
        }


_YN = ("N", "Y")
_FLAG = (0, 1)
G = FeatureGroup


def _num(name: str, group: FeatureGroup, label: str, unit: str = "") -> Feature:
    return Feature(name, FeatureKind.CONTINUOUS, group, label, unit)


def _bin(name: str, group: FeatureGroup, label: str, categories: tuple = _YN) -> Feature:
    return Feature(name, FeatureKind.BINARY, group, label, categories=categories)


FEATURES: tuple[Feature, ...] = (
    # Demographic
    _num("Age", G.DEMOGRAPHIC, "Age", "years"),
    _num("Weight", G.DEMOGRAPHIC, "Weight", "kg"),
    _num("Length", G.DEMOGRAPHIC, "Height", "cm"),
    _bin("Sex", G.DEMOGRAPHIC, "Sex (male)", ("Female", "Male")),
    _num("BMI", G.DEMOGRAPHIC, "BMI", "kg/m²"),
    # History / risk factors
    _bin("DM", G.HISTORY, "Diabetes mellitus", _FLAG),
    _bin("HTN", G.HISTORY, "Hypertension", _FLAG),
    _bin("Current Smoker", G.HISTORY, "Current smoker", _FLAG),
    _bin("EX-Smoker", G.HISTORY, "Ex-smoker", _FLAG),
    _bin("FH", G.HISTORY, "Family history of CAD", _FLAG),
    _bin("Obesity", G.HISTORY, "Obesity"),
    _bin("CRF", G.HISTORY, "Chronic renal failure"),
    _bin("CVA", G.HISTORY, "Prior stroke (CVA)"),
    _bin("Airway disease", G.HISTORY, "Airway disease"),
    _bin("Thyroid Disease", G.HISTORY, "Thyroid disease"),
    _bin("CHF", G.HISTORY, "Congestive heart failure"),
    _bin("DLP", G.HISTORY, "Dyslipidemia"),
    # Symptoms & examination
    _num("BP", G.EXAMINATION, "Blood pressure", "mmHg"),
    _num("PR", G.EXAMINATION, "Pulse rate", "bpm"),
    _bin("Edema", G.EXAMINATION, "Edema", _FLAG),
    _bin("Weak Peripheral Pulse", G.EXAMINATION, "Weak peripheral pulse"),
    _bin("Lung rales", G.EXAMINATION, "Lung rales"),
    _bin("Systolic Murmur", G.EXAMINATION, "Systolic murmur"),
    _bin("Diastolic Murmur", G.EXAMINATION, "Diastolic murmur"),
    _bin("Typical Chest Pain", G.EXAMINATION, "Typical chest pain", _FLAG),
    _bin("Dyspnea", G.EXAMINATION, "Dyspnea"),
    Feature("Function Class", FeatureKind.ORDINAL, G.EXAMINATION, "Functional class", categories=(0, 1, 2, 3)),
    _bin("Atypical", G.EXAMINATION, "Atypical chest pain"),
    _bin("Nonanginal", G.EXAMINATION, "Non-anginal chest pain"),
    _bin("LowTH Ang", G.EXAMINATION, "Low-threshold angina"),
    # ECG
    _bin("Q Wave", G.ECG, "Q wave", _FLAG),
    _bin("St Elevation", G.ECG, "ST elevation", _FLAG),
    _bin("St Depression", G.ECG, "ST depression", _FLAG),
    _bin("Tinversion", G.ECG, "T-wave inversion", _FLAG),
    _bin("LVH", G.ECG, "LV hypertrophy"),
    _bin("Poor R Progression", G.ECG, "Poor R-wave progression"),
    Feature("BBB", FeatureKind.NOMINAL, G.ECG, "Bundle branch block", categories=("N", "LBBB", "RBBB")),
    # Laboratory
    _num("FBS", G.LABORATORY, "Fasting blood sugar", "mg/dL"),
    _num("CR", G.LABORATORY, "Creatinine", "mg/dL"),
    _num("TG", G.LABORATORY, "Triglycerides", "mg/dL"),
    _num("LDL", G.LABORATORY, "LDL cholesterol", "mg/dL"),
    _num("HDL", G.LABORATORY, "HDL cholesterol", "mg/dL"),
    _num("BUN", G.LABORATORY, "Blood urea nitrogen", "mg/dL"),
    _num("ESR", G.LABORATORY, "ESR", "mm/h"),
    _num("HB", G.LABORATORY, "Hemoglobin", "g/dL"),
    _num("K", G.LABORATORY, "Potassium", "mEq/L"),
    _num("Na", G.LABORATORY, "Sodium", "mEq/L"),
    _num("WBC", G.LABORATORY, "White blood cells", "cells/µL"),
    _num("Lymph", G.LABORATORY, "Lymphocytes", "%"),
    _num("Neut", G.LABORATORY, "Neutrophils", "%"),
    _num("PLT", G.LABORATORY, "Platelets", "×10³/µL"),
    # Echocardiography
    _num("EF-TTE", G.ECHO, "Ejection fraction (TTE)", "%"),
    Feature("Region RWMA", FeatureKind.ORDINAL, G.ECHO, "Regions with wall-motion abnormality", categories=(0, 1, 2, 3, 4)),
    Feature("VHD", FeatureKind.ORDINAL, G.ECHO, "Valvular heart disease", categories=("N", "mild", "Moderate", "Severe")),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
FEATURES_BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}


def features_of_kind(kind: FeatureKind) -> tuple[Feature, ...]:
    return tuple(f for f in FEATURES if f.kind is kind)


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Requirement 1d: target columns must never reach a model as inputs."""
    leaked = sorted(set(columns) & set(LEAKAGE_COLUMNS))
    if leaked:
        raise ValueError(f"Target leakage: {leaked} must not be used as model inputs")


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns exactly the registered input columns, in registry order."""
    missing = [name for name in FEATURE_NAMES if name not in df.columns]
    if missing:
        raise KeyError(f"Missing input features: {missing}")
    selected = df.loc[:, list(FEATURE_NAMES)]
    assert_no_leakage(selected.columns)
    return selected


if len(FEATURES_BY_NAME) != len(FEATURES):
    raise ValueError("Duplicate feature names in registry")
assert_no_leakage(FEATURE_NAMES)
