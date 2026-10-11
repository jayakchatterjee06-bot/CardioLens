"""Curated real patients from the dataset for the "load sample patient" buttons."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.config import TARGETS
from ml.data_loader import binary_target, load_raw_dataset
from ml.features import FEATURE_NAMES, select_features

TRAINING_DATA_NOTE = (
    "This patient is part of the training data, so the models are more confident about them "
    "than they would be about a new patient."
)


@dataclass(frozen=True)
class SampleCase:
    row: int  # positional row in the dataset
    id: str
    title: str
    description: str


SAMPLE_CASES: tuple[SampleCase, ...] = (
    SampleCase(8, "no-cad", "No coronary disease", "Normal angiogram: no significant stenosis in any vessel."),
    SampleCase(3, "borderline", "Borderline presentation", "Normal angiogram, but with risk features that make the models less certain."),
    SampleCase(2, "lad-only", "Single-vessel disease (LAD)", "Significant stenosis of the left anterior descending artery only."),
    SampleCase(11, "three-vessel", "Three-vessel disease", "Significant stenosis of the LAD, LCX and RCA."),
)


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value


def load_sample_patients(df: pd.DataFrame | None = None) -> list[dict]:
    df = load_raw_dataset() if df is None else df
    features = select_features(df)
    labels = {key: binary_target(df, key) for key in TARGETS}
    return [
        {
            "id": case.id,
            "title": case.title,
            "description": case.description,
            "note": TRAINING_DATA_NOTE,
            "ground_truth": {key: bool(labels[key].iloc[case.row]) for key in TARGETS},
            "patient": {name: _plain(features.iloc[case.row][name]) for name in FEATURE_NAMES},
        }
        for case in SAMPLE_CASES
    ]
