"""The deployable unit for one target: preprocessing + classifier pipeline + probability calibrator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.calibration import PlattCalibrator
from ml.features import select_features


@dataclass
class RiskModel:
    pipeline: Pipeline
    calibrator: PlattCalibrator

    @property
    def feature_names_in_(self) -> np.ndarray:
        return self.pipeline.feature_names_in_

    def raw_probability(self, patients: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(select_features(patients))[:, 1]

    def predict_proba(self, patients: pd.DataFrame) -> np.ndarray:
        """Calibrated P(positive) per patient, shape (n,)."""
        return self.calibrator(self.raw_probability(patients))
