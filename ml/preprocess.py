"""Builds the shared preprocessing step from the feature registry.

The transformer is always fitted *inside* a per-target sklearn Pipeline (see estimators.py),
so cross-validation refits it on each training fold and inference reuses the exact fitted
object saved with the model.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ml.features import FeatureKind, features_of_kind


def _names(kind: FeatureKind) -> list[str]:
    return [f.name for f in features_of_kind(kind)]


def _levels(kind: FeatureKind) -> list[list]:
    return [list(f.categories) for f in features_of_kind(kind)]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), _names(FeatureKind.CONTINUOUS)),
            ("binary", OrdinalEncoder(categories=_levels(FeatureKind.BINARY)), _names(FeatureKind.BINARY)),
            (
                "ordinal",
                Pipeline([
                    ("encode", OrdinalEncoder(categories=_levels(FeatureKind.ORDINAL))),
                    ("scale", StandardScaler()),
                ]),
                _names(FeatureKind.ORDINAL),
            ),
            (
                "nominal",
                OneHotEncoder(categories=_levels(FeatureKind.NOMINAL), sparse_output=False),
                _names(FeatureKind.NOMINAL),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def encoded_feature_sources(preprocessor: ColumnTransformer) -> list[str]:
    """For each encoded output column (in order), the raw clinical feature it came from.

    Used to fold SHAP values for one-hot columns back into a single clinical feature.
    """
    sources: list[str] = []
    for _, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or not len(columns):
            continue
        if isinstance(transformer, OneHotEncoder):
            for column, levels in zip(columns, transformer.categories_):
                sources.extend([column] * len(levels))
        else:
            sources.extend(columns)
    return sources
