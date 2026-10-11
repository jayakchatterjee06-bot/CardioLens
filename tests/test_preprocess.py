import numpy as np

from ml.features import FEATURE_NAMES
from ml.preprocess import build_preprocessor, encoded_feature_sources


def test_preprocessor_output_is_finite_and_complete(X):
    encoded = build_preprocessor().fit_transform(X)
    assert encoded.shape == (len(X), 56)  # 54 features, BBB expands to 3 columns
    assert np.isfinite(encoded).all()


def test_every_encoded_column_maps_back_to_a_clinical_feature(X):
    preprocessor = build_preprocessor().fit(X)
    sources = encoded_feature_sources(preprocessor)
    assert len(sources) == len(preprocessor.get_feature_names_out())
    assert set(sources) == set(FEATURE_NAMES)


def test_binary_features_encode_to_zero_one(X):
    preprocessor = build_preprocessor().fit(X)
    names = list(preprocessor.get_feature_names_out())
    encoded = preprocessor.transform(X)
    for column in ("Sex", "Obesity", "DM"):
        assert set(np.unique(encoded[:, names.index(column)])) <= {0.0, 1.0}
