import pytest

from ml.config import CONSTANT_COLUMNS, LEAKAGE_COLUMNS
from ml.features import FEATURE_NAMES, FEATURES, Feature, FeatureGroup, FeatureKind, assert_no_leakage, select_features


def test_registry_covers_exactly_the_non_target_columns(raw_df):
    expected = [c for c in raw_df.columns if c not in LEAKAGE_COLUMNS + CONSTANT_COLUMNS]
    assert list(FEATURE_NAMES) == expected
    assert len(FEATURE_NAMES) == 54


@pytest.mark.parametrize("column", LEAKAGE_COLUMNS)
def test_leakage_guard_rejects_every_target_column(column):
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(["Age", column])


def test_select_features_drops_targets(raw_df):
    selected = select_features(raw_df)
    assert not set(selected.columns) & set(LEAKAGE_COLUMNS)


def test_select_features_reports_missing_inputs(raw_df):
    with pytest.raises(KeyError, match="Age"):
        select_features(raw_df.drop(columns=["Age"]))


def test_every_categorical_value_in_data_is_declared(raw_df):
    for feature in FEATURES:
        if feature.kind is not FeatureKind.CONTINUOUS:
            assert set(raw_df[feature.name].unique()) <= set(feature.categories), feature.name


def test_every_clinical_group_is_represented():
    assert {f.group for f in FEATURES} == set(FeatureGroup)


def test_feature_definition_is_validated():
    with pytest.raises(ValueError):
        Feature("X", FeatureKind.BINARY, FeatureGroup.ECG, "X", categories=("N",))


def test_every_observed_value_is_inside_its_clinical_bounds(raw_df):
    for feature in FEATURES:
        if feature.kind is FeatureKind.CONTINUOUS:
            low, high = feature.bounds
            assert raw_df[feature.name].between(low, high).all(), feature.name


def test_bounds_are_only_allowed_on_continuous_features():
    with pytest.raises(ValueError):
        Feature("X", FeatureKind.BINARY, FeatureGroup.ECG, "X", categories=("N", "Y"), bounds=(0, 1))
    with pytest.raises(ValueError):
        Feature("X", FeatureKind.CONTINUOUS, FeatureGroup.LABORATORY, "X", bounds=None)
