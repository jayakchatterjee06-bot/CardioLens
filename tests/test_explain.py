import pytest

from ml.calibration import PlattCalibrator
from ml.data_loader import binary_target
from ml.estimators import CANDIDATES, build_pipeline
from ml.explain import RiskModelExplainer
from ml.features import FEATURE_NAMES
from ml.risk_model import RiskModel

CALIBRATORS = {"identity": PlattCalibrator(), "stretching": PlattCalibrator(slope=1.8, intercept=-0.4)}


@pytest.fixture(scope="module")
def background(X):
    return X.sample(n=50, random_state=0)


@pytest.mark.parametrize("calibrator_name", sorted(CALIBRATORS))
@pytest.mark.parametrize("model_name", sorted(CANDIDATES))
def test_contributions_add_up_to_the_calibrated_prediction(model_name, calibrator_name, raw_df, X, background):
    pipeline = build_pipeline(model_name).fit(X, binary_target(raw_df, "lad"))
    model = RiskModel(pipeline, CALIBRATORS[calibrator_name])
    patient = X.iloc[[7]]
    explanation = RiskModelExplainer(model, background).explain(patient)

    assert len(explanation.contributions) == len(FEATURE_NAMES)
    total = explanation.base_value + sum(c.contribution for c in explanation.contributions)
    assert total == pytest.approx(explanation.probability, abs=5e-3)
    assert explanation.probability == pytest.approx(model.predict_proba(patient)[0])


def test_contributions_report_raw_values_sorted_by_magnitude(raw_df, X, background):
    model = RiskModel(build_pipeline("logistic_regression").fit(X, binary_target(raw_df, "cad")), PlattCalibrator())
    explanation = RiskModelExplainer(model, background).explain(X.iloc[[0]])

    by_feature = {c.feature: c for c in explanation.contributions}
    assert by_feature["Age"].value == X.iloc[0]["Age"]  # unscaled
    magnitudes = [abs(c.contribution) for c in explanation.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explain_rejects_batches(raw_df, X, background):
    model = RiskModel(build_pipeline("logistic_regression").fit(X, binary_target(raw_df, "cad")), PlattCalibrator())
    with pytest.raises(ValueError):
        RiskModelExplainer(model, background).explain(X.iloc[:2])
