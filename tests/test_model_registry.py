from usability_teleop.modeling.registry import classification_model_specs, regression_model_specs


def test_regression_registry_has_ten_models() -> None:
    specs = regression_model_specs()
    assert len(specs) == 10
    assert specs[0].name == "LinearRegression"


def test_classification_registry_has_ten_models() -> None:
    specs = classification_model_specs()
    assert len(specs) == 10
