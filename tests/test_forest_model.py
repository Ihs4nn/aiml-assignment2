import numpy as np

# Test RF01
def test_check_feature_output(trained_rf_model):
    rf_model, _ = trained_rf_model
    # Check that the feature importances are not empty
    assert hasattr(rf_model, 'feature_importances_')
    assert rf_model.feature_importances_.size > 0
    # Check that the sum of importances is around 1
    assert np.isclose(rf_model.feature_importances_.sum(), 1.0)

# Test RF02
def test_rf_estimators_output(trained_rf_model):
    rf_model, _ = trained_rf_model
    # Check that the number of estimators is as expected and length is equal
    assert hasattr(rf_model, 'n_estimators')
    assert rf_model.n_estimators == 100
    assert len(rf_model.estimators_) == 100