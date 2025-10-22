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

# Test RF03
def test_predicition_match_input(trained_rf_model):
    # Gets the predictions and checks their length
    rf_model, X_test_scaled = trained_rf_model
    predictions = rf_model.predict(X_test_scaled)
    assert predictions is not None
    assert len(predictions) == X_test_scaled.shape[0]