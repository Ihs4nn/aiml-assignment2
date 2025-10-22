import numpy as np

# Test RF01
def test_check_feature_output(trained_rf_model):
    rf_model, _ = trained_rf_model
    # Check that the feature importances are not empty
    assert hasattr(rf_model, 'feature_importances_')
    assert rf_model.feature_importances_.size > 0
    # Check that the sum of importances is around 1
    assert np.isclose(rf_model.feature_importances_.sum(), 1.0)