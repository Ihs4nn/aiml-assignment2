import numpy as np

# Test LR01
def test_check_coefficients(trained_lr_model):
    lr_model, _ = trained_lr_model
    # Check that the coefficients and intercept are not empty
    assert hasattr(lr_model, 'coef_')
    assert hasattr(lr_model, 'intercept_')
    assert lr_model.coef_.size > 0
    assert lr_model.intercept_.size > 0

# Test LR02
def test_prediction_match(trained_lr_model):
    # Gets the predictions and checks their length
    lr_model, X_test_scaled = trained_lr_model
    predictions = lr_model.predict(X_test_scaled)
    assert predictions is not None
    assert len(predictions) == X_test_scaled.shape[0]

# Test LR03
def test_lr_probability_output(trained_lr_model):
    # Gets the probability predictions and checks their shape
    lr_model, X_test_scaled = trained_lr_model
    prob_predictions = lr_model.predict_proba(X_test_scaled)
    # Checks that probabilities are between 0 and 1 and sum to 1 for each sample
    assert np.all((prob_predictions >= 0) & (prob_predictions <= 1))
    assert prob_predictions.shape[0] == X_test_scaled.shape[0]
    assert np.all(np.isclose(prob_predictions.sum(axis=1), 1.0))

