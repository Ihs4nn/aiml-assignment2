# Test LR01
def test_check_coefficients(trained_lr_model):
    lr_model, _ = trained_lr_model
    assert hasattr(lr_model, 'coef_')
    assert hasattr(lr_model, 'intercept_')
    assert lr_model.coef_.size > 0
    assert lr_model.intercept_.size > 0

# Test LR02