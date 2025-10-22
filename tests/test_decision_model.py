import pandas as pd
import numpy as np

# Test DT01
def test_correct_encoding(trained_dt_model):
    _, X_test = trained_dt_model
    # Check that categorical features are encoded as integers
    categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in categorical_cols:
        assert pd.api.types.is_numeric_dtype(X_test[col])

# Test DT02
def test_tree_depth(trained_dt_model):
    dt_model, _ = trained_dt_model
    # Check that the tree depth is as expected
    assert hasattr(dt_model, 'get_depth')
    assert dt_model.max_depth == 5
    assert dt_model.get_depth() <= 5

# Test DT03
def test_feature_knowledge(trained_dt_model):
    dt_model, X_test = trained_dt_model
    # Checks that the model knows how many features it was trained on
    assert hasattr(dt_model, 'n_features_in_')
    assert dt_model.n_features_in_ == X_test.shape[1]