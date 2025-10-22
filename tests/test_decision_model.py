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