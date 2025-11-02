import os
import joblib
import numpy as np
import pytest
from unittest.mock import patch

from ml_component.decision_tree import load_and_preprocess as dt_load_and_preprocess, train_decision_tree
from ml_component.random_forest import load_and_preprocess as rf_load_and_preprocess, train_random_forest
from ml_component.logical_regression import load_and_preprocess as lr_load_and_preprocess, train_logistic_regression

# Test IT01-03
@pytest.mark.parametrize("train_func, model_filename", [
    (train_decision_tree, "decision_tree_model.pkl"),
    (train_random_forest, "random_forest_model.pkl"),
    (train_logistic_regression, "logistic_regression_model.pkl"),
])
def test_create_model_files(tmp_path, train_func, model_filename):
    # Prepare dummy data for training
    if "decision_tree" in model_filename:
        X_train, _, y_train, _, cw_dict = dt_load_and_preprocess()
    elif "random_forest" in model_filename:
        X_train, _, y_train, _, cw_dict = rf_load_and_preprocess()
    else:
        X_train, _, y_train, _, cw_dict = lr_load_and_preprocess()

    # Patch joblib.dump to write to tmp_path
    model_path = tmp_path / model_filename

    # Save the original function to avoid recursion
    original_joblib_dump = joblib.dump

    with patch("joblib.dump") as mock_dump:
        def fake_dump(obj, filename, *args, **kwargs):
            # Use the original, unpatched joblib.dump
            # Save the model to the temporary directory
            return original_joblib_dump(obj, model_path)
        mock_dump.side_effect = fake_dump

        model = train_func(X_train, y_train, cw_dict)

    # Now check that the file exists in the temp directory
    assert model_path.exists(), f"Model file {model_path} was not created in test environment."
