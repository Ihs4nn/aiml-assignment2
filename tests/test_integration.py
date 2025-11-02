import os
import joblib
import numpy as np
import pytest
from unittest.mock import patch

from ml_component.decision_tree import load_and_preprocess as dt_load_and_preprocess, train_decision_tree
from ml_component.random_forest import load_and_preprocess as rf_load_and_preprocess, train_random_forest
from ml_component.logical_regression import load_and_preprocess as lr_load_and_preprocess, train_logistic_regression

# Test IT01-03
@pytest.mark.parametrize("train_func, load_func, expected_files", [
    (
        train_decision_tree,
        dt_load_and_preprocess,
        ["decision_tree_model.pkl", "dt_label_encoders.pkl"]
    ),
    (
        train_random_forest,
        rf_load_and_preprocess,
        ["random_forest_model.pkl", "rf_ohe_encoder.pkl", "rf_scaler.pkl", "rf_non_cat_cols.pkl", "rf_cat_cols.pkl"]
    ),
    (
        train_logistic_regression,
        lr_load_and_preprocess,
        ["logistic_regression_model.pkl", "lr_ohe_encoder.pkl", "lr_scaler.pkl", "lr_non_cat_cols.pkl", "lr_cat_cols.pkl"]
    ),
])
def test_create_and_load_model_files(tmp_path, train_func, load_func, expected_files):
    
    created_files = []

    # Save the original function to avoid recursion
    original_joblib_dump = joblib.dump

    with patch("joblib.dump") as mock_dump:
        def fake_dump(obj, filename, *args, **kwargs):
            # If the filename is a string and ends with .pkl, redirect to tmp_path
            if isinstance(filename, str) and filename.endswith('.pkl'):
                # Ensure path is only tmp_path with just the file name (not the whole path from filename)
                redirected_path = tmp_path / os.path.basename(filename)
                created_files.append(redirected_path)
                # Use the original, unpatched joblib.dump and saves files to the temporary directory
                return original_joblib_dump(obj, redirected_path)
            # If it's not a .pkl file, just use the original
            return original_joblib_dump(obj, filename)

        mock_dump.side_effect = fake_dump
        # Intercept the below, where joblib.dump occurs, in the patch
        # Load and preprocess
        X_train, _, y_train, _, cw_dict = load_func()
        # Train the models
        model = train_func(X_train, y_train, cw_dict)

    # Check that all expected files exist in tmp_path and can be loaded
    for file_name in expected_files:
        file_path = tmp_path / file_name
        assert file_path.exists(), f"{file_name} was not created in test environment."

        loaded_obj = joblib.load(file_path)
        assert loaded_obj is not None, f"{file_name} could not be loaded with joblib."

# Test IT04
def test_decision_tree_prediction(trained_dt_model):
    dt_model, X_test = trained_dt_model
    y_pred = dt_model.predict(X_test)

    # Check that a prediction is returned
    assert y_pred is not None
    # Check that the number of predictions = number of samples in test dataset
    assert len(y_pred) == X_test.shape[0]
    # Check that predictions are in expected class set (0 or 1)
    assert set(np.unique(y_pred)).issubset({0, 1})

# Test IT05
def test_random_forest_prediction(trained_rf_model):
    rf_model, X_test = trained_rf_model
    y_pred = rf_model.predict(X_test)

    # Check that a prediction is returned
    assert y_pred is not None
    # Check that the number of predictions = number of samples in test dataset
    assert len(y_pred) == X_test.shape[0]
    # Check that predictions are in expected class set (0 or 1)
    assert set(np.unique(y_pred)).issubset({0, 1})

# Test IT06
def test_logistic_regression_prediction(trained_lr_model):
    lr_model, X_test = trained_lr_model
    y_pred = lr_model.predict(X_test)

    # Check that a prediction is returned
    assert y_pred is not None
    # Check that the number of predictions = number of samples in test dataset
    assert len(y_pred) == X_test.shape[0]
    # Check that predictions are in expected class set (0 or 1)
    assert set(np.unique(y_pred)).issubset({0, 1})