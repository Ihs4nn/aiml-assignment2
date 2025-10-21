import pytest
import pandas as pd
from unittest.mock import patch

from ml_component.logical_regression import load_and_preprocess

# Test ID: ALL01
def test_successful_data_loading(valid_csv_path):
    with patch('pandas.read_csv', return_value=pd.read_csv(valid_csv_path)):
        X_train_scaled, X_test_scaled, y_train, y_test, cw_dict = load_and_preprocess()
        # Check that the returned values are as expected
        assert X_train_scaled.shape[0] > 0
        assert X_test_scaled.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert isinstance(cw_dict, dict)
        assert all(isinstance(key, (int, str)) for key in cw_dict.keys())
        assert all(isinstance(value, float) for value in cw_dict.values())

# Test ID: ALL02
def test_file_not_found():
    with patch('pandas.read_csv', side_effect=FileNotFoundError):
        # Checks that the function exits the program
        with pytest.raises(SystemExit):
            load_and_preprocess()

# Test ID: ALL03
def test_data_with_empty_csv(empty_csv_path):
    with patch('pandas.read_csv', return_value=pd.read_csv(empty_csv_path)):
        # Checks that it sends a ValueError because the data is empty
        with pytest.raises(ValueError):
            load_and_preprocess()

# Test ID: ALL04
def test_data_with_one_class(one_class_csv_path):
    with patch('pandas.read_csv', return_value=pd.read_csv(one_class_csv_path)):
        # Checks that it sends a ValueError because there is only one class in target variable
        with pytest.raises(ValueError):
            load_and_preprocess()