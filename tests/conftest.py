import pytest
import pandas as pd
from unittest.mock import patch
import os

from ml_component.logical_regression import load_and_preprocess as lr_load_and_preprocess
from ml_component.random_forest import load_and_preprocess as rf_load_and_preprocess
from ml_component.decision_tree import load_and_preprocess as dt_load_and_preprocess
from ml_component.decision_tree import train_decision_tree as dt_train_decision_tree


@pytest.fixture
# Creating a valid CSV file for testing (ALL01 and ALL02)
def valid_csv_path(tmp_path):
    data = {
        # fake data simulating the 'balanced_data.csv' structure
        'Age': [25, 45, 35, 55, 22, 60, 41, 33],
        'Sex': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
        'Job': ['skilled', 'management', 'unskilled', 'skilled', 'skilled', 'management', 'unskilled', 'skilled'],
        'Housing': ['own', 'rent', 'own', 'free', 'own', 'rent', 'own', 'free'],
        'Saving accounts': ['little', 'rich', 'moderate', 'little', 'little', 'rich', 'moderate', 'little'],
        'Checking account': ['moderate', 'rich', 'little', 'moderate', 'moderate', 'rich', 'little', 'moderate'],
        'Credit amount': [2500, 5000, 3000, 7500, 1500, 10000, 4500, 2000],
        'Duration': [24, 36, 18, 48, 12, 60, 30, 15],
        'Purpose': ['radio/TV', 'car', 'furniture', 'business', 'education', 'car', 'repairs', 'radio/TV'],
        'Credit score': [700, 650, 720, 680, 750, 620, 710, 730],
        'Income': [50000, 75000, 45000, 90000, 40000, 120000, 60000, 55000],
        'Risk': ['Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad']
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "valid_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

# Creating an empty CSV file for testing (ALL03)
@pytest.fixture
def empty_csv_path(tmp_path):
    file_path = tmp_path / "empty_data.csv"
    file_path.touch()
    return str(file_path)

# Creating a CSV file with only one class in the target variable for testing (ALL04)
@pytest.fixture
def one_class_csv_path(tmp_path):
    """Creates a temporary CSV with only one class in the target variable."""
    data = {
        'Age': [25, 45, 35, 55], 'Sex': ['male', 'female', 'male', 'female'],
        'Job': ['skilled', 'management', 'unskilled', 'skilled'],
        'Housing': ['own', 'rent', 'own', 'free'],
        'Saving accounts': ['little', 'rich', 'moderate', 'little'],
        'Checking account': ['moderate', 'rich', 'little', 'moderate'],
        'Credit amount': [2500, 5000, 3000, 7500], 'Duration': [24, 36, 18, 48],
        'Purpose': ['radio/TV', 'car', 'furniture', 'business'],
        'Credit score': [700, 650, 720, 680], 'Income': [50000, 75000, 45000, 90000],
        # Risk only has 'Good' values
        'Risk': ['Good', 'Good', 'Good', 'Good'] 
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "one_class_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

# Creating a LR model fixture for testing (LR01, LR02, LR03)
@pytest.fixture
def trained_lr_model(valid_csv_path):
    # Trains the 'dummy' model and returns specific values for it
    with patch('pandas.read_csv', return_value=pd.read_csv(valid_csv_path)):
        X_train_scaled, X_test_scaled, y_train, cw_dict = lr_load_and_preprocess()
        # Mock joblib.dump to avoid file creation during tests
        with patch('joblib.dump'):
            lr_model = lr_load_and_preprocess(X_train_scaled, y_train, cw_dict)
        return lr_model, X_test_scaled
    
# Creating a RF model fixture for testing (RF01, RF02, RF03)
@pytest.fixture
def trained_rf_model(valid_csv_path):
    # Uses RF functions to train a model and return specific values for it
    with patch('pandas.read_csv', return_value=pd.read_csv(valid_csv_path)):
        X_train_scaled, X_test_scaled, y_train, cw_dict = rf_load_and_preprocess()
        # Mock joblib.dump to avoid file creation during tests
        with patch('joblib.dump'):
            rf_model = rf_load_and_preprocess(X_train_scaled, y_train, cw_dict)
        return rf_model, X_test_scaled

# Creating a DT model fixture for testing (DT01, DT02, DT03)
@pytest.fixture
def trained_dt_model(valid_csv_path):
    # Uses DT functions to train a model and return specific values for it
    with patch('pandas.read_csv', return_value=pd.read_csv(valid_csv_path)):
        X_train, X_test, y_train, _, cw_dict = dt_load_and_preprocess()
        # Mock joblib.dump to avoid file creation during tests
        with patch('joblib.dump'):
            dt_model = dt_train_decision_tree(X_train, y_train, cw_dict)
        return dt_model, X_test