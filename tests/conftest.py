import pytest
import pandas as pd
import os

@pytest.fixture
# Creating a valid CSV file for testing
def valid_csv_path(tmp_path):
    data = {
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

# Creating a CSV file with only one class in the target variable for testing
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

# Creating an empty CSV file for testing
@pytest.fixture
def empty_csv_path(tmp_path):
    file_path = tmp_path / "empty_data.csv"
    file_path.touch()
    return str(file_path)