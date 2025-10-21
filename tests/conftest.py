import pytest
import pandas as pd
import os

@pytest.fixture
# Creating a temporary valid CSV file for testing
def valid_csv_path(tmp_path):
    data = {
        'Age': [25, 45, 35, 55],
        'Sex': ['male', 'female', 'male', 'female'],
        'Job': ['skilled', 'management', 'unskilled', 'skilled'],
        'Housing': ['own', 'rent', 'own', 'free'],
        'Saving accounts': ['little', 'rich', 'moderate', 'little'],
        'Checking account': ['moderate', 'rich', 'little', 'moderate'],
        'Credit amount': [2500, 5000, 3000, 7500],
        'Duration': [24, 36, 18, 48],
        'Purpose': ['radio/TV', 'car', 'furniture', 'business'],
        'Credit score': [700, 650, 720, 680],
        'Income': [50000, 75000, 45000, 90000],
        'Risk': ['Good', 'Bad', 'Good', 'Bad']
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "valid_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)