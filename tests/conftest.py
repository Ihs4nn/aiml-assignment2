import pytest
import pandas as pd
import os

@pytest.fixture
# Creating a temporary valid CSV file for testing
def valid_csv_path(tmp_path):
    data = {
        'Age': [25, 45], 'Sex': ['male', 'female'], 'Job': ['skilled', 'management'],
        'Housing': ['own', 'rent'], 'Saving accounts': ['little', 'rich'],
        'Checking account': ['moderate', 'rich'], 'Credit amount': [2500, 5000],
        'Duration': [24, 36], 'Purpose': ['radio/TV', 'car'],
        'Credit score': [700, 650], 'Income': [50000, 75000],
        'Risk': ['Good', 'Bad']
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "valid_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)