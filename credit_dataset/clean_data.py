import pandas as pd

dataset = pd.read_csv("/credit_dataset/raw_data/original_data.csv")

# Replace all NA values with 'Test' for testing purposes
dataset.replace('NA', 'Unknown', inplace=True)
dataset.to_csv("/credit_dataset/clean_data/cleaned_data.csv", index=False)
print("Cleaned data saved to /credit_dataset/clean_data/cleaned_data.csv")

