import pandas as pd

dataset = pd.read_csv("raw_data/orginial_data.csv")

# Replace all NA values with 'Test' for testing purposes
dataset.replace('NA', 'Unknown', inplace=True)
dataset.to_csv("modified_data.csv", index=False)
print("Cleaned data saved to modified_data.csv")

