import pandas as pd

dataset = pd.read_csv("raw_data/orginial_data.csv")

# Goes through the dataset and replaces 'NA' values with -1
for col in dataset.columns:
    dataset[col] = dataset[col].replace('NA', -1)

# Update 'Risk' column, missing values are kept as -1
if 'Risk' in dataset.columns:
    dataset['Risk'] = dataset['Risk'].map({'good': 0, 'bad': 1, -1: -1})

# Saves the cleaned dataset to a new CSV file for ML training
dataset.to_csv("cleaned_data.csv", index=False)
print("Cleaned data saved to cleaned_data.csv")

