import pandas as pd
import random

# Load dataset
dataset = pd.read_csv("raw_data/original_data.csv")

# Add 'Credit score' and 'Income' columns if not present
if 'Credit score' not in dataset.columns:
    # Random credit score between 300 and 850 (typical range)
    dataset['Credit score'] = [random.randint(300, 850) for _ in range(len(dataset))]
if 'Income' not in dataset.columns:
    # Random income between 10,000 and 100,000
    dataset['Income'] = [random.randint(10000, 100000) for _ in range(len(dataset))]

# Goes through the dataset and replaces 'NA' values with -1
dataset.replace('NA', pd.NA, inplace=True)
dataset.fillna(-1, inplace=True)

# Hard coding risk values
if 'Risk' in dataset.columns:
    dataset['Risk'] = dataset['Risk'].map({'good': 0, 'bad': 1})
    dataset['Risk'] = dataset['Risk'].fillna(-1)

# With numerical columns keep them the same but if there is a non-numeric value then replace with -1
for col in ['Age', 'Job', 'Credit amount', 'Duration', 'Credit score', 'Income']:
    if col in dataset.columns:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce').fillna(-1)

# Convert all object/string columns to integer codes
for col in dataset.select_dtypes(include=['object']).columns:
    dataset[col] = dataset[col].astype('category').cat.codes

# Removes 'Unnamed' column
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# Saves the cleaned dataset to a new CSV file for ML training
dataset.to_csv("cleaned_data.csv", index=False)
print("Cleaned data saved to cleaned_data.csv")

