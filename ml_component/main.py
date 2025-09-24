import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import LogisticRegression
from sklearn.tree import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load the cleaned dataset for training
df = pd.read_csv("credit_dataset/cleaned_data.csv")
feature_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
               'Checking account', 'Credit amount', 'Duration', 'Purpose']
X = df[feature_cols]
# Seperate target variable (what we want to predict)
y = df['Risk']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
