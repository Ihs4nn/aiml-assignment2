import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

# Load the cleaned dataset for training
df = pd.read_csv("../credit_dataset/cleaned_data.csv")
feature_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
               'Checking account', 'Credit amount', 'Duration', 'Purpose']
X = df[feature_cols]
# Seperate target variable (what we want to predict)
y = df['Risk']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n")
# Training Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
# Printing Logistic Regression Model results
print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]))
print("PR_AUC:", average_precision_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]))
print("\n")
# Training Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
# Printing Random Forest Model results
print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
print("PR_AUC:", average_precision_score(y_test, rf.predict_proba(X_test)[:, 1]))
print("\n")
