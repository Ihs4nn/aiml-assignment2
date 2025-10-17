import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Load the cleaned dataset for training
df = pd.read_csv("../credit_dataset/cleaned_data.csv")

# Define features and target
feature_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
               'Checking account', 'Credit amount', 'Duration', 'Purpose', 'Credit score', 'Income']
X = df[feature_cols]
# Seperate target variable (what we want to predict)
y = df['Risk']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Compute class weights for imbalance
y_classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=y_classes, y=y_train)
cw_dict = {cls: weight for cls, weight in zip(y_classes, class_weights)}

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n")
# Training Logistic Regression model
lr = LogisticRegression(class_weight=cw_dict)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
# Printing Logistic Regression Model results
print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]))
print("PR_AUC:", average_precision_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]))
print("\n")

joblib.dump(lr, "logistic_regression_model.pkl")

def predict_credit_risk(applicant_features):
    # Load the ML model
    ml_model = joblib.load("logistic_regression_model.pkl")
    # Make a prediction based on the persons features
    prediction = ml_model.predict([applicant_features])
    # Return the predicition
    return int(prediction[0])