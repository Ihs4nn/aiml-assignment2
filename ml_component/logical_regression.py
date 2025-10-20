import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Load the cleaned dataset for training
try:
    df = pd.read_csv("../credit_dataset/cleaned_data.csv")
except FileNotFoundError:
    print("Error: 'cleaned_data.csv' not found.")
    exit()

# Define features and target
feature_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
               'Checking account', 'Credit amount', 'Duration', 'Purpose', 'Credit score', 'Income']

X = df[feature_cols].copy()
# Seperate target variable (what we want to predict)
y = df['Risk']

# One-hot encode categorical features
categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])
X_noncat = X.drop(columns=categorical_cols)
X_final = np.hstack([X_noncat.values, X_encoded])

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.25, random_state=42, stratify=y)

# Compute class weights for imbalance
y_classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=y_classes, y=y_train)
cw_dict = {cls: weight for cls, weight in zip(y_classes, class_weights)}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n")
# Training Logistic Regression model
lr = LogisticRegression(C=1.0, class_weight=cw_dict, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
# Printing Logistic Regression Model results
print("Logistic Regression Results:")
print("ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]))
from sklearn.metrics import log_loss
print("Log-loss:", log_loss(y_test, lr.predict_proba(X_test_scaled)))
print("Model Coefficients:", lr.coef_)
print("\n")

# joblib.dump(lr, "logistic_regression_model.pkl")

# def predict_credit_risk(applicant_features):
#     # Load the ML model
#     ml_model = joblib.load("logistic_regression_model.pkl")
#     # Make a prediction based on the persons features
#     prediction = ml_model.predict([applicant_features])
#     # Return the predicition
#     return int(prediction[0])