import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Function to load and preprocess the dataset
def load_and_preprocess():
    try:
        df = pd.read_csv("../credit_dataset/cleaned_data.csv")
    except FileNotFoundError:
        print("Error: 'cleaned_data.csv' not found.")
        exit()
    except pd.errors.EmptyDataError:
        raise ValueError("The provided CSV file is empty.")
    if df['Risk'].nunique() < 2:
        raise ValueError("The target variable 'Risk' must contain at least two unique classes for training.")

    feature_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
                   'Checking account', 'Credit amount', 'Duration', 'Purpose', 'Credit score', 'Income']
    X = df[feature_cols].copy()
    y = df['Risk']

    categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X[categorical_cols])
    X_noncat = X.drop(columns=categorical_cols)
    X_final = np.hstack([X_noncat.values, X_encoded])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.25, random_state=42, stratify=y)

    y_classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=y_classes, y=y_train)
    cw_dict = {cls: weight for cls, weight in zip(y_classes, class_weights)}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, cw_dict

# Function to train the random forest model
def train_random_forest(X_train_scaled, y_train, cw_dict):
    rf = RandomForestClassifier(class_weight=cw_dict, n_estimators=100, max_depth=7, random_state=42)
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, "random_forest_model.pkl")
    print("Model saved to 'random_forest_model.pkl'")
    return rf

# Function to print evaluation results
def print_results(rf, X_test_scaled, y_test):
    y_pred_rf = rf.predict(X_test_scaled)
    print("\nRandom Forest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1]))
    print("Feature Importances:", rf.feature_importances_)
    print("\n")

if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test, cw_dict = load_and_preprocess()
    rf = train_random_forest(X_train_scaled, y_train, cw_dict)
    print_results(rf, X_test_scaled, y_test)
