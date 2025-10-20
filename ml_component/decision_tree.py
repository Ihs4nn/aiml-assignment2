import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

def predict_credit_risk(applicant_features):
    # Load the ML model
    ml_model = joblib.load("decision_tree_model.pkl")
    # Make a prediction based on the persons features
    prediction = ml_model.predict([applicant_features])
    # Return the predicition
    return int(prediction[0])

# Loading dataset and preprocessing
def load_and_preprocess():
    try:
        df = pd.read_csv("../credit_dataset/cleaned_data.csv")
    except FileNotFoundError:
        print("Error: 'cleaned_data.csv' not found.")
        exit()

    feature_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
                   'Checking account', 'Credit amount', 'Duration', 'Purpose', 'Credit score', 'Income']
    X = df[feature_cols].copy()
    y = df['Risk']

    categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    y_classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=y_classes, y=y_train)
    cw_dict = {cls: weight for cls, weight in zip(y_classes, class_weights)}

    X_train_scaled = X_train
    X_test_scaled = X_test

    return X_train_scaled, X_test_scaled, y_train, y_test, cw_dict

# Function to train the decision tree model and save it
def train_decision_tree(X_train_scaled, y_train, cw_dict):
    dt = DecisionTreeClassifier(class_weight=cw_dict, max_depth=5)
    dt.fit(X_train_scaled, y_train)
    joblib.dump(dt, "decision_tree_model.pkl")
    print("Model saved to 'decision_tree_model.pkl'")
    return dt

# Function to print evaluation results
def print_results(dt, X_test_scaled, y_test):
    y_pred_dt = dt.predict(X_test_scaled)
    print("\nDecision Tree Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_dt))
    print("ROC-AUC:", roc_auc_score(y_test, dt.predict_proba(X_test_scaled)[:, 1]))
    print("Tree Depth:", dt.get_depth())
    print("\n")

if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test, cw_dict = load_and_preprocess()
    dt = train_decision_tree(X_train_scaled, y_train, cw_dict)
    print_results(dt, X_test_scaled, y_test)

