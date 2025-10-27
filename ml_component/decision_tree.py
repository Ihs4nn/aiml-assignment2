import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Function used to make predections with decision tree model
def dt_predict(input_data):
    try:
        model = joblib.load("decision_tree_model.pkl")
        encoders = joblib.load("dt_label_encoders.pkl")
    except FileNotFoundError:
        print("Error: model or encoders file not found, run the training scripts first")
        return None
    # Similar to testing style, create a dataframe for the input data
    feature_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts','Checking account', 'Credit amount', 'Duration', 'Purpose', 'Credit score', 'Income']
    df_model = pd.DataFrame([input_data], columns=feature_cols)
    for col in encoders:
        df_model[col] = encoders[col].transform(df_model[col].astype(str))
    df_model = df_model[feature_cols]
    prediction = model.predict(df_model)
    # Return the predicition for the integration component
    return int(prediction[0])
    
# Loading dataset and preprocessing
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

    encoders = {}
    categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Save the label encoders for prediciton function
    joblib.dump(encoders, "dt_label_encoders.pkl")
    print("Label encoders saved to 'dt_label_encoders.pkl'")

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

