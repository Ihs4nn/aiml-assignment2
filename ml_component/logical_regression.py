import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# --- NEW PREDICTION FUNCTION ---
def lr_predict(input_data):
    try:
        # Load model and preprocessors
        ml_model = joblib.load("logistic_regression_model.pkl")
        encoder = joblib.load("lr_ohe_encoder.pkl")
        scaler = joblib.load("lr_scaler.pkl")
        non_cat_cols = joblib.load("lr_non_cat_cols.pkl")
        cat_cols = joblib.load("lr_cat_cols.pkl")
    except FileNotFoundError:
        print("Error: model or encoders file not found, run the training scripts first")
        return None
    # Create a DataFrame from the input dictionary
    df_model = pd.DataFrame([input_data])
    # Apply preprocessing like we did in testing
    X_encoded = encoder.transform(df_model[cat_cols])
    X_noncat = df_model[non_cat_cols].values
    X_final = np.hstack([X_noncat, X_encoded])
    X_scaled = scaler.transform(X_final)
    prediction = ml_model.predict(X_scaled)
    # Return the predicition for the integration component
    return int(prediction[0])

# Loading dataset and preporocessing 
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

    # Define the feature columns to be used for prediction
    feature_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
                   'Checking account', 'Credit amount', 'Duration', 'Purpose', 'Credit score', 'Income']
    X = df[feature_cols].copy()
    y = df['Risk']
    
    # Categorical columns that need to be encoded
    categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X[categorical_cols])
    X_noncat = X.drop(columns=categorical_cols)
    # Save the encoder and column lists for prediction function
    non_cat_cols_list = X_noncat.columns.tolist()
    X_final = np.hstack([X_noncat.values, X_encoded])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.25, random_state=42, stratify=y)

    y_classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=y_classes, y=y_train)
    cw_dict = {cls: weight for cls, weight in zip(y_classes, class_weights)}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the preprocessing objects for prediction function
    joblib.dump(encoder, "lr_ohe_encoder.pkl")
    print("OneHotEncoder saved to 'lr_ohe_encoder.pkl'")
    joblib.dump(scaler, "lr_scaler.pkl")
    print("StandardScaler saved to 'lr_scaler.pkl'")
    joblib.dump(non_cat_cols_list, "lr_non_cat_cols.pkl")
    print("Non-categorical column list saved.")
    joblib.dump(categorical_cols, "lr_cat_cols.pkl")
    print("Categorical column list saved.")

    return X_train_scaled, X_test_scaled, y_train, y_test, cw_dict

# Function to train logistic regression model and save it
def train_logistic_regression(X_train_scaled, y_train, cw_dict):
    lr = LogisticRegression(C=1.0, class_weight=cw_dict, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    joblib.dump(lr, "logistic_regression_model.pkl")
    print("Model saved to 'logistic_regression_model.pkl'")
    return lr

# Function to print evaluation results
# def print_results(lr, X_test_scaled, y_test):
#     y_pred_lr = lr.predict(X_test_scaled)
#     print("\nLogistic Regression Results:")
#     print("Accuracy:", accuracy_score(y_test, y_pred_lr))
#     print("ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]))
#     print("Log-loss:", log_loss(y_test, lr.predict_proba(X_test_scaled)))
#     print("Model Coefficients:", lr.coef_)
#     print("Intercept:", lr.intercept_)
#     print("\n")

# Print results using the function
if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test, cw_dict = load_and_preprocess()
    lr = train_logistic_regression(X_train_scaled, y_train, cw_dict)
    print_results(lr, X_test_scaled, y_test)

