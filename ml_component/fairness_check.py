import numpy as np
import pandas as pd
from decision_tree import load_and_preprocess as dt_load_and_preprocess, train_decision_tree
from logical_regression import load_and_preprocess as lr_load_and_preprocess, train_logistic_regression
from random_forest import load_and_preprocess as rf_load_and_preprocess, train_random_forest

# Function to perform group fairness check
def group_fairness_check(y_pred, sensitive_features, group_names=None):
    results = {}
    unique_groups = np.unique(sensitive_features)
    for group in unique_groups:
        idx = np.where(np.array(sensitive_features) == group)[0]
        rate = np.mean(np.array(y_pred)[idx])
        name = group_names[group] if group_names and group in group_names else str(group)
        results[name] = rate
    # Prints out max difference between the groups
    if len(results) > 1:
        rates = list(results.values())
        diff = max(rates) - min(rates)
        print(f"Max difference in positive rates between groups: {diff:.3f}")
    return results

if __name__ == "__main__":
    group_names = {1: "Female", 2: "Male"}

    # Decision Tree
    X_train_dt, X_test_dt, y_train_dt, y_test_dt, cw_dict_dt = dt_load_and_preprocess()
    dt_model = train_decision_tree(X_train_dt, y_train_dt, cw_dict_dt)
    y_pred_dt = dt_model.predict(X_test_dt)
    # Checking sex as it is the sensitive feature
    sensitive_dt = X_test_dt["Sex"].values
    valid_mask_dt = (sensitive_dt == 1) | (sensitive_dt == 2)
    filtered_y_pred_dt = y_pred_dt[valid_mask_dt]
    filtered_sensitive_dt = sensitive_dt[valid_mask_dt]
    # Prints out fairness results
    print("\nFairness check for Decision Tree:")
    fairness_dt = group_fairness_check(filtered_y_pred_dt, filtered_sensitive_dt, group_names)
    for group, rate in fairness_dt.items():
        print(f"  {group}: {rate:.2f}")

    # Logistic Regression
    X_train_lr, X_test_lr, y_train_lr, y_test_lr, cw_dict_lr = lr_load_and_preprocess()
    lr_model = train_logistic_regression(X_train_lr, y_train_lr, cw_dict_lr)
    y_pred_lr = lr_model.predict(X_test_lr)
    # Need to get the whole original cleaned data and attributes because of onehot encoder
    df = pd.read_csv("../credit_dataset/cleaned_data.csv")
    from sklearn.model_selection import train_test_split
    feature_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
                    'Checking account', 'Credit amount', 'Duration', 'Purpose', 'Credit score', 'Income']
    X = df[feature_cols].copy()
    y = df['Risk']
    _, X_test_orig, _, _, = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    sensitive_lr = X_test_orig["Sex"]
    sensitive_lr = sensitive_lr.values
    valid_mask_lr = (sensitive_lr == 1) | (sensitive_lr == 2)
    filtered_y_pred_lr = y_pred_lr[valid_mask_lr]
    filtered_sensitive_lr = sensitive_lr[valid_mask_lr]
    # Prints out fairness results
    print("\nFairness check for Logistic Regression:")
    fairness_lr = group_fairness_check(filtered_y_pred_lr, filtered_sensitive_lr, group_names)
    for group, rate in fairness_lr.items():
        print(f"  {group}: {rate:.2f}")

    # Random Forest
    X_train_rf, X_test_rf, y_train_rf, y_test_rf, cw_dict_rf = rf_load_and_preprocess()
    rf_model = train_random_forest(X_train_rf, y_train_rf, cw_dict_rf)
    y_pred_rf = rf_model.predict(X_test_rf)
    valid_mask_rf = (sensitive_lr == 1) | (sensitive_lr == 2)
    filtered_y_pred_rf = y_pred_rf[valid_mask_rf]
    filtered_sensitive_rf = sensitive_lr[valid_mask_rf]
    # Prints out fairness results
    print("\nFairness check for Random Forest:")
    fairness_rf = group_fairness_check(filtered_y_pred_rf, filtered_sensitive_rf, group_names)
    for group, rate in fairness_rf.items():
        print(f"  {group}: {rate:.2f}")