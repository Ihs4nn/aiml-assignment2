import sys
import os
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_component.logical_regression import LogisticRegression, X_train_scaled, y_train

def test_logistic_regression_training():
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    assert hasattr(model, "coef_")