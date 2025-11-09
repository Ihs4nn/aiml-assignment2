import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml_component')))
import joblib
import tempfile
import unittest
from unittest.mock import patch

import app

class TestLoanAppGUI(unittest.TestCase):
    def setUp(self):
        app.train_models()
        self.app = app.LoanAppGUI()
        self.app.update()

    def tearDown(self):
        self.app.destroy()

        directory = os.path.dirname(__file__)
        files_to_delete = [
            "decision_tree_model.pkl",
            "dt_label_encoders.pkl",
            "logistic_regression_model.pkl",
            "lr_cat_cols.pkl",
            "lr_non_cat_cols.pkl",
            "lr_ohe_encoder.pkl",
            "lr_scaler.pkl",
            "random_forest_model.pkl",
            "rf_cat_cols.pkl",
            "rf_non_cat_cols.pkl",
            "rf_ohe_encoder.pkl",
            "rf_scaler.pkl",
        ]
        # Manually delete each model/encoder file
        for file_path in files_to_delete:
            if os.path.exists(os.path.abspath(os.path.join(directory, file_path))):
                os.remove(os.path.abspath(os.path.join(directory, file_path)))

    # Test ST01
    def test_valid_application_approval(self):
        self.app.age_entry.insert(0, '30')
        self.app.sex_entry.set('1')
        self.app.job_entry.set('1')
        self.app.housing_entry.set('2')
        self.app.saving_entry.set('2')
        self.app.checking_entry.set('2')
        self.app.credit_entry.insert(0, '5000')
        self.app.duration_entry.insert(0, '24')
        self.app.purpose_entry.set('1')
        self.app.credit_score_entry.insert(0, '700')
        self.app.income_entry.insert(0, '30000')
        self.app.submit_application()
        result = self.app.result_label.get("1.0", "end")

        # Check that one of the 3 results are outputted
        self.assertTrue(any(status in result for status in ["Approved", "Rejected", "Flagged"]))
        # Check that a reason is output
        self.assertIn("Reason", result)

    # Test ST02
    def test_fail_age_requirement(self):
        # Applicant is under 18
        self.app.age_entry.insert(0, '17')
        # Normal values below
        self.app.sex_entry.set('2')
        self.app.job_entry.set('1')
        self.app.housing_entry.set('1')
        self.app.saving_entry.set('2')
        self.app.checking_entry.set('1')
        self.app.credit_entry.insert(0, '1000')
        self.app.duration_entry.insert(0, '6')
        self.app.purpose_entry.set('3')
        self.app.credit_score_entry.insert(0, '600')
        self.app.income_entry.insert(0, '20000')
        self.app.submit_application()
        result = self.app.result_label.get("1.0", "end")

        # Check that the applicant is rejected
        self.assertIn("Rejected", result)
        # Check that the valid reason is outputted
        self.assertIn("Applicant must be at least 18 years old.", result)

    # Test ST03
    def test_missing_required_field(self):
        # Leave age blank
        self.app.age_entry.insert(0, '')
        # Fill in rest of the fields normally
        self.app.sex_entry.set('1')
        self.app.job_entry.set('2')
        self.app.housing_entry.set('2')
        self.app.saving_entry.set('2')
        self.app.checking_entry.set('2')
        self.app.credit_entry.insert(0, '5000')
        self.app.duration_entry.insert(0, '24')
        self.app.purpose_entry.set('1')
        self.app.credit_score_entry.insert(0, '700')
        self.app.income_entry.insert(0, '30000')
        with patch('tkinter.messagebox.showerror') as mock_error:
            self.app.submit_application()
            mock_error.assert_called_once_with("Input Error", "Please enter valid data in all fields for submission")
