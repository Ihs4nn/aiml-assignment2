import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tkinter as tk
from tkinter import messagebox
from decision_tree import load_and_preprocess as dt_load_and_preprocess, train_decision_tree
from logic_component.main import process
from logical_regression import load_and_preprocess as lr_load_and_preprocess, train_logistic_regression
from random_forest import load_and_preprocess as rf_load_and_preprocess, train_random_forest
from decision_tree import dt_predict
from logical_regression import lr_predict
from random_forest import rf_predict


class LoanAppGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Loan Application Decision System")
        self.geometry("1000x1200")
        # Create input fields
        self.create_input_fields()
        # Create submit button
        self.submit_button = tk.Button(self, text="Submit Application", command=self.submit_application)
        self.submit_button.pack(pady=20)
        # Create result text:
        tk.Label(self, text="Application Result:").pack()
        self.result_label = tk.Text(self, height=15, width=100)
        self.result_label.pack()

    # Function to create input fields
    def create_input_fields(self):
        # Age field
        tk.Label(self, text="Age*:").pack()
        self.age_entry = tk.Entry(self)
        self.age_entry.pack()
        # Sex field
        tk.Label(self, text="Sex* (1=Male, 0=Female):").pack()
        self.sex_entry = tk.Entry(self)
        self.sex_entry.pack()
        # Job field
        tk.Label(self, text="Jobs* (0=Unemployed, 1=Employed, 2=Self-Employed):").pack()
        self.job_entry = tk.Entry(self)
        self.job_entry.pack()
        # Housing
        tk.Label(self, text="Housing* (0=Free, 1=Own, 2=Rent):").pack()
        self.housing_entry = tk.Entry(self)
        self.housing_entry.pack()
        # Saving accounts
        tk.Label(self, text="Saving Accounts* (0=None, 1=Low, 2=Medium, 3=High, 4=Very High):").pack()
        self.saving_entry = tk.Entry(self)
        self.saving_entry.pack()
        # Checking account
        tk.Label(self, text="Checking Account* (0=None, 1=Low, 2=Medium, 3=High, 4=Very High):").pack()
        self.checking_entry = tk.Entry(self)
        self.checking_entry.pack()
        # Credit amount
        tk.Label(self, text="Credit Amount*:").pack()
        self.credit_entry = tk.Entry(self)
        self.credit_entry.pack()
        # Duration
        tk.Label(self, text="Duration* (in months):").pack()
        self.duration_entry = tk.Entry(self)
        self.duration_entry.pack()
        # Purpose
        tk.Label(self, text="Purpose* (0=Business, 1=Car, 2=Domestic Appliances, 3=Education, 4=Furniture/Equipment, 5=Radio/TV, 6=Repairs, 7=Vacation/Others):").pack()
        self.purpose_entry = tk.Entry(self)
        self.purpose_entry.pack()
        # Credit Score
        tk.Label(self, text="Credit Score*:").pack()
        self.credit_score_entry = tk.Entry(self)
        self.credit_score_entry.pack()
        # Income
        tk.Label(self, text="Income*:").pack()
        self.income_entry = tk.Entry(self)
        self.income_entry.pack()
    
    # Function to get the data from input fields
    def get_customer_data(self):
        try:
            customer_data = {
                "Age": int(self.age_entry.get()),
                "Sex": int(self.sex_entry.get()),
                "Job": int(self.job_entry.get()),
                "Housing": int(self.housing_entry.get()),
                "Saving accounts": int(self.saving_entry.get()),
                "Checking account": int(self.checking_entry.get()),
                "Credit amount": float(self.credit_entry.get()),
                "Duration": int(self.duration_entry.get()),
                "Purpose": int(self.purpose_entry.get()),
                "Credit score": float(self.credit_score_entry.get()),
                "Income": float(self.income_entry.get())
            }
            return customer_data
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid data in all fields for submission")
            return None

    # Function to submit the application
    def submit_application(self):
        customer_data = self.get_customer_data()
        if customer_data is None:
            return
        # Getting predictions from all three models
        dt_score = dt_predict(customer_data)
        lr_score = lr_predict(customer_data)
        rf_score = rf_predict(customer_data)
        # Using logic component to get result
        ml_risk_scores = [dt_score, lr_score, rf_score]
        result = process(customer_data, ml_risk_scores)
        # Display the result in the result text area
        self.result_label.delete(1.0, tk.END)  # Clear previous result
        info = (
            f"Status: {result['status']}\n"
            f"Reason: {result['reason']}\n\n"
            f"ML Risk Scores:\n"
            f"  Decision Tree: {dt_score}\n" 
            f"  Logistic Regression: {lr_score}\n"
            f"  Random Forest: {rf_score}\n"
        )
        self.result_label.insert(tk.END, info)

# Function used to train ML models
def train_models():
    try:
        # Training the Decision Tree model to be used when submitting applications
        print("Training Decision Tree Model")
        X_train_dt, X_test_dt, y_train_dt, y_test_dt, cw_dict_dt = dt_load_and_preprocess()
        dt_model = train_decision_tree(X_train_dt, y_train_dt, cw_dict_dt)
        # Training the Logisitic Regression model to be used when submitting applications
        print("Training Logistic Regression Model")
        X_train_lr, X_test_lr, y_train_lr, y_test_lr, cw_dict_lr = lr_load_and_preprocess()
        lr_model = train_logistic_regression(X_train_lr, y_train_lr, cw_dict_lr)
        # Training the Random Forest model to be used when submitting applications
        print("Training Random Forest Model")
        X_train_rf, X_test_rf, y_train_rf, y_test_rf, cw_dict_rf = rf_load_and_preprocess()
        rf_model = train_random_forest(X_train_rf, y_train_rf, cw_dict_rf)
        print("All models trained and saved.")
    except Exception as e:
        print("Error during model training:", str(e))

if __name__ == "__main__":
    train_models()
    app = LoanAppGUI()
    app.mainloop()