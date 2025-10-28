import tkinter as tk
from tkinter import messagebox
from ml_component.random_forest import rf_predict
from ml_component.decision_tree import dt_predict
from ml_component.logical_regression import lr_predict
from logic_component.main import process

class LoanAppGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Loan Application Decision System")
        self.geometry("400x600")
        # Create input fields
        self.create_input_fields()
        # Create submit button
        self.submit_button = tk.Button(self, text="Submit Application", command=self.submit_application)
        self.submit_button.pack(pady=20)

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
        