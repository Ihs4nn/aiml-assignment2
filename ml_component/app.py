import tkinter as tk
from tkinter import messagebox
# from random_forest import rf_predict
# from decision_tree import dt_predict
# from logical_regression import lr_predict

class LoanAppGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Loan Application Decision System")
        self.geometry("1000x650")
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
        if customer_data is not None:
            messagebox.showinfo("Processing", "Your application is being processed. Please wait...")
            print("Customer Data:", customer_data)
    
if __name__ == "__main__":
    app = LoanAppGUI()
    app.mainloop()