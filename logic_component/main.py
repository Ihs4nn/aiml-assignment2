from statistics import mode

# customer_data in format:
# {
#   Age,
#   Sex,
#   Job,
#   Housing,
#   Saving accounts,
#   Checking account,
#   Credit amount,
#   Duration,
#   Purpose,
#   Credit score,
#   Income,
# }

example_customer_data = {
    "Age": 30,
    "Sex": 1, # 1 = male, 0 = female
    "Job": 1,
    "Housing": 1, # 0 = free, 1 = own, 2 = rent
    "Saving accounts": 3, # 0 = NA, 1 = little, 2 = moderate, 3 = quite rich, 4 = rich
    "Checking account": 2, # 0 = NA, 1 = little, 2 = moderate, 3 = quite rich, 4 = rich
    "Credit amount": 2000,
    "Duration": 6, # in months
    "Purpose": 1, # 0 = business, 1 = car, 2 = domestic appliances, 3 = education,
                  # 4 = furniture/equipment, 5 = radio/TV, 6 = repairs, 7 = vacation/others
    "Credit score": 701,
    "Income": 50000,
}


def reject(reason, applicant_id=None):
    """
    Rejects the loan application, prints the reason and returns a rejection message.

    Parameters:
        reason (str): Reason for application being rejected
        applicant_id (int): Applicant's ID

    Returns:
        result (dict): Dictionary with the rejected status, reason and applicant's ID
    """

    # For audit purposes, log the reason for rejection
    print(f"REJECTED Applicant {applicant_id if applicant_id else ""}")
    print(f"Reason: {reason}")

    return {
        "status": "Rejected",
        "reason": reason,
        "applicant_id": applicant_id
    }


def flag(reason, applicant_id=None):
    """
    Flags the loan application for manual review, prints the reason and returns a flag message.

    Parameters:
        reason (str): Reason for application being flagged for review
        applicant_id (int): Applicant's ID

    Returns:
        result (dict): Dictionary with the flagged for review status, reason and applicant's ID
    """

    # For audit purposes, log the reason for flagging for review
    print(f"FLAGGED Applicant {applicant_id if applicant_id else ""}")
    print(f"Reason: {reason}")
    
    return {
        "status": "Flagged for Review",
        "reason": reason,
        "applicant_id": applicant_id
    }


def approve(reason, applicant_id=None):
    """
    Approves the loan application, prints the reason and returns an approval message.

    Parameters:
        reason (str): Reason for application being approved
        applicant_id (int): Applicant's ID
        
    Returns:
        result (dict): Dictionary with the approved status, reason and applicant's ID
    """

    # For audit purposes, log the reason for approval
    print(f"APPROVED Applicant {applicant_id if applicant_id else ""}")
    print(f"Reason: {reason}")
    
    return {
        "status": "Approved",
        "reason": reason,
        "applicant_id": applicant_id
    }


def process(customer_data, ml_risk_scores):
    """
    Makes a decision on a customer's loan application using their data and 3 ML risk scores.

    Parameters:
        customer_data (dict): Dictionary of customer data
        ml_risk_scores (list): List of the 3 ML models' scores (0 being good risk, 1 being bad risk)

    Returns:
        result (dict): Dictionary with the status, reason and applicant's ID
    """

    age = customer_data.get("Age")
    sex = customer_data.get("Sex")
    num_jobs = customer_data.get("Job")
    housing_status = customer_data.get("Housing")
    savings_accounts = customer_data.get("Saving accounts")
    checking_account = customer_data.get("Checking account")
    credit_amount = customer_data.get("Credit amount")
    loan_duration = customer_data.get("Duration")
    purpose = customer_data.get("Purpose")
    credit_score = customer_data.get("Credit score")
    income = customer_data.get("Income")
    risk = mode(ml_risk_scores) # 0 = good risk, 1 = bad risk

    # strict rules
    if age < 18:
        return reject("Applicant must be at least 18 years old.")
    elif credit_score < 600:
        return reject("Credit score below minimum thresholds.")
    elif income < 20000:
        return reject("Income below minimum thresholds.")
    elif credit_amount > income * 5:
        return reject("Requested credit exceeds safe borrowing limit.")
    # scenarios that would need reviewing
    elif savings_accounts == 0 and checking_account == 0: # NA amount in both accounts
        return flag("No active bank accounts/balances so financial stability is unclear - review required.")
    elif loan_duration > 60:
        return flag("Loan duration exceeds maximum allowed term - review required.")
    elif num_jobs > 3:
        return flag("Frequent job changes indicates employment instability - review required.")
    # checking risk level
    elif risk == 1: # bad risk
        return reject("Application rejected due to high risk classification.")
    elif risk == 0: # good risk, therefore passes all the rules
        return approve("Application approved as it meets all criteria.")