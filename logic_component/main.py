from statistics import mode

# customer_data in format:
# {
#   id,
#   age,
#   sex,
#   num_of_jobs,
#   housing,
#   savings accounts,
#   checking account,
#   credit amount,
#   loan_duration,
#   purpose,
#   credit score,
#   income,
#   risk
# }

example_customer_data = {
    "id": 1,
    "age": 30,
    "sex": 1, # 1 = male, 0 = female
    "num_of_jobs": 1,
    "housing": 1, # 0 = free, 1 = own, 2 = rent
    "savings_accounts": 3, # 0 = NA, 1 = little, 2 = moderate, 3 = quite rich, 4 = rich
    "checking_account": 2, # 0 = NA, 1 = little, 2 = moderate, 3 = quite rich, 4 = rich
    "credit_amount": 2000,
    "loan_duration": 6, # in months
    "purpose": 1, # 0 = business, 1 = car, 2 = domestic appliances, 3 = education,
                  # 4 = furniture/equipment, 5 = radio/TV, 6 = repairs, 7 = vacation/others
    "credit_score": 701,
    "income": 50000,
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
    
    id = customer_data["id"]
    age = customer_data["age"]
    sex = customer_data["sex"]
    num_jobs = customer_data["num_of_jobs"]
    housing_status = customer_data["housing"]
    savings_accounts = customer_data["savings_accounts"]
    checking_account = customer_data["checking_account"]
    credit_amount = customer_data["credit_amount"]
    loan_duration = customer_data["loan_duration"]
    purpose = customer_data["purpose"]
    credit_score = customer_data["credit_score"]
    income = customer_data["income"]
    risk = mode(ml_risk_scores) # 0 = good risk, 1 = bad risk

    # strict rules
    if age < 18:
        return reject("Applicant must be at least 18 years old.")
    elif credit_score < 600 and income < 20000:
        return reject("Credit score and income below minimum thresholds.")
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