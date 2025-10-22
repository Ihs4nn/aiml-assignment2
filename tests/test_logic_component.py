import pytest
from statistics import mode
from logic_component.main import reject, flag, approve, process

@pytest.fixture
def base_customer():
    return {
        "id": 1,
        "age": 30,
        "sex": 1,
        "num_of_jobs": 1,
        "housing": 1,
        "savings_accounts": 3,
        "checking_account": 2,
        "credit_amount": 2000,
        "loan_duration": 6,
        "purpose": 1,
        "credit_score": 701,
        "income": 50000,
    }


@pytest.fixture
def good_ml_risk_scores():
    return [0,0,0]


@pytest.fixture
def bad_ml_risk_scores():
    return [1,1,1]


class TestReject:
    status = "Rejected"

    def test_reject_reason_and_id(self):
        reason = "Applicant must be at least 18 years old."
        applicant_id = 1

        result = reject(reason, applicant_id)

        assert result["status"] == self.status, "status does not match expected value: 'Rejected'"
        assert result["reason"] == reason, "reason does not match expected value"
        assert result["applicant_id"] == applicant_id, "applicant ID does not match expected value"


    def test_reject_reason_only(self):
        reason = "Credit score and income below minimum thresholds."

        result = reject(reason)

        assert result["status"] == self.status, "status does not match expected value: 'Rejected'"
        assert result["reason"] == reason, "reason does not match expected value"
        assert result["applicant_id"] is None, "applicant ID does not match expected value"


class TestFlag:
    status = "Flagged for Review"

    def test_flag_reason_and_id(self):
        reason = "No active bank accounts/balances so financial stability is unclear - review required."
        applicant_id = 1

        result = flag(reason, applicant_id)

        assert result["status"] == self.status, "status does not match expected value: 'Flagged for Review'"
        assert result["reason"] == reason, "reason does not match expected value"
        assert result["applicant_id"] == applicant_id, "applicant ID does not match expected value"


    def test_flag_reason_only(self):
        reason = "Loan duration exceeds maximum allowed term - review required."

        result = flag(reason)

        assert result["status"] == self.status, "status does not match expected value: 'Flagged for Review'"
        assert result["reason"] == reason, "reason does not match expected value"
        assert result["applicant_id"] is None, "applicant ID does not match expected value"


class TestApprove:
    status = "Approved"

    def test_approve_reason_and_id(self):
        reason = "Application approved as it meets all criteria."
        applicant_id = 1

        result = approve(reason, applicant_id)

        assert result["status"] == self.status, "status does not match expected value: 'Approved'"
        assert result["reason"] == reason, "reason does not match expected value"
        assert result["applicant_id"] == applicant_id, "applicant ID does not match expected value"


    def test_approve_reason_only(self):
        reason = "Application approved as it meets all criteria."

        result = approve(reason)

        assert result["status"] == self.status, "status does not match expected value: 'Approved'"
        assert result["reason"] == reason, "reason does not match expected value"
        assert result["applicant_id"] is None, "applicant ID does not match expected value"


class TestMLRiskScores:
    def test_mode_all_bad_risk(self):
        ml_risk_scores = [1,1,1]

        risk = mode(ml_risk_scores)
        assert risk == 1, "risk should be 1 if all three models output risks of 1"


    def test_mode_all_good_risk(self):
        ml_risk_scores = [0,0,0]

        risk = mode(ml_risk_scores)
        assert risk == 0, "risk should be 0 if all three models output risks of 0"


    def test_mode_majority_bad_risk(self):
        ml_risk_scores = [1,1,0]

        risk = mode(ml_risk_scores)
        assert risk == 1, "risk should be 1 if two of three models output risks of 1"


    def test_mode_majority_good_risk(self):
        ml_risk_scores = [0,0,1]

        risk = mode(ml_risk_scores)
        assert risk == 0, "risk should be 0 if two of three models output risks of 0"


class TestProcess:
    rejected = "Rejected"
    flagged = "Flagged for Review"
    approved = "Approved"

    # application rejections ------------------------------------------------------------------------------------

    def test_process_reject_underage(self, base_customer, good_ml_risk_scores):
        # make customer underage
        base_customer["age"] = 17

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.rejected, "status does not match expected value: 'Rejected'"
        assert "at least 18 years old" in result["reason"], "reason does not include expected value: 'at least 18 years old'"


    def test_process_reject_low_credit(self, base_customer, good_ml_risk_scores):
        # lowering credit score
        base_customer["credit_score"] = 500

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.rejected, "status does not match expected value: 'Rejected'"
        assert "Credit score below minimum" in result["reason"], "reason does not match expected value: 'Credit score below minimum'"
    
    
    def test_process_reject_low_income(self, base_customer, good_ml_risk_scores):
        # lowering income
        base_customer["income"] = 15000

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.rejected, "status does not match expected value: 'Rejected'"
        assert "Income below minimum" in result["reason"], "reason does not match expected value: 'Income below minimum'"


    def test_process_reject_credit_amount_too_high(self, base_customer, good_ml_risk_scores):
        # increasing credit amount to be more than 5 * income
        # base customer income = 50,000
        base_customer["credit_amount"] = 300000

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.rejected, "status does not match expected value: 'Rejected'"
        assert "exceeds safe borrowing limit" in result["reason"], "reason does not match expected value: 'exceeds safe borrowing limit'"


    def test_process_reject_bad_risk(self, base_customer, bad_ml_risk_scores):
        # use bad_ml_risk_scores fixture which outputs a risk of 1 (bad risk)
        result = process(base_customer, ml_risk_scores=bad_ml_risk_scores)

        assert result["status"] == self.rejected, "status does not match expected value: 'Rejected'"
        assert "high risk classification" in result["reason"], "reason does not match expected value: 'high risk classification'"


    def test_process_reject_majority_bad_risk(self, base_customer):
        # use good_ml_risk_scores fixture which outputs a risk of 0 (good risk)
        ml_risk_scores = [1,1,0]

        result = process(base_customer, ml_risk_scores)

        assert result["status"] == self.rejected, "status does not match expected value: 'Rejected'"
        assert "high risk classification" in result["reason"], "reason does not match expected value: 'high risk classification'"

    # application flags ------------------------------------------------------------------------------------

    def test_process_flag_no_bank_accounts(self, base_customer, good_ml_risk_scores):
        # zeroing bank account values
        base_customer["savings_accounts"] = 0
        base_customer["checking_account"] = 0

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.flagged, "status does not match expected value: 'Flagged for Review'"
        assert "No active bank accounts/balances" in result["reason"], "reason does not match expected value: 'No active bank accounts/balances'"


    def test_process_flag_long_loan_duration(self, base_customer, good_ml_risk_scores):
        # increasing loan duration
        base_customer["loan_duration"] = 61

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.flagged, "status does not match expected value: 'Flagged for Review'"
        assert "duration exceeds maximum allowed term" in result["reason"], "reason does not match expected value: 'duration exceeds maximum allowed term'"


    def test_process_flag_frequent_job_changes(self, base_customer, good_ml_risk_scores):
        # increasing number of jobs
        base_customer["num_of_jobs"] = 4

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.flagged, "status does not match expected value: 'Flagged for Review'"
        assert "employment instability" in result["reason"], "reason does not match expected value: 'employment instability'"

    # application approvals ------------------------------------------------------------------------------------

    def test_process_approve_good_risk(self, base_customer, good_ml_risk_scores):
        # use good_ml_risk_scores fixture which outputs a risk of 0 (good risk)

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.approved, "status does not match expected value: 'Approved'"
        assert "meets all criteria" in result["reason"], "reason does not match expected value: 'meets all criteria'"


    def test_process_approve_majority_good_risk(self, base_customer):
        # use good_ml_risk_scores fixture which outputs a risk of 0 (good risk)
        ml_risk_scores = [0,0,1]

        result = process(base_customer, ml_risk_scores)

        assert result["status"] == self.approved, "status does not match expected value: 'Approved'"
        assert "meets all criteria" in result["reason"], "reason does not match expected value: 'meets all criteria'"