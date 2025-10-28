import pytest
from statistics import mode
from logic_component.main import reject, flag, approve, process

@pytest.fixture
def base_customer():
    return {
        "Age": 30,
        "Sex": 1,
        "Job": 1,
        "Housing": 1,
        "Saving Accounts": 3,
        "Checking account": 2,
        "Credit amount": 2000,
        "Duration": 6,
        "Purpose": 1,
        "Credit score": 701,
        "Income": 50000,
    }


@pytest.fixture
def good_ml_risk_scores():
    return [0,0,0]


@pytest.fixture
def bad_ml_risk_scores():
    return [1,1,1]


class TestReject:
    status = "Rejected"

    # Test R01
    def test_reject_reason_and_id(self):
        reason = "Applicant must be at least 18 years old."
        applicant_id = 1

        result = reject(reason, applicant_id)

        assert result["status"] == self.status, "status does not match expected value: 'Rejected'"
        assert result["reason"] == reason, "reason does not match expected value"
        assert result["applicant_id"] == applicant_id, "applicant ID does not match expected value"

    # Test R02
    def test_reject_reason_only(self):
        reason = "Credit score and income below minimum thresholds."

        result = reject(reason)

        assert result["status"] == self.status, "status does not match expected value: 'Rejected'"
        assert result["reason"] == reason, "reason does not match expected value"
        assert result["applicant_id"] is None, "applicant ID does not match expected value"


class TestFlag:
    status = "Flagged for Review"

    # Test F01
    def test_flag_reason_and_id(self):
        reason = "No active bank accounts/balances so financial stability is unclear - review required."
        applicant_id = 1

        result = flag(reason, applicant_id)

        assert result["status"] == self.status, "status does not match expected value: 'Flagged for Review'"
        assert result["reason"] == reason, "reason does not match expected value"
        assert result["applicant_id"] == applicant_id, "applicant ID does not match expected value"

    # Test F02
    def test_flag_reason_only(self):
        reason = "Loan duration exceeds maximum allowed term - review required."

        result = flag(reason)

        assert result["status"] == self.status, "status does not match expected value: 'Flagged for Review'"
        assert result["reason"] == reason, "reason does not match expected value"
        assert result["applicant_id"] is None, "applicant ID does not match expected value"


class TestApprove:
    status = "Approved"

    # Test A01
    def test_approve_reason_and_id(self):
        reason = "Application approved as it meets all criteria."
        applicant_id = 1

        result = approve(reason, applicant_id)

        assert result["status"] == self.status, "status does not match expected value: 'Approved'"
        assert result["reason"] == reason, "reason does not match expected value"
        assert result["applicant_id"] == applicant_id, "applicant ID does not match expected value"

    # Test A02
    def test_approve_reason_only(self):
        reason = "Application approved as it meets all criteria."

        result = approve(reason)

        assert result["status"] == self.status, "status does not match expected value: 'Approved'"
        assert result["reason"] == reason, "reason does not match expected value"
        assert result["applicant_id"] is None, "applicant ID does not match expected value"


class TestMLRiskScores:
    # Test RS01
    def test_mode_all_bad_risk(self):
        ml_risk_scores = [1,1,1]

        risk = mode(ml_risk_scores)
        assert risk == 1, "risk should be 1 if all three models output risks of 1"

    # Test RS02
    def test_mode_all_good_risk(self):
        ml_risk_scores = [0,0,0]

        risk = mode(ml_risk_scores)
        assert risk == 0, "risk should be 0 if all three models output risks of 0"

    # Test RS03
    def test_mode_majority_bad_risk(self):
        ml_risk_scores = [1,1,0]

        risk = mode(ml_risk_scores)
        assert risk == 1, "risk should be 1 if two of three models output risks of 1"

    # Test RS04
    def test_mode_majority_good_risk(self):
        ml_risk_scores = [0,0,1]

        risk = mode(ml_risk_scores)
        assert risk == 0, "risk should be 0 if two of three models output risks of 0"


class TestProcess:
    rejected = "Rejected"
    flagged = "Flagged for Review"
    approved = "Approved"

    # application rejections ------------------------------------------------------------------------------------

    # Test PR01
    def test_process_reject_underage(self, base_customer, good_ml_risk_scores):
        # make customer underage
        base_customer["Age"] = 17

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.rejected, "status does not match expected value: 'Rejected'"
        assert "at least 18 years old" in result["reason"], "reason does not include expected value: 'at least 18 years old'"

    # Test PR02
    def test_process_reject_low_credit(self, base_customer, good_ml_risk_scores):
        # lowering credit score
        base_customer["Credit score"] = 500

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.rejected, "status does not match expected value: 'Rejected'"
        assert "Credit score below minimum" in result["reason"], "reason does not match expected value: 'Credit score below minimum'"
    
    # Test PR03
    def test_process_reject_low_income(self, base_customer, good_ml_risk_scores):
        # lowering income
        base_customer["Income"] = 15000

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.rejected, "status does not match expected value: 'Rejected'"
        assert "Income below minimum" in result["reason"], "reason does not match expected value: 'Income below minimum'"

    # Test PR04
    def test_process_reject_credit_amount_too_high(self, base_customer, good_ml_risk_scores):
        # increasing credit amount to be more than the £60k limit
        base_customer["Credit amount"] = 70000

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.rejected, "status does not match expected value: 'Rejected'"
        assert "exceeds safe borrowing limit" in result["reason"], "reason does not match expected value: 'exceeds safe borrowing limit'"

    # Test PR05
    def test_process_reject_bad_risk(self, base_customer, bad_ml_risk_scores):
        # use bad_ml_risk_scores fixture which outputs a risk of 1 (bad risk)
        result = process(base_customer, ml_risk_scores=bad_ml_risk_scores)

        assert result["status"] == self.rejected, "status does not match expected value: 'Rejected'"
        assert "high risk classification" in result["reason"], "reason does not match expected value: 'high risk classification'"

    # Test PR06
    def test_process_reject_majority_bad_risk(self, base_customer):
        # use good_ml_risk_scores fixture which outputs a risk of 0 (good risk)
        ml_risk_scores = [1,1,0]

        result = process(base_customer, ml_risk_scores)

        assert result["status"] == self.rejected, "status does not match expected value: 'Rejected'"
        assert "high risk classification" in result["reason"], "reason does not match expected value: 'high risk classification'"

    # application flags ------------------------------------------------------------------------------------

    # Test PF01
    def test_process_flag_no_bank_accounts(self, base_customer, good_ml_risk_scores):
        # zeroing bank account values
        base_customer["Saving accounts"] = 0
        base_customer["Checking account"] = 0

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.flagged, "status does not match expected value: 'Flagged for Review'"
        assert "No active bank accounts/balances" in result["reason"], "reason does not match expected value: 'No active bank accounts/balances'"

    # Test PF02
    def test_process_flag_long_loan_duration(self, base_customer, good_ml_risk_scores):
        # increasing loan duration
        base_customer["Duration"] = 61

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.flagged, "status does not match expected value: 'Flagged for Review'"
        assert "duration exceeds maximum allowed term" in result["reason"], "reason does not match expected value: 'duration exceeds maximum allowed term'"

    # Test PF03
    def test_process_flag_frequent_job_changes(self, base_customer, good_ml_risk_scores):
        # increasing number of jobs
        base_customer["Job"] = 4

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.flagged, "status does not match expected value: 'Flagged for Review'"
        assert "employment instability" in result["reason"], "reason does not match expected value: 'employment instability'"

    # application approvals ------------------------------------------------------------------------------------

    # Test PA01
    def test_process_approve_good_risk(self, base_customer, good_ml_risk_scores):
        # use good_ml_risk_scores fixture which outputs a risk of 0 (good risk)

        result = process(base_customer, ml_risk_scores=good_ml_risk_scores)

        assert result["status"] == self.approved, "status does not match expected value: 'Approved'"
        assert "meets all criteria" in result["reason"], "reason does not match expected value: 'meets all criteria'"

    # Test PA02
    def test_process_approve_majority_good_risk(self, base_customer):
        # use good_ml_risk_scores fixture which outputs a risk of 0 (good risk)
        ml_risk_scores = [0,0,1]

        result = process(base_customer, ml_risk_scores)

        assert result["status"] == self.approved, "status does not match expected value: 'Approved'"
        assert "meets all criteria" in result["reason"], "reason does not match expected value: 'meets all criteria'"