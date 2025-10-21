import pytest
from statistics import mode
from logic_component.main import reject, flag, approve

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