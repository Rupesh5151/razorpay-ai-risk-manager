"""
Unit tests for threshold logic and cold-start fallback
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Threshold routing logic ───────────────────────────────────────────────────
class TestThresholdLogic:

    APPROVE_THRESH = 0.10
    STEPUP_THRESH  = 0.35
    DECLINE_THRESH = 0.35

    def route(self, p):
        if p < self.APPROVE_THRESH:  return "APPROVE"
        if p < self.STEPUP_THRESH:   return "STEP_UP_2FA"
        return "DECLINE"

    def test_low_score_approves(self):
        assert self.route(0.05) == "APPROVE"

    def test_mid_score_step_up(self):
        assert self.route(0.20) == "STEP_UP_2FA"

    def test_high_score_declines(self):
        assert self.route(0.80) == "DECLINE"

    def test_boundary_approve_stepup(self):
        assert self.route(0.10) == "STEP_UP_2FA"

    def test_boundary_stepup_decline(self):
        assert self.route(0.35) == "DECLINE"

    def test_zero_score_approves(self):
        assert self.route(0.0) == "APPROVE"

    def test_one_score_declines(self):
        assert self.route(1.0) == "DECLINE"

    def test_all_decisions_covered(self):
        scores = [0.05, 0.20, 0.80]
        decisions = {self.route(s) for s in scores}
        assert decisions == {"APPROVE", "STEP_UP_2FA", "DECLINE"}


# ── Cold-start rules ──────────────────────────────────────────────────────────
class TestColdStartRules:

    COLD_RULES = {
        "amount_hard_limit":  500,
        "night_multiplier":   0.7,
        "risky_domain_block": True,
        "default_risk_score": 0.45,
    }
    APPROVE_THRESH = 0.10
    DECLINE_THRESH = 0.35

    def cold_start_decision(self, txn):
        reasons = []
        p = self.COLD_RULES["default_risk_score"]
        limit = self.COLD_RULES["amount_hard_limit"]
        if txn.get("is_night"):
            limit *= self.COLD_RULES["night_multiplier"]
        if txn.get("TransactionAmt", 0) > limit:
            p = max(p, 0.78)
            reasons.append("COLD_AMOUNT_EXCEEDS_LIMIT")
        if txn.get("risky_email_domain") and self.COLD_RULES["risky_domain_block"]:
            p = max(p, 0.85)
            reasons.append("COLD_RISKY_EMAIL_DOMAIN")
        if txn.get("is_night"):
            p = min(p + 0.10, 0.95)
            reasons.append("COLD_NIGHT_TRANSACTION")
        if not reasons:
            reasons.append("COLD_DEFAULT_PRIOR")
        if p < self.APPROVE_THRESH:    decision = "APPROVE"
        elif p < self.DECLINE_THRESH:  decision = "STEP_UP_2FA"
        else:                           decision = "DECLINE"
        return {"p_fraud": round(p, 4), "decision": decision, "reasons": reasons}

    def test_default_prior_is_decline(self):
        r = self.cold_start_decision({"TransactionAmt": 50})
        assert r["decision"] == "DECLINE"

    def test_large_amount_increases_risk(self):
        r = self.cold_start_decision({"TransactionAmt": 600})
        assert r["p_fraud"] >= 0.78

    def test_night_reduces_limit(self):
        r_day   = self.cold_start_decision({"TransactionAmt": 400, "is_night": 0})
        r_night = self.cold_start_decision({"TransactionAmt": 400, "is_night": 1})
        assert r_night["p_fraud"] >= r_day["p_fraud"]

    def test_risky_email_triggers(self):
        r = self.cold_start_decision({
            "TransactionAmt": 50,
            "risky_email_domain": 1
        })
        assert r["p_fraud"] >= 0.85
        assert "COLD_RISKY_EMAIL_DOMAIN" in r["reasons"]

    def test_night_reason_code(self):
        r = self.cold_start_decision({
            "TransactionAmt": 50,
            "is_night": 1
        })
        assert "COLD_NIGHT_TRANSACTION" in r["reasons"]

    def test_p_fraud_capped_at_095(self):
        r = self.cold_start_decision({
            "TransactionAmt": 9999,
            "is_night": 1,
            "risky_email_domain": 1
        })
        assert r["p_fraud"] <= 0.95

    def test_reasons_never_empty(self):
        r = self.cold_start_decision({"TransactionAmt": 10})
        assert len(r["reasons"]) > 0