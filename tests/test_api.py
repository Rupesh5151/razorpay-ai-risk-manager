"""
Unit tests for Razorpay AI Risk Manager API
Run: pytest tests/ -v
"""

import pytest
import requests
import json

API_BASE = "http://127.0.0.1:8000"

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_transaction():
    return {
        "TransactionAmt": 250.0,
        "card1": 9500,
        "hour_of_day": 14,
        "day_of_week": 2,
        "is_night": 0,
        "is_weekend": 0,
        "is_cold_start": 0,
        "risky_email_domain": 0,
        "addr_mismatch": 0,
        "card1_vel_3600s": 3.0,
        "card1_vel_21600s": 8.0,
        "card1_vel_86400s": 15.0,
        "merchant_id": "test_merchant",
    }

@pytest.fixture
def cold_start_transaction():
    return {
        "TransactionAmt": 800.0,
        "is_cold_start": 1,
        "is_night": 1,
        "risky_email_domain": 0,
    }

@pytest.fixture
def high_risk_transaction():
    return {
        "TransactionAmt": 5000.0,
        "card1": 9999,
        "hour_of_day": 2,
        "day_of_week": 6,
        "is_night": 1,
        "is_weekend": 1,
        "is_cold_start": 0,
        "risky_email_domain": 1,
        "addr_mismatch": 1,
        "card1_vel_3600s": 15.0,
        "card1_vel_21600s": 40.0,
        "card1_vel_86400s": 80.0,
        "merchant_id": "test_merchant",
    }


# ── Health check ──────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self):
        r = requests.get(f"{API_BASE}/health")
        assert r.status_code == 200

    def test_health_has_model_version(self):
        r = requests.get(f"{API_BASE}/health")
        data = r.json()
        assert "model_ver" in data
        assert data["model_ver"] == "lgbm-v1.0"

    def test_health_has_thresholds(self):
        r = requests.get(f"{API_BASE}/health")
        data = r.json()
        assert "thresholds" in data
        t = data["thresholds"]
        assert "approve" in t
        assert "stepup" in t
        assert "decline" in t

    def test_health_razorpay_connected(self):
        r = requests.get(f"{API_BASE}/health")
        data = r.json()
        assert "razorpay_connected" in data
        assert data["razorpay_connected"] is True


# ── Score endpoint ────────────────────────────────────────────────────────────
class TestScore:
    def test_score_returns_200(self, sample_transaction):
        r = requests.post(f"{API_BASE}/score", json=sample_transaction)
        assert r.status_code == 200

    def test_score_response_schema(self, sample_transaction):
        r = requests.post(f"{API_BASE}/score", json=sample_transaction)
        data = r.json()
        assert "transaction_id" in data
        assert "p_fraud" in data
        assert "decision" in data
        assert "reasons" in data
        assert "path" in data
        assert "model_ver" in data
        assert "latency_ms" in data
        assert "audit" in data

    def test_score_p_fraud_in_range(self, sample_transaction):
        r = requests.post(f"{API_BASE}/score", json=sample_transaction)
        data = r.json()
        assert 0.0 <= data["p_fraud"] <= 1.0

    def test_score_decision_valid(self, sample_transaction):
        r = requests.post(f"{API_BASE}/score", json=sample_transaction)
        data = r.json()
        assert data["decision"] in ["APPROVE", "STEP_UP_2FA", "DECLINE"]

    def test_score_reasons_not_empty(self, sample_transaction):
        r = requests.post(f"{API_BASE}/score", json=sample_transaction)
        data = r.json()
        assert len(data["reasons"]) > 0

    def test_score_latency_under_200ms(self, sample_transaction):
        r = requests.post(f"{API_BASE}/score", json=sample_transaction)
        data = r.json()
        assert data["latency_ms"] < 200

    def test_score_uses_ml_model_for_warm(self, sample_transaction):
        r = requests.post(f"{API_BASE}/score", json=sample_transaction)
        data = r.json()
        assert data["path"] == "ML_MODEL"

    def test_score_missing_amount_returns_422(self):
        r = requests.post(f"{API_BASE}/score", json={"card1": 9500})
        assert r.status_code == 422

    def test_score_negative_amount_returns_422(self):
        r = requests.post(f"{API_BASE}/score",
                          json={"TransactionAmt": -100})
        assert r.status_code == 422


# ── Cold-start path ───────────────────────────────────────────────────────────
class TestColdStart:
    def test_cold_start_uses_rules_path(self, cold_start_transaction):
        r = requests.post(f"{API_BASE}/score", json=cold_start_transaction)
        data = r.json()
        assert data["path"] == "COLD_START"

    def test_cold_start_high_amount_night_declines(self):
        txn = {
            "TransactionAmt": 800.0,
            "is_cold_start": 1,
            "is_night": 1,
            "risky_email_domain": 0,
        }
        r = requests.post(f"{API_BASE}/score", json=txn)
        data = r.json()
        assert data["decision"] in ["STEP_UP_2FA", "DECLINE"]

    def test_cold_start_risky_email_high_risk(self):
        txn = {
            "TransactionAmt": 50.0,
            "is_cold_start": 1,
            "is_night": 0,
            "risky_email_domain": 1,
        }
        r = requests.post(f"{API_BASE}/score", json=txn)
        data = r.json()
        assert data["p_fraud"] >= 0.85


# ── Threshold routing ─────────────────────────────────────────────────────────
class TestThresholds:
    def test_high_risk_routes_to_decline_or_stepup(self,
                                                    high_risk_transaction):
        r = requests.post(f"{API_BASE}/score",
                          json=high_risk_transaction)
        data = r.json()
        assert data["decision"] in ["STEP_UP_2FA", "DECLINE"]
        assert data["p_fraud"] > 0.1

    def test_decision_consistent_with_p_fraud(self, sample_transaction):
        r = requests.post(f"{API_BASE}/score", json=sample_transaction)
        data = r.json()
        p = data["p_fraud"]
        decision = data["decision"]
        thresholds = requests.get(
            f"{API_BASE}/health").json()["thresholds"]

        if p < thresholds["approve"]:
            assert decision == "APPROVE"
        elif p < thresholds["stepup"]:
            assert decision == "STEP_UP_2FA"
        else:
            assert decision == "DECLINE"


# ── Audit endpoints ───────────────────────────────────────────────────────────
class TestAudit:
    def test_audit_stats_returns_200(self):
        r = requests.get(f"{API_BASE}/audit/stats")
        assert r.status_code == 200

    def test_audit_stats_has_required_fields(self):
        r = requests.get(f"{API_BASE}/audit/stats")
        data = r.json()
        assert "total_decisions" in data

    def test_audit_history_returns_200(self):
        r = requests.get(f"{API_BASE}/audit/history")
        assert r.status_code == 200

    def test_audit_history_is_list(self):
        r = requests.get(f"{API_BASE}/audit/history")
        data = r.json()
        assert isinstance(data, list)

    def test_score_creates_audit_record(self, sample_transaction):
        score_r = requests.post(f"{API_BASE}/score", json=sample_transaction)
        txn_id = score_r.json()["audit"]["txn_id"]

        history_r = requests.get(f"{API_BASE}/audit/history?limit=500")
        history = history_r.json()
        # Match on partial ID (first 12 chars)
        txn_ids = [h.get("audit_id", "")[:12] for h in history]
        assert txn_id[:12] in txn_ids


# ── Batch endpoint ────────────────────────────────────────────────────────────
class TestBatch:
    def test_batch_returns_200(self, sample_transaction):
        r = requests.post(f"{API_BASE}/batch",
                          json=[sample_transaction])
        assert r.status_code == 200

    def test_batch_summary_correct_count(self, sample_transaction):
        txns = [sample_transaction, sample_transaction]
        r = requests.post(f"{API_BASE}/batch", json=txns)
        data = r.json()
        assert data["summary"]["total"] == 2

    def test_batch_results_match_count(self, sample_transaction):
        txns = [sample_transaction] * 3
        r = requests.post(f"{API_BASE}/batch", json=txns)
        data = r.json()
        assert len(data["results"]) == 3