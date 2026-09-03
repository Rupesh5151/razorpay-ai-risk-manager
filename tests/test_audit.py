"""
Unit tests for audit logger
"""

import pytest
import os
import json
import tempfile
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from audit.logger import AuditLogger


@pytest.fixture
def temp_logger(tmp_path):
    log_file = str(tmp_path / "test_decisions.jsonl")
    return AuditLogger(log_file=log_file)


@pytest.fixture
def sample_record():
    return {
        "audit_id":       "test_123",
        "transaction_id": "order_abc",
        "decision":       "STEP_UP_2FA",
        "p_fraud":        0.3443,
        "reasons":        ["HIGH_C14", "LOW_CARD1_MEAN_AMT"],
        "amount":         250.0,
        "merchant_id":    "merchant_001",
        "model_version":  "lgbm-v1.0",
        "latency_ms":     27.5,
        "path":           "ML_MODEL",
    }


class TestAuditLogger:

    def test_log_decision_returns_true(self, temp_logger, sample_record):
        result = temp_logger.log_decision(sample_record)
        assert result is True

    def test_log_creates_file(self, temp_logger, sample_record):
        temp_logger.log_decision(sample_record)
        assert os.path.exists(temp_logger.log_file)

    def test_logged_record_retrievable(self, temp_logger, sample_record):
        temp_logger.log_decision(sample_record)
        history = temp_logger.get_decision_history()
        assert len(history) == 1

    def test_logged_record_has_timestamp(self, temp_logger, sample_record):
        temp_logger.log_decision(sample_record)
        history = temp_logger.get_decision_history()
        assert "timestamp" in history[0]

    def test_logged_record_has_correct_decision(self, temp_logger, sample_record):
        temp_logger.log_decision(sample_record)
        history = temp_logger.get_decision_history()
        assert history[0]["decision"] == "STEP_UP_2FA"

    def test_multiple_records_logged(self, temp_logger, sample_record):
        temp_logger.log_decision(sample_record)
        temp_logger.log_decision(sample_record)
        temp_logger.log_decision(sample_record)
        history = temp_logger.get_decision_history()
        assert len(history) == 3

    def test_filter_by_transaction_id(self, temp_logger, sample_record):
        temp_logger.log_decision(sample_record)
        other = {**sample_record, "transaction_id": "other_xyz"}
        temp_logger.log_decision(other)
        history = temp_logger.get_decision_history(
            transaction_id="order_abc")
        assert all(h["transaction_id"] == "order_abc" for h in history)

    def test_empty_history_returns_list(self, temp_logger):
        history = temp_logger.get_decision_history()
        assert isinstance(history, list)
        assert len(history) == 0

    def test_stats_returns_dict(self, temp_logger, sample_record):
        temp_logger.log_decision(sample_record)
        stats = temp_logger.get_stats()
        assert isinstance(stats, dict)

    def test_log_is_append_only(self, temp_logger, sample_record):
        temp_logger.log_decision(sample_record)
        size1 = os.path.getsize(temp_logger.log_file)
        temp_logger.log_decision(sample_record)
        size2 = os.path.getsize(temp_logger.log_file)
        assert size2 > size1