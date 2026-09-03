"""
Audit Logger
Immutable log of all fraud decisions for compliance and debugging
"""

import json
import os
from datetime import datetime, timedelta, timezone


class AuditLogger:
    """Write immutable audit trail to disk"""

    def __init__(self, log_file: str = "audit/decisions.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_decision(self, decision_record: dict) -> bool:
        """
        Log a fraud decision (append-only)
        """
        try:
            record = {
                "timestamp":      datetime.now(timezone.utc).isoformat(),
                "audit_id":       decision_record.get("audit_id"),
                "transaction_id": decision_record.get("transaction_id"),
                "decision":       decision_record.get("decision"),
                "p_fraud":        decision_record.get("p_fraud"),
                "reasons":        decision_record.get("reasons", []),
                "amount":         decision_record.get("amount"),
                "merchant_id":    decision_record.get("merchant_id"),
                "model_version":  decision_record.get("model_version"),
                "latency_ms":     decision_record.get("latency_ms"),
                "path":           decision_record.get("path"),
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            return True
        except Exception as e:
            print(f"Failed to log decision: {e}")
            return False

    def get_decision_history(self, transaction_id: str = None,
                             limit: int = 100) -> list:
        """
        Read decision history

        Args:
            transaction_id: Filter by transaction (None = all)
            limit: Max records to return

        Returns:
            List of decision records (most recent first)
        """
        try:
            records = []
            with open(self.log_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if (transaction_id is None or
                            record.get("transaction_id") == transaction_id):
                        records.append(record)
                    if len(records) >= limit:
                        break
            return list(reversed(records))
        except FileNotFoundError:
            return []

    def get_stats(self, hours: int = 24) -> dict:
        """Get decision statistics for last N hours"""
        try:
            records = self.get_decision_history(limit=10000)
            cutoff  = datetime.now(timezone.utc) - timedelta(hours=hours)

            recent = []
            for r in records:
                try:
                    ts = r.get("timestamp", "")
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt > cutoff:
                        recent.append(r)
                except Exception:
                    recent.append(r)

            return {
                "total_decisions": len(recent),
                "approve":  sum(1 for r in recent
                                if r.get("decision") == "APPROVE"),
                "step_up":  sum(1 for r in recent
                                if r.get("decision") == "STEP_UP_2FA"),
                "decline":  sum(1 for r in recent
                                if r.get("decision") == "DECLINE"),
                "avg_fraud_score": (
                    sum(r.get("p_fraud", 0) for r in recent) / len(recent)
                    if recent else 0.0
                ),
            }
        except Exception as e:
            return {
                "total_decisions": 0,
                "approve":         0,
                "step_up":         0,
                "decline":         0,
                "avg_fraud_score": 0.0,
                "error":           str(e),
            }