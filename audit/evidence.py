"""
Chargeback Evidence Pack Generator
Generates PDF evidence packs for disputed transactions
"""

import json
import os
from datetime import datetime, timezone


def generate_evidence_text(record: dict) -> str:
    """
    Generate plain-text evidence report for a transaction decision.
    Can be used as PDF content or standalone text file.

    Args:
        record: Audit log record dict

    Returns:
        Formatted evidence string
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    decision    = record.get("decision", "UNKNOWN")
    p_fraud     = record.get("p_fraud", 0)
    reasons     = record.get("reasons", [])
    amount      = record.get("amount", 0)
    merchant_id = record.get("merchant_id", "N/A")
    txn_id      = record.get("transaction_id", "N/A")
    audit_id    = record.get("audit_id", "N/A")
    model_ver   = record.get("model_version", "N/A")
    latency     = record.get("latency_ms", "N/A")
    path        = record.get("path", "N/A")
    timestamp   = record.get("timestamp", "N/A")

    reasons_text = "\n".join(
        f"  {i+1}. {r}" for i, r in enumerate(reasons)
    ) if reasons else "  N/A"

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           RAZORPAY AI RISK MANAGER                           ║
║           CHARGEBACK EVIDENCE PACK                           ║
╚══════════════════════════════════════════════════════════════╝

Generated    : {now}
Report Type  : Fraud Decision Evidence

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSACTION DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transaction ID   : {txn_id}
Audit ID         : {audit_id}
Timestamp        : {timestamp}
Amount           : INR {amount:,.2f}
Merchant ID      : {merchant_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FRAUD DECISION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Decision         : {decision}
Fraud Score      : {p_fraud:.4f} (range: 0.0 = safe, 1.0 = fraud)
Detection Path   : {path}
Latency          : {latency}ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK FACTORS (SHAP EXPLANATION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Top risk factors that contributed to this decision:

{reasons_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model Version    : {model_ver}
Algorithm        : LightGBM + Isotonic Calibration
Features Used    : 451 (V, C, D, M, id cols + engineered)
Explainability   : TreeSHAP

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THRESHOLD CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPROVE    : P(Fraud) < 0.10
STEP_UP    : 0.10 <= P(Fraud) < 0.35
DECLINE    : P(Fraud) >= 0.35

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This evidence pack was generated automatically by the Razorpay
AI Risk Manager system. The fraud score and decision are based
on machine learning analysis of transaction features.
This document is for internal use and chargeback dispute only.

══════════════════════════════════════════════════════════════
END OF EVIDENCE PACK
══════════════════════════════════════════════════════════════
"""
    return report.strip()


def save_evidence_pack(record: dict,
                       output_dir: str = "audit/evidence_packs") -> str:
    """
    Save evidence pack as text file

    Args:
        record: Audit log record
        output_dir: Directory to save evidence packs

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    txn_id    = record.get("transaction_id", "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename  = f"evidence_{txn_id[:12]}_{timestamp}.txt"
    filepath  = os.path.join(output_dir, filename)

    content = generate_evidence_text(record)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def generate_evidence_from_audit_log(
        transaction_id: str,
        log_file: str = "audit/decisions.jsonl") -> str:
    """
    Generate evidence pack directly from audit log

    Args:
        transaction_id: Transaction ID to look up
        log_file: Path to audit log

    Returns:
        Evidence pack text, or error message
    """
    try:
        with open(log_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("transaction_id") == transaction_id:
                    return generate_evidence_text(record)
        return f"Transaction {transaction_id} not found in audit log."
    except FileNotFoundError:
        return "Audit log not found."
    except Exception as e:
        return f"Error generating evidence: {e}"