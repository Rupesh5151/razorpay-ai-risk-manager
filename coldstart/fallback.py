"""
Cold-Start Fallback
Rule-based scoring for entities with < 10 transaction history
"""


COLD_RULES = {
    "amount_hard_limit":  500,
    "night_multiplier":   0.7,
    "risky_domain_block": True,
    "default_risk_score": 0.45,
}

APPROVE_THRESH = 0.10
DECLINE_THRESH = 0.35


def cold_start_decision(txn: dict) -> dict:
    """
    Rule-based fallback for cold-start entities.
    Returns same schema as ML inference for consistency.

    Args:
        txn: Transaction dict with TransactionAmt, is_night,
             risky_email_domain, etc.

    Returns:
        dict with p_fraud, decision, reasons, path
    """
    reasons = []
    p_fraud = COLD_RULES["default_risk_score"]

    # Rule 1: Large amount on unknown entity
    limit = COLD_RULES["amount_hard_limit"]
    if txn.get("is_night"):
        limit *= COLD_RULES["night_multiplier"]
    if txn.get("TransactionAmt", 0) > limit:
        p_fraud = max(p_fraud, 0.78)
        reasons.append("COLD_AMOUNT_EXCEEDS_LIMIT")

    # Rule 2: Risky email domain
    if txn.get("risky_email_domain") and COLD_RULES["risky_domain_block"]:
        p_fraud = max(p_fraud, 0.85)
        reasons.append("COLD_RISKY_EMAIL_DOMAIN")

    # Rule 3: Night-time transaction
    if txn.get("is_night"):
        p_fraud = min(p_fraud + 0.10, 0.95)
        reasons.append("COLD_NIGHT_TRANSACTION")

    if not reasons:
        reasons.append("COLD_DEFAULT_PRIOR")

    # Route decision
    if p_fraud < APPROVE_THRESH:
        decision = "APPROVE"
    elif p_fraud < DECLINE_THRESH:
        decision = "STEP_UP_2FA"
    else:
        decision = "DECLINE"

    return {
        "p_fraud":  round(p_fraud, 4),
        "decision": decision,
        "reasons":  reasons,
        "path":     "COLD_START",
    }


def is_cold_start(entity_txn_count: int,
                  warm_threshold: int = 10) -> bool:
    """
    Check if entity qualifies as cold start

    Args:
        entity_txn_count: Number of historical transactions for entity
        warm_threshold: Minimum transactions to be considered warm

    Returns:
        True if cold start
    """
    return entity_txn_count < warm_threshold


def get_cold_start_rules() -> dict:
    """Return current cold-start rule configuration"""
    return {
        **COLD_RULES,
        "approve_thresh": APPROVE_THRESH,
        "decline_thresh": DECLINE_THRESH,
        "warm_threshold": 10,
    }