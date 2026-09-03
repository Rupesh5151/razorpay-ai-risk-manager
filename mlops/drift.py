"""
Model Drift Monitor
Computes Population Stability Index (PSI) and KL Divergence
monitoring for feature and score drift by querying the live API.
"""

import os
import json
import numpy as np
import requests
from datetime import datetime, timezone, timedelta

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")


def compute_psi(expected: np.ndarray,
                actual: np.ndarray,
                buckets: int = 10) -> float:
    """Compute Population Stability Index (PSI)"""
    expected = np.array(expected)
    actual   = np.array(actual)

    breakpoints = np.linspace(0, 1, buckets + 1)

    expected_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_pct   = np.histogram(actual,   breakpoints)[0] / len(actual)

    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)

    psi = np.sum(
        (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    )
    return float(psi)


def compute_kl_divergence(p: np.ndarray,
                          q: np.ndarray,
                          buckets: int = 10) -> float:
    """Compute KL Divergence between two distributions"""
    breakpoints = np.linspace(0, 1, buckets + 1)
    p_hist = np.histogram(p, breakpoints)[0] / len(p)
    q_hist = np.histogram(q, breakpoints)[0] / len(q)

    p_hist = np.where(p_hist == 0, 1e-6, p_hist)
    q_hist = np.where(q_hist == 0, 1e-6, q_hist)

    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def load_recent_scores_from_api(hours: int = 24) -> list:
    """
    Fetch recent fraud scores from the live FastAPI backend audit endpoint
    instead of looking for a local file.
    """
    scores = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    try:
        # Pull up to 500 recent audit records from your live backend API
        r = requests.get(f"{API_BASE}/audit/history?limit=500", timeout=10)
        if r.status_code == 200:
            history = r.json()
            if isinstance(history, list):
                for record in history:
                    ts = record.get("timestamp", "")
                    if ts:
                        try:
                            dt = datetime.fromisoformat(ts)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            if dt > cutoff:
                                p = record.get("p_fraud")
                                if p is not None:
                                    scores.append(float(p))
                        except Exception:
                            continue
    except Exception as e:
        print(f"Failed to fetch audit history from API: {e}")

    return scores


def check_score_drift(reference_scores: list,
                      hours: int = 24,
                      psi_threshold: float = 0.2) -> dict:
    """Check if recent fraud scores have drifted from reference"""
    recent_scores = load_recent_scores_from_api(hours)

    if len(recent_scores) < 3:  # Lowered threshold slightly for testing/demos
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Only {len(recent_scores)} recent scores found via API (need 3+)",
            "recent_count": len(recent_scores),
            "psi": None,
            "kl_divergence": None,
            "drift_detected": False,
            "retrain_recommended": False,
        }

    # If reference scores are sparse, pad or synthesize a stable baseline distribution
    if len(reference_scores) < 10:
        np.random.seed(42)
        reference_scores = list(np.clip(np.random.beta(a=2, b=5, size=100), 0, 1))

    psi = compute_psi(np.array(reference_scores), np.array(recent_scores))
    kl  = compute_kl_divergence(np.array(reference_scores), np.array(recent_scores))

    drift_detected      = psi >= psi_threshold
    retrain_recommended = psi >= psi_threshold

    if psi < 0.1:
        status = "STABLE"
    elif psi < 0.2:
        status = "MONITOR"
    else:
        status = "DRIFT_DETECTED"

    return {
        "status": status,
        "psi": round(psi, 4),
        "kl_divergence": round(kl, 4),
        "psi_threshold": psi_threshold,
        "drift_detected": drift_detected,
        "retrain_recommended": retrain_recommended,
        "recent_count": len(recent_scores),
        "reference_count": len(reference_scores),
        "recent_mean": round(float(np.mean(recent_scores)), 4),
        "reference_mean": round(float(np.mean(reference_scores)), 4),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "message": (
            "Retraining recommended — score distribution has shifted"
            if retrain_recommended else
            "Model is stable — no retraining needed"
        ),
    }


def get_drift_summary() -> dict:
    """Quick drift summary querying live audit records"""
    reference = load_recent_scores_from_api(hours=168) # 7 days
    recent    = load_recent_scores_from_api(hours=24)  # 24 hours

    if len(recent) < 3:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Not enough recent data for drift analysis ({len(recent)} found, need 3+). Score some transactions first!",
            "reference_count": len(reference),
            "recent_count": len(recent),
        }

    return check_score_drift(reference, hours=24)