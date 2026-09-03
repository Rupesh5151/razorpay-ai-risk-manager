"""
Batch CSV Scorer
Upload a CSV of transactions, get back scored results with SHAP reasons
"""

import os
import pandas as pd
import numpy as np
import requests
import json
from typing import List

# Dynamically load API_BASE from environment variables, removing accidental trailing slashes
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")

REQUIRED_COLS = ["TransactionAmt"]

OPTIONAL_COLS = {
    "card1": 9500,
    "card2": None,
    "card3": None,
    "card5": None,
    "hour_of_day": 12,
    "day_of_week": 1,
    "is_night": 0,
    "is_weekend": 0,
    "is_cold_start": 0,
    "risky_email_domain": 0,
    "addr_mismatch": 0,
    "card1_vel_3600s": 1.0,
    "card1_vel_21600s": 3.0,
    "card1_vel_86400s": 10.0,
    "merchant_id": None,
    "C1": None,
    "C2": None,
    "C6": None,
    "D1": None,
    "V1": None,
}

def validate_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """Check CSV has required columns"""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    return True, "OK"

def score_batch_csv(df: pd.DataFrame,
                    progress_callback=None) -> pd.DataFrame:
    """
    Score each row in the CSV via the /score API
    
    Args:
        df: Input dataframe with transaction data
        progress_callback: Optional callable(pct) for progress updates
    
    Returns:
        DataFrame with original cols + P(Fraud), Decision, Reasons, Latency
    """
    results = []
    total = len(df)

    for i, row in df.iterrows():
        # Build payload with defaults for missing cols
        payload = {"TransactionAmt": float(row["TransactionAmt"])}

        for col, default in OPTIONAL_COLS.items():
            if col in df.columns and pd.notna(row[col]):
                val = row[col]
                if col in ["card1","card2","card3","card5","hour_of_day",
                           "day_of_week","is_night","is_weekend",
                           "is_cold_start","risky_email_domain","addr_mismatch"]:
                    try:
                        payload[col] = int(val)
                    except:
                        payload[col] = default
                elif col == "merchant_id":
                    payload[col] = str(val)
                else:
                    try:
                        payload[col] = float(val)
                    except:
                        payload[col] = default
            elif default is not None:
                payload[col] = default

        # Derive time features if not present
        if "hour_of_day" in payload:
            h = payload["hour_of_day"]
            payload["is_night"] = int(h >= 22 or h <= 5)
        if "day_of_week" in payload:
            payload["is_weekend"] = int(payload["day_of_week"] >= 5)

        try:
            r = requests.post(
                f"{API_BASE}/score",
                json=payload,
                timeout=60  # Increased to 60 seconds for Render cold starts
            )
            data = r.json()
            results.append({
                "p_fraud":      data.get("p_fraud", None),
                "decision":     data.get("decision", "ERROR"),
                "reason_1":     data.get("reasons", ["N/A"])[0] if data.get("reasons") else "N/A",
                "reason_2":     data.get("reasons", ["N/A","N/A"])[1] if len(data.get("reasons",[])) > 1 else "N/A",
                "reason_3":     data.get("reasons", ["N/A","N/A","N/A"])[2] if len(data.get("reasons",[])) > 2 else "N/A",
                "path":         data.get("path", "N/A"),
                "latency_ms":   data.get("latency_ms", None),
                "transaction_id": data.get("transaction_id", "N/A"),
            })
        except Exception as e:
            results.append({
                "p_fraud": None, "decision": "ERROR",
                "reason_1": str(e), "reason_2": "N/A",
                "reason_3": "N/A", "path": "N/A",
                "latency_ms": None, "transaction_id": "N/A",
            })

        if progress_callback:
            progress_callback((i + 1) / total)

    result_df = pd.concat([
        df.reset_index(drop=True),
        pd.DataFrame(results)
    ], axis=1)

    return result_df

def generate_sample_csv() -> str:
    """Generate a sample CSV for download"""
    sample = pd.DataFrame({
        "TransactionAmt": [250.0, 1500.0, 45.0, 800.0, 12.0,
                           3000.0, 99.0, 500.0, 75.0, 2000.0],
        "card1":          [9500, 1234, 5678, 9999, 1111,
                           2222, 3333, 4444, 5555, 6666],
        "hour_of_day":    [14, 23, 10, 2, 16,
                           22, 9, 15, 11, 3],
        "day_of_week":    [1, 5, 2, 6, 3,
                           0, 4, 1, 2, 5],
        "card1_vel_3600s":[3, 1, 5, 2, 8,
                           1, 4, 2, 6, 1],
        "card1_vel_21600s":[8, 2, 12, 5, 20,
                            3, 9, 6, 15, 2],
        "card1_vel_86400s":[15, 5, 25, 10, 40,
                            8, 18, 12, 30, 4],
        "is_cold_start":  [0, 0, 0, 1, 0,
                           1, 0, 0, 0, 1],
        "risky_email_domain": [0, 0, 0, 0, 1,
                                0, 0, 0, 0, 1],
        "addr_mismatch":  [0, 1, 0, 1, 0,
                           0, 0, 1, 0, 1],
        "merchant_id":    ["merchant_001"] * 10,
    })
    return sample.to_csv(index=False)