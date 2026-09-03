"""
Razorpay AI Risk Manager - Ops Dashboard
Real-time fraud monitoring and transaction scoring
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from batch.scorer import score_batch_csv, validate_csv, generate_sample_csv
from mlops.drift import get_drift_summary, check_score_drift

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Razorpay AI Risk Manager",
    page_icon="https://razorpay.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide the default Streamlit gradient line at the top */
    [data-testid="stDecoration"] { display: none; }

    /* Keep primary button styling but allow everything else to adapt naturally */
    .stButton > button {
        background-color: #338cf0 !important;
        color: #ffffff !important;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover { background-color: #1d72d6 !important; }
    
    .stDownloadButton > button {
        background-color: transparent !important;
        border: 1px solid #338cf0 !important;
        color: #338cf0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────
def get_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

def get_audit_stats(hours=24):
    try:
        r = requests.get(f"{API_BASE}/audit/stats?hours={hours}", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return data
        return {}
    except Exception:
        return {}

def get_audit_history(limit=20):
    try:
        r = requests.get(f"{API_BASE}/audit/history?limit={limit}", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                return data
        return []
    except Exception:
        return []

def score_transaction(payload: dict):
    try:
        r = requests.post(f"{API_BASE}/score", json=payload, timeout=15)
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def create_order(amount_inr: float, merchant_id: str):
    try:
        r = requests.post(
            f"{API_BASE}/razorpay/create-order",
            params={"amount_inr": amount_inr, "merchant_id": merchant_id},
            timeout=15
        )
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://razorpay.com/favicon.ico", width=32)
    st.title("Risk Manager")
    st.caption("Track 02 — AI Buildathon 2026")
    st.divider()

    health = get_health()
    if health:
        st.success("API Online")
        rzp_status = "Connected" if health.get("razorpay_connected") else "Disconnected"
        st.caption(f"Razorpay: {rzp_status}")
        st.caption(f"Model: {health.get('model_ver', 'unknown')}")
    else:
        st.error("API Offline")
        health = {}

    st.divider()
    auto_refresh = st.toggle("Auto Refresh", value=False)
    refresh_interval = st.slider("Refresh interval (sec)", 5, 60, 10)

    st.divider()
    st.caption("Thresholds")
    t = health.get("thresholds", {}) if health else {}
    st.caption(f"APPROVE  < {t.get('approve', 'N/A')}")
    st.caption(f"STEP_UP  < {t.get('stepup',  'N/A')}")
    st.caption(f"DECLINE >= {t.get('decline', 'N/A')}")

# ── Main header ───────────────────────────────────────────────────────────────
st.title("Razorpay AI Risk Manager")
st.caption(
    f"Last updated: {datetime.now().strftime('%H:%M:%S')} | "
    "Track 02 — AI Buildathon 2026"
)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Live Dashboard",
    "Score Transaction",
    "Audit History",
    "Model Info",
    "Batch Scorer",
    "Drift Monitor",
])

# ════════════════════════════════════════════════════════════
# TAB 1 — LIVE DASHBOARD
# ════════════════════════════════════════════════════════════
with tab1:
    stats     = get_audit_stats(hours=24)
    total     = int(stats.get("total_decisions", 0))
    approve   = int(stats.get("approve", 0))
    step_up   = int(stats.get("step_up", 0))
    decline   = int(stats.get("decline", 0))
    avg_score = float(stats.get("avg_fraud_score", 0.0))

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Decisions", f"{total:,}")
    col2.metric("Approved",    f"{approve:,}",  f"{approve/max(total,1):.1%}", delta_color="normal")
    col3.metric("Step-Up 2FA", f"{step_up:,}",  f"{step_up/max(total,1):.1%}", delta_color="off")
    col4.metric("Declined",    f"{decline:,}",  f"{decline/max(total,1):.1%}", delta_color="inverse")
    col5.metric("Avg P(Fraud)", f"{avg_score:.4f}")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Decision Breakdown")
        if total > 0:
            fig = go.Figure(data=[go.Pie(
                labels=["Approve", "Step-Up 2FA", "Decline"],
                values=[approve, step_up, decline],
                hole=0.5,
                marker_colors=["#10b981", "#f59e0b", "#ef4444"],
            )])
            fig.update_layout(
                showlegend=True,
                margin=dict(t=20, b=20, l=20, r=20), height=280,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No decisions yet — score some transactions first")

    with col_right:
        st.subheader("Fraud Score Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Avg P(Fraud)"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar":  {"thickness": 0.2, "color": "#888888"}, 
                "steps": [
                    {"range": [0,    0.10], "color": "#10b981"},
                    {"range": [0.10, 0.35], "color": "#f59e0b"},
                    {"range": [0.35, 1.0],  "color": "#ef4444"},
                ],
            },
        ))
        fig.update_layout(
            height=320,
            margin=dict(t=60, b=20, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Decisions")
    history = get_audit_history(limit=10)
    if history:
        rows = []
        for rec in history:
            if not isinstance(rec, dict):
                continue
            decision = rec.get("decision", "")
            rows.append({
                "Time":     str(rec.get("timestamp", ""))[:19].replace("T", " "),
                "Decision": decision,
                "P(Fraud)": f"{float(rec.get('p_fraud', 0)):.4f}",
                "Amount":   f"Rs.{rec.get('amount', 0):,.0f}",
                "Reasons":  ", ".join(rec.get("reasons", [])[:2]),
                "Path":     rec.get("path", ""),
                "Latency":  f"{rec.get('latency_ms', 0):.1f}ms",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No decisions logged yet")

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

# ════════════════════════════════════════════════════════════
# TAB 2 — SCORE TRANSACTION
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Score a Transaction")
    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.caption("Transaction Details")
        amount      = st.number_input("Amount (INR)", min_value=1.0, max_value=100000.0, value=250.0)
        merchant_id = st.text_input("Merchant ID", value="merchant_123")
        card1       = st.number_input("Card ID (card1)", value=9500)
        hour        = st.slider("Hour of Day", 0, 23, 14)
        day_name    = st.selectbox("Day of Week",
                                   ["Monday","Tuesday","Wednesday",
                                    "Thursday","Friday","Saturday","Sunday"],
                                   index=1)
        day_num    = ["Monday","Tuesday","Wednesday",
                      "Thursday","Friday","Saturday","Sunday"].index(day_name)
        is_night   = 1 if (hour >= 22 or hour <= 5) else 0
        is_weekend = 1 if day_num >= 5 else 0

        vel_1h  = st.number_input("Card velocity (1h)",  value=3.0,  step=1.0)
        vel_6h  = st.number_input("Card velocity (6h)",  value=8.0,  step=1.0)
        vel_24h = st.number_input("Card velocity (24h)", value=15.0, step=1.0)

        col_a, col_b = st.columns(2)
        with col_a:
            is_cold     = st.checkbox("Cold Start")
            risky_email = st.checkbox("Risky Email")
        with col_b:
            addr_mismatch = st.checkbox("Address Mismatch")

        create_rzp = st.checkbox("Create Razorpay order first", value=True)
        score_btn  = st.button("Score Transaction", use_container_width=True)

    with col_result:
        st.caption("Result")
        if score_btn:
            order_id = None
            if create_rzp:
                with st.spinner("Creating Razorpay order..."):
                    order = create_order(amount, merchant_id)
                if "error" not in order:
                    order_id = order.get("order_id")
                    st.success(f"Order created: `{order_id}`")
                else:
                    st.warning(f"Order creation failed: {order['error']}")

            payload = {
                "TransactionAmt":     float(amount),
                "card1":              int(card1),
                "hour_of_day":        int(hour),
                "day_of_week":        int(day_num),
                "is_night":           int(is_night),
                "is_weekend":         int(is_weekend),
                "is_cold_start":      int(is_cold),
                "risky_email_domain": int(risky_email),
                "addr_mismatch":      int(addr_mismatch),
                "card1_vel_3600s":    float(vel_1h),
                "card1_vel_21600s":   float(vel_6h),
                "card1_vel_86400s":   float(vel_24h),
                "merchant_id":        str(merchant_id),
                "order_id":           order_id,
            }

            with st.spinner("Scoring..."):
                result = score_transaction(payload)

            if "error" not in result:
                decision = result.get("decision", "")
                p_fraud  = float(result.get("p_fraud", 0))
                reasons  = result.get("reasons", [])
                latency  = float(result.get("latency_ms", 0))

                color = {"APPROVE": "#10b981", "STEP_UP_2FA": "#f59e0b",
                         "DECLINE": "#ef4444"}.get(decision, "#338cf0")
                st.markdown(
                    f"<h2 style='color:{color}; font-weight:bold;'>{decision}</h2>",
                    unsafe_allow_html=True
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("P(Fraud)", f"{p_fraud:.4f}")
                m2.metric("Latency",  f"{latency:.1f}ms")
                m3.metric("Path",     result.get("path", ""))

                st.caption("Risk Factors")
                for r in reasons:
                    st.code(r)

                with st.expander("Audit Trail"):
                    st.json(result.get("audit", {}))
            else:
                st.error(f"Error: {result['error']}")

# ════════════════════════════════════════════════════════════
# TAB 3 — AUDIT HISTORY
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Audit History")
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        txn_filter = st.text_input("Filter by Transaction ID", placeholder="optional")
    with col_f2:
        limit = st.selectbox("Show last", [10, 25, 50, 100], index=0)

    history = get_audit_history(limit=int(limit))
    if txn_filter:
        history = [h for h in history
                   if isinstance(h, dict) and
                   txn_filter in str(h.get("transaction_id", ""))]

    if history:
        rows = []
        for rec in history:
            if not isinstance(rec, dict):
                continue
            decision = rec.get("decision", "")
            rows.append({
                "Timestamp":  str(rec.get("timestamp", ""))[:19].replace("T", " "),
                "Txn ID":     str(rec.get("transaction_id", ""))[:12],
                "Decision":   decision,
                "P(Fraud)":   round(float(rec.get("p_fraud", 0)), 4),
                "Amount":     f"Rs.{rec.get('amount', 0):,.0f}",
                "Merchant":   str(rec.get("merchant_id", "N/A")),
                "Top Reason": (rec.get("reasons", ["N/A"])[0]
                               if rec.get("reasons") else "N/A"),
                "Path":       rec.get("path", ""),
                "Latency":    f"{rec.get('latency_ms', 0):.1f}ms",
            })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No audit records found")

# ════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model Information")
    if health:
        metrics    = health.get("eval_metrics", {})
        thresholds = health.get("thresholds", {})
        col1, col2 = st.columns(2)

        with col1:
            st.caption("Eval Metrics — IEEE-CIS held-out val set")
            for k, v in {
                "AUC-ROC":              metrics.get("auc_roc", "N/A"),
                "Precision (High)":     metrics.get("precision_high", "N/A"),
                "Recall (High)":        metrics.get("recall_high", "N/A"),
                "FPR (High)":           metrics.get("fpr_high", "N/A"),
                "Precision (Balanced)": metrics.get("precision_balanced", "N/A"),
                "Recall (Balanced)":    metrics.get("recall_balanced", "N/A"),
                "FPR (Balanced)":       metrics.get("fpr_balanced", "N/A"),
            }.items():
                st.metric(k, v)

        with col2:
            st.caption("Threshold Configuration")
            st.metric("APPROVE threshold", f"< {thresholds.get('approve', 'N/A')}")
            st.metric("STEP_UP threshold", f"< {thresholds.get('stepup',  'N/A')}")
            st.metric("DECLINE threshold", f">= {thresholds.get('decline','N/A')}")
            st.divider()
            st.caption("Architecture")
            st.markdown("""
- **Model**: LightGBM (3,129 trees)
- **Calibration**: Isotonic Regression
- **Features**: 451 (V, C, D, M, id cols + engineered)
- **Cold-start**: Rule-based fallback (< 10 txn history)
- **Explainability**: TreeSHAP reason codes
- **Dataset**: IEEE-CIS Fraud Detection
- **Training size**: 472,432 transactions
            """)
    else:
        st.error("API offline — cannot load model info")

# ════════════════════════════════════════════════════════════
# TAB 5 — BATCH CSV SCORER
# ════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Batch CSV Scorer")
    st.caption("Upload a CSV of transactions and get fraud scores + SHAP reasons per row")

    col_dl, col_info = st.columns([1, 3])
    with col_dl:
        sample_csv = generate_sample_csv()
        st.download_button(
            label="Download Sample CSV",
            data=sample_csv,
            file_name="sample_transactions.csv",
            mime="text/csv",
        )
    with col_info:
        st.info("Required column: `TransactionAmt`. All other columns are optional.")

    st.divider()
    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        valid, msg = validate_csv(df_input)

        if not valid:
            st.error(f"Invalid CSV: {msg}")
        else:
            st.success(f"CSV loaded: {len(df_input)} transactions")
            with st.expander("Preview input data"):
                st.dataframe(df_input.head(5), use_container_width=True)

            if st.button("Score All Transactions", use_container_width=True):
                progress_bar = st.progress(0)
                status_text  = st.empty()

                def update_progress(pct):
                    progress_bar.progress(float(pct))
                    status_text.caption(
                        f"Scoring... {int(pct * len(df_input))}/{len(df_input)}")

                with st.spinner("Scoring transactions..."):
                    result_df = score_batch_csv(df_input, update_progress)

                progress_bar.progress(1.0)
                status_text.caption("Done!")

                st.divider()
                total_b   = len(result_df)
                approve_b = int((result_df["decision"] == "APPROVE").sum())
                stepup_b  = int((result_df["decision"] == "STEP_UP_2FA").sum())
                decline_b = int((result_df["decision"] == "DECLINE").sum())
                avg_p     = float(result_df["p_fraud"].mean()) if "p_fraud" in result_df.columns else 0.0

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total",       total_b)
                c2.metric("Approved",    approve_b, f"{approve_b/max(total_b,1):.1%}", delta_color="normal")
                c3.metric("Step-Up 2FA", stepup_b,  f"{stepup_b/max(total_b,1):.1%}", delta_color="off")
                c4.metric("Declined",    decline_b, f"{decline_b/max(total_b,1):.1%}", delta_color="inverse")
                c5.metric("Avg P(Fraud)", f"{avg_p:.4f}")

                st.divider()
                st.subheader("Results")
                display_cols = (
                    ["TransactionAmt"] +
                    [c for c in ["merchant_id","card1","hour_of_day","is_cold_start"]
                     if c in result_df.columns] +
                    ["p_fraud","decision","reason_1","reason_2","reason_3","path","latency_ms"]
                )
                display_df = result_df[[c for c in display_cols if c in result_df.columns]]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                csv_out = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Scored Results CSV",
                    data=csv_out,
                    file_name=f"scored_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

                if total_b > 0:
                    st.divider()
                    st.subheader("Decision Distribution")
                    fig = go.Figure(data=[go.Bar(
                        x=["Approve", "Step-Up 2FA", "Decline"],
                        y=[approve_b, stepup_b, decline_b],
                        marker_color=["#10b981", "#f59e0b", "#ef4444"],
                    )])
                    fig.update_layout(
                        height=300,
                        margin=dict(t=20, b=20, l=20, r=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 6 — DRIFT MONITOR
# ════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Model Drift Monitor")
    st.caption("PSI and KL divergence monitoring — retrain alert when PSI > 0.2")

    col_refresh, col_window = st.columns([1, 2])
    with col_refresh:
        run_drift = st.button("Run Drift Check", use_container_width=True)
    with col_window:
        st.caption("Compares last 24h scores against last 7 days as reference baseline")

    st.divider()

    if run_drift:
        with st.spinner("Computing PSI and KL divergence..."):
            summary = get_drift_summary()

        status = summary.get("status", "UNKNOWN")
        if status == "STABLE":
            st.success("Model is STABLE — No retraining needed")
        elif status == "MONITOR":
            st.warning("MONITOR — Moderate drift detected, keep watching")
        elif status == "DRIFT_DETECTED":
            st.error("DRIFT DETECTED — Retraining recommended")
        else:
            st.info(summary.get("message", "Insufficient data for drift analysis"))

        st.divider()

        if status not in ["INSUFFICIENT_DATA", "UNKNOWN"]:
            psi = float(summary.get("psi", 0) or 0)
            kl  = float(summary.get("kl_divergence", 0) or 0)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PSI Score", f"{psi:.4f}",
                      "Stable" if psi < 0.1 else "Monitor" if psi < 0.2 else "Retrain!",
                      delta_color="normal" if psi < 0.1 else "off" if psi < 0.2 else "inverse")
            c2.metric("KL Divergence",    f"{kl:.4f}")
            c3.metric("Recent Scores",    int(summary.get("recent_count", 0)))
            c4.metric("Reference Scores", int(summary.get("reference_count", 0)))

            st.divider()
            col_l, col_r = st.columns(2)

            with col_l:
                st.subheader("PSI Interpretation")
                psi_df = pd.DataFrame({
                    "Range":    ["< 0.1", "0.1 to 0.2", "> 0.2"],
                    "Status":   ["Stable", "Monitor", "Retrain"],
                    "Action":   ["None", "Keep watching", "Trigger retraining"],
                    "Your PSI": [
                        f"{psi:.4f}" if psi < 0.1 else "",
                        f"{psi:.4f}" if 0.1 <= psi < 0.2 else "",
                        f"{psi:.4f}" if psi >= 0.2 else "",
                    ]
                })
                st.dataframe(psi_df, use_container_width=True, hide_index=True)

            with col_r:
                st.subheader("Score Means")
                ref_mean    = float(summary.get("reference_mean", 0) or 0)
                recent_mean = float(summary.get("recent_mean",    0) or 0)
                fig = go.Figure(data=[go.Bar(
                    x=["Reference (7d)", "Recent (24h)"],
                    y=[ref_mean, recent_mean],
                    marker_color=["#3d7fff", "#ef4444"],
                    text=[f"{ref_mean:.4f}", f"{recent_mean:.4f}"],
                    textposition="auto",
                )])
                fig.update_layout(
                    height=250,
                    margin=dict(t=20, b=20, l=20, r=20),
                    yaxis=dict(title="Avg P(Fraud)"),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Retraining")
            if summary.get("retrain_recommended"):
                st.error("Retraining recommended — PSI has exceeded the 0.2 threshold.")
                if st.button("Trigger Retraining", type="primary", use_container_width=True):
                    st.info(
                        "In production this triggers the retraining pipeline. "
                        "For this demo, re-run the Kaggle training notebook "
                        "and replace the artifacts."
                    )
                    st.code("python models/trainer.py --retrain --data latest",
                            language="bash")
            else:
                st.success("No retraining needed — model is stable.")

            st.divider()
            st.caption(
                f"Last checked: {summary.get('checked_at', 'N/A')} | "
                f"PSI threshold: {summary.get('psi_threshold', 0.2)}"
            )
        else:
            st.info(
                f"Not enough data yet. Need at least 10 recent scores. "
                f"Current: {summary.get('recent_count', 0)}"
            )
    else:
        st.info("Click 'Run Drift Check' to analyse model drift.")
        st.subheader("How Drift Detection Works")
        st.markdown("""
**PSI (Population Stability Index)** measures how much the fraud score
distribution has shifted between a 7-day reference window and the last 24 hours.

| PSI Range | Status | Action |
|---|---|---|
| < 0.1 | Stable | No action needed |
| 0.1 to 0.2 | Monitor | Watch closely |
| > 0.2 | Drift | Retrain model |

**KL Divergence** is a complementary signal measuring information loss between distributions.
        """)