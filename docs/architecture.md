# Architecture — Razorpay AI Risk Manager

## Overview

Real-time fraud detection pipeline built on LightGBM + cold-start
fallback logic, with full Razorpay test-mode integration, merchant
webhooks, and an immutable audit trail.

## Pipeline (left to right)

```
Transaction (Checkout / Webhook)
        │
        ▼
┌─────────────────────────────────┐
│ 1. INGESTION & GATEWAY          │
│   FastAPI + Pydantic V2         │
│   Schema validation < 1ms       │
│   OPA hard blocklist < 2ms      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 2. COLD-START ROUTER            │
│   Entity history check          │
│   WARM  → ML model path         │
│   COLD  → rule-based fallback   │
│   Warm-up threshold: 10 txns    │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 3. FEATURE STORE                │
│   Velocity (1h / 6h / 24h)      │
│   Amount features               │
│   Time features                 │
│   Entity aggregates             │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 4. HYBRID MODEL INFERENCE       │
│   LightGBM (3,129 trees)        │
│   451 features                  │
│   Isotonic calibration          │
│   P(Fraud) in [0, 1]            │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 5. DYNAMIC THRESHOLD ENGINE     │
│   3 operating configs           │
│   FPR ceiling enforced < 0.02   │
│   APPROVE / STEP_UP / DECLINE   │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 6. EXPLAINABILITY               │
│   TreeSHAP top-3 reason codes   │
│   Human-readable format         │
│   e.g. HIGH_C14, LOW_CARD1_AMT  │
└────────────┬────────────────────┘
             │
        ┌────┴─────┐
        ▼          ▼
  Audit Logger   Merchant Webhook
  (JSONL)        (STEP_UP/DECLINE)
        │
        ▼
  Evidence Pack
  (chargeback PDF)
```

## Key Design Decisions

**Cold-start router** — New entities (< 10 txn history) bypass the ML
model entirely and use conservative static rules. This prevents the
model from making overconfident decisions on sparse data.

**Isotonic calibration** — Raw LightGBM probabilities are calibrated
using isotonic regression. This ensures P(Fraud) = 0.3 actually means
30% of such transactions are fraudulent.

**Dynamic threshold routing** — Thresholds are not static. Three
configs (HIGH_PRECISION, BALANCED, HIGH_RECALL) let Razorpay choose
the business tradeoff. FPR ceiling of 2% is enforced as a hard
constraint.

**STEP_UP band** — Transactions in the middle-risk band are routed to
2FA challenge rather than hard declined. This recovers ~26.5% of all
fraud via authentication rather than blocking.

**Immutable audit log** — Every decision is logged to an append-only
JSONL file with model version, feature snapshot, and SHAP values.
This generates chargeback evidence packs automatically.

## Eval Metrics

| Config | Precision | Recall | FPR |
|---|---|---|---|
| HIGH_PRECISION | 0.885 | 0.276 | 0.0013 |
| BALANCED | 0.705 | 0.401 | 0.006 |

- AUC-ROC: 0.9187
- Fraud addressed: 66.6% (DECLINE + STEP_UP)
- Latency: < 40ms P95
- Dataset: IEEE-CIS, 590,540 transactions

## Repo Structure

```
razorpay-ai-risk-manager/
├── api/main.py                  <- FastAPI server
├── artifacts/                   <- Model files
├── audit/logger.py              <- Immutable decision log
├── audit/evidence.py            <- Chargeback evidence pack
├── batch/scorer.py              <- CSV batch scorer
├── coldstart/fallback.py        <- Cold-start rule engine
├── dashboard/app.py             <- Streamlit ops panel
├── mlops/drift.py               <- PSI drift detection
├── notebooks/                   <- Training notebook
├── outputs/                     <- EDA + eval charts
├── rzp/                         <- Razorpay client
├── tests/                       <- 51 unit tests
├── webhooks/merchant.py         <- Merchant webhook
├── docker/                      <- Containerization
└── docs/                        <- Architecture docs
```

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Model | LightGBM + Isotonic Regression |
| Explainability | TreeSHAP |
| Dashboard | Streamlit + Plotly |
| Payments | Razorpay test-mode API |
| Audit | Append-only JSONL |
| Drift | PSI + KL Divergence |
| Tests | pytest (51 tests) |
| Container | Docker + docker-compose |