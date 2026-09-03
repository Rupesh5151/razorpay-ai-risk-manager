"""
Razorpay AI Risk Manager - FastAPI Inference Server
With Razorpay test-mode order integration and merchant webhooks
"""

import time, json, pickle, logging, math, uuid, os
from typing import Optional
from datetime import datetime

import numpy as np
import lightgbm as lgb
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rzp.client import RazorpayClient
from rzp.orders import OrderManager
from webhooks.merchant import MerchantWebhookManager
from audit.logger import AuditLogger

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Razorpay AI Risk Manager",
    description="Real-time fraud detection with Razorpay integration - Track 02 AI Buildathon 2026",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODEL = CALIBRATOR = METADATA = EXPLAINER = FEATURE_COLS = THRESHOLDS = None
order_manager = None
webhook_manager = None
audit_logger = AuditLogger()

@app.on_event("startup")
async def load_models():
    global MODEL, CALIBRATOR, METADATA, EXPLAINER, FEATURE_COLS, THRESHOLDS
    global order_manager, webhook_manager
    
    try:
        MODEL = lgb.Booster(model_file="artifacts/lgbm_risk_model.txt")
        with open("artifacts/isotonic_calibrator.pkl", "rb") as f:
            CALIBRATOR = pickle.load(f)
        with open("artifacts/model_metadata.json") as f:
            METADATA = json.load(f)
        
        FEATURE_COLS = METADATA["feature_cols"]
        THRESHOLDS   = METADATA["thresholds"]
        EXPLAINER    = shap.TreeExplainer(MODEL)
        
        order_manager = OrderManager()
        webhook_manager = MerchantWebhookManager()
        
        logger.info(f"Model loaded: {METADATA['model_version']}")
        logger.info(f"Razorpay client initialized")
        
        # Test Razorpay connection
        if order_manager.client.health_check():
            logger.info("✅ Razorpay API connection verified")
        else:
            logger.warning("⚠️ Razorpay API connection failed - check credentials")
    
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

class TransactionRequest(BaseModel):
    TransactionAmt:      float         = Field(..., gt=0)
    card1:               Optional[int] = None
    card2:               Optional[int] = None
    card3:               Optional[int] = None
    card5:               Optional[int] = None
    hour_of_day:         Optional[int] = Field(None, ge=0, le=23)
    day_of_week:         Optional[int] = Field(None, ge=0, le=6)
    is_night:            Optional[int] = Field(0, ge=0, le=1)
    is_weekend:          Optional[int] = Field(0, ge=0, le=1)
    is_cold_start:       Optional[int] = Field(0, ge=0, le=1)
    risky_email_domain:  Optional[int] = Field(0, ge=0, le=1)
    addr_mismatch:       Optional[int] = Field(0, ge=0, le=1)
    card1_vel_3600s:     Optional[float] = Field(1.0)
    card1_vel_21600s:    Optional[float] = Field(1.0)
    card1_vel_86400s:    Optional[float] = Field(1.0)
    merchant_id:         Optional[str] = None
    order_id:            Optional[str] = None
    C1:  Optional[float] = None
    C2:  Optional[float] = None
    C6:  Optional[float] = None
    D1:  Optional[float] = None
    V1:  Optional[float] = None

    class Config:
        json_schema_extra = {"example": {
            "TransactionAmt": 250.0, "card1": 9500,
            "hour_of_day": 14, "is_cold_start": 0,
            "card1_vel_3600s": 3.0, "card1_vel_21600s": 8.0,
            "card1_vel_86400s": 15.0, "merchant_id": "merchant_123"
        }}

COLD_RULES = {"amount_hard_limit": 500, "night_multiplier": 0.7,
              "risky_domain_block": True, "default_risk_score": 0.45}

def cold_start_score(txn):
    p, reasons = COLD_RULES["default_risk_score"], []
    limit = COLD_RULES["amount_hard_limit"]
    if txn.is_night: limit *= COLD_RULES["night_multiplier"]
    if txn.TransactionAmt > limit:
        p = max(p, 0.78); reasons.append("COLD_AMOUNT_EXCEEDS_LIMIT")
    if txn.risky_email_domain and COLD_RULES["risky_domain_block"]:
        p = max(p, 0.85); reasons.append("COLD_RISKY_EMAIL_DOMAIN")
    if txn.is_night:
        p = min(p + 0.10, 0.95); reasons.append("COLD_NIGHT_TRANSACTION")
    if not reasons: reasons.append("COLD_DEFAULT_PRIOR")
    return {"p_fraud": round(p, 4), "reasons": reasons}

def ml_score(txn):
    d = txn.dict()
    d["log_amount"]     = math.log1p(txn.TransactionAmt)
    d["amount_rounded"] = int(txn.TransactionAmt % 1 == 0)
    d["amount_gt_500"]  = int(txn.TransactionAmt > 500)
    x = np.array([[d.get(f, -999) for f in FEATURE_COLS]], dtype=np.float32)
    raw  = MODEL.predict(x)[0]
    cal  = float(CALIBRATOR.predict([raw])[0])
    sv   = EXPLAINER.shap_values(x)
    sv   = sv[1][0] if isinstance(sv, list) else sv[0]
    top3 = sorted(zip(sv, FEATURE_COLS), key=lambda t: abs(t[0]), reverse=True)[:3]
    reasons = [f"{'HIGH' if v>0 else 'LOW'}_{f.upper()}" for v, f in top3]
    return {"p_fraud": round(cal, 4), "reasons": reasons}

def route(p):
    if p < THRESHOLDS["approve"]: return "APPROVE"
    if p < THRESHOLDS["stepup"]:  return "STEP_UP_2FA"
    return "DECLINE"

@app.post("/score")
async def score_transaction(txn: TransactionRequest):
    t0    = time.perf_counter()
    txn_id = str(uuid.uuid4())[:12]
    
    try:
        if txn.is_cold_start:
            result, path, ver = cold_start_score(txn), "COLD_START", "rules-v1.0"
        else:
            result, path, ver = ml_score(txn), "ML_MODEL", METADATA["model_version"]
        
        p        = result["p_fraud"]
        decision = route(p)
        latency  = round((time.perf_counter() - t0) * 1000, 2)
        
        # Log to audit trail
        audit_logger.log_decision({
            "audit_id": txn_id,
            "transaction_id": txn.order_id or txn_id,
            "decision": decision,
            "p_fraud": p,
            "reasons": result["reasons"],
            "amount": txn.TransactionAmt,
            "merchant_id": txn.merchant_id,
            "model_version": ver,
            "latency_ms": latency,
            "path": path,
        })
        
        # Fire webhook for STEP_UP and DECLINE
        if decision in ["STEP_UP_2FA", "DECLINE"]:
            webhook_manager.fire_webhook(decision, {
                "transaction_id": txn.order_id or txn_id,
                "audit_id": txn_id,
                "decision": decision,
                "p_fraud": p,
                "reasons": result["reasons"],
                "amount": txn.TransactionAmt,
                "merchant_id": txn.merchant_id,
            })
        
        logger.info(f"[{txn_id}] {decision} p={p} path={path} {latency}ms")
        
        return {
            "transaction_id": txn_id,
            "p_fraud": p,
            "decision": decision,
            "reasons": result["reasons"],
            "path": path,
            "model_ver": ver,
            "latency_ms": latency,
            "audit": {
                "txn_id": txn_id,
                "order_id": txn.order_id,
                "merchant_id": txn.merchant_id,
                "amount": txn.TransactionAmt,
                "p_fraud": p,
                "decision": decision,
                "reasons": result["reasons"],
                "thresholds": THRESHOLDS,
                "latency_ms": latency,
            }
        }
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    razorpay_ok = order_manager.client.health_check() if order_manager else False
    return {
        "status": "ok",
        "model_ver": METADATA.get("model_version"),
        "razorpay_connected": razorpay_ok,
        "eval_metrics": METADATA.get("eval_metrics"),
        "thresholds": THRESHOLDS,
    }

@app.post("/razorpay/create-order")
async def create_razorpay_order(amount_inr: float, merchant_id: str = None):
    """Create a Razorpay test order for fraud scoring"""
    try:
        order = order_manager.create_test_order(
            amount_inr=amount_inr,
            metadata={"merchant_id": merchant_id} if merchant_id else {}
        )
        return {
            "order_id": order.get("id"),
            "amount": order.get("amount"),
            "currency": order.get("currency"),
            "created_at": order.get("created_at"),
            "next_step": "POST /score with this order_id to get fraud decision"
        }
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/audit/stats")
async def get_audit_stats(hours: int = 24):
    """Get fraud decision statistics"""
    return audit_logger.get_stats(hours)

@app.get("/audit/history")
async def get_audit_history(transaction_id: str = None, limit: int = 10):
    """Get decision history"""
    return audit_logger.get_decision_history(transaction_id, limit)

@app.post("/batch")
async def batch_score(transactions: list[TransactionRequest]):
    results = []
    for txn in transactions:
        r = await score_transaction(txn)
        results.append(r)
    counts = {}
    for r in results: counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    return {
        "summary": {**counts, "total": len(results),
                    "avg_latency_ms": round(
                        sum(r["latency_ms"] for r in results)/len(results), 2)},
        "results": results
    }
