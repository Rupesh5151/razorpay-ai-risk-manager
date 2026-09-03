"""
Merchant Webhook Handler
Fire webhooks when fraud decisions are made (STEP_UP, DECLINE)
"""

import requests
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MerchantWebhookManager:
    """Send webhooks to merchant endpoints on fraud events"""
    
    def __init__(self, merchant_webhook_url: str = None):
        """
        Args:
            merchant_webhook_url: URL to send webhooks (can be overridden per call)
        """
        self.default_url = merchant_webhook_url
    
    def fire_webhook(self, decision: str, transaction: dict, 
                     webhook_url: str = None, timeout: int = 5) -> bool:
        """
        Send webhook to merchant
        
        Args:
            decision: APPROVE, STEP_UP_2FA, or DECLINE
            transaction: Transaction details including p_fraud, reasons, etc.
            webhook_url: Override default webhook URL
            timeout: Request timeout in seconds
        
        Returns:
            True if webhook sent successfully
        """
        url = webhook_url or self.default_url
        
        if not url:
            logger.warning("No webhook URL configured, skipping")
            return False
        
        # Only fire webhooks for STEP_UP and DECLINE
        if decision not in ["STEP_UP_2FA", "DECLINE"]:
            return True  # APPROVE doesn't need webhook
        
        payload = {
            "event": "fraud_decision",
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision,
            "transaction_id": transaction.get("transaction_id"),
            "p_fraud": transaction.get("p_fraud"),
            "reasons": transaction.get("reasons", []),
            "amount": transaction.get("amount"),
            "merchant_id": transaction.get("merchant_id"),
            "action_required": decision == "STEP_UP_2FA",
            "audit_id": transaction.get("audit_id"),
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Webhook sent: {decision} for txn {payload['transaction_id']}")
                return True
            else:
                logger.warning(f"Webhook failed: {response.status_code} - {response.text}")
                return False
        
        except requests.exceptions.Timeout:
            logger.error(f"Webhook timeout for {url}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Webhook error: {e}")
            return False
    
    def fire_step_up_webhook(self, transaction: dict, webhook_url: str = None) -> bool:
        """Fire STEP_UP webhook (2FA required)"""
        return self.fire_webhook("STEP_UP_2FA", transaction, webhook_url)
    
    def fire_decline_webhook(self, transaction: dict, webhook_url: str = None) -> bool:
        """Fire DECLINE webhook (transaction blocked)"""
        return self.fire_webhook("DECLINE", transaction, webhook_url)