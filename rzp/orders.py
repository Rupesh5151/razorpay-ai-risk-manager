"""
Razorpay Order Management
Create test orders and score them with fraud model
"""

from rzp.client import RazorpayClient
from datetime import datetime
import uuid

class OrderManager:
    """Manage Razorpay orders and link them to fraud scores"""
    
    def __init__(self):
        self.client = RazorpayClient()
    
    def create_test_order(self, amount_inr: float, metadata: dict = None) -> dict:
        """
        Create a test order ready for fraud scoring
        
        Args:
            amount_inr: Amount in INR (not paise)
            metadata: Custom metadata (merchant_id, customer_id, etc.)
        
        Returns:
            Order dict with id and details
        """
        amount_paise = int(amount_inr * 100)
        
        meta = {
            "created_by": "fraud_detection_system",
            "created_at": datetime.utcnow().isoformat(),
            "test_order": True,
            **(metadata or {})
        }
        
        try:
            order = self.client.create_order(
                amount=amount_paise,
                currency="INR",
                receipt=f"test_{uuid.uuid4().hex[:8]}",
                metadata=meta
            )
            return order
        except Exception as e:
            raise Exception(f"Failed to create test order: {e}")
    
    def get_order_details(self, order_id: str) -> dict:
        """Fetch full order details including payment status"""
        try:
            return self.client.fetch_order(order_id)
        except Exception as e:
            raise Exception(f"Failed to get order details: {e}")
    
    def list_recent_orders(self, count: int = 10) -> list:
        """Get recent orders for batch scoring"""
        try:
            payments = self.client.fetch_payments(count=count)
            return payments.get("items", [])
        except Exception as e:
            raise Exception(f"Failed to list orders: {e}")