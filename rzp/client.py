"""
Razorpay API Client Wrapper
Handles authentication and communication with Razorpay test-mode API
"""

import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

class RazorpayClient:
    """Wrapper around Razorpay Payments API"""
    
    BASE_URL = "https://api.razorpay.com/v1"
    
    def __init__(self):
        self.key_id = os.getenv("RAZORPAY_KEY_ID")
        self.key_secret = os.getenv("RAZORPAY_KEY_SECRET")
        
        if not self.key_id or not self.key_secret:
            raise ValueError("RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set in .env")
        
        self.auth = HTTPBasicAuth(self.key_id, self.key_secret)
    
    def create_order(self, amount: int, currency: str = "INR", 
                     receipt: str = None, metadata: dict = None) -> dict:
        """
        Create a Razorpay order
        
        Args:
            amount: Amount in paise (100 paise = 1 INR)
            currency: Currency code (default: INR)
            receipt: Receipt ID for tracking
            metadata: Custom metadata dict
        
        Returns:
            Order dict with id, amount, currency, etc.
        """
        url = f"{self.BASE_URL}/orders"
        
        payload = {
            "amount": amount,
            "currency": currency,
            "receipt": receipt or f"order_{int(os.urandom(4).hex(), 16)}",
        }
        
        if metadata:
            payload["notes"] = metadata
        
        try:
            response = requests.post(url, json=payload, auth=self.auth)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to create order: {e}")
    
    def fetch_order(self, order_id: str) -> dict:
        """Fetch order details by ID"""
        url = f"{self.BASE_URL}/orders/{order_id}"
        
        try:
            response = requests.get(url, auth=self.auth)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch order: {e}")
    
    def fetch_payments(self, count: int = 10, skip: int = 0) -> dict:
        """Fetch recent payments"""
        url = f"{self.BASE_URL}/payments"
        params = {"count": count, "skip": skip}
        
        try:
            response = requests.get(url, params=params, auth=self.auth)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch payments: {e}")
    
    def health_check(self) -> bool:
        """Verify API credentials are valid"""
        try:
            self.fetch_payments(count=1)
            return True
        except:
            return False