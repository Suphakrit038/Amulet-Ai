"""
Simple test script to verify the connection between frontend and backend
"""
import requests
import sys
import os
import time

def test_api_connection():
    """Test basic connection to the API server"""
    api_url = "http://127.0.0.1:8000"
    try:
        print(f"Testing connection to {api_url}/health...")
        resp = requests.get(f"{api_url}/health", timeout=5)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            print("✅ API Connection Successful")
            try:
                print(f"Response: {resp.json()}")
            except Exception:
                print(f"Response: {resp.text[:100]}...")
            return True
        else:
            print(f"❌ API Connection Failed - Status: {resp.status_code}")
            print(f"Response: {resp.text[:100]}...")
            return False
    except Exception as e:
        print(f"❌ API Connection Error: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing API Connection ===")
    test_api_connection()
