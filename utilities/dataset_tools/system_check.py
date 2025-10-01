#!/usr/bin/env python3
"""
Quick System Check - ตรวจสอบระบบด่วน
"""

import requests
import time
import subprocess
import os

def check_api_status():
    """ตรวจสอบสถานะ API"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data
        return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def check_frontend_status():
    """ตรวจสอบสถานะ Frontend"""
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """ตรวจสอบระบบทั้งหมด"""
    print("🔍 Amulet-AI System Health Check")
    print("=" * 40)
    
    # Check API
    print("1. 🚀 API Server (Port 8000)")
    api_ok, api_data = check_api_status()
    if api_ok:
        print("   ✅ Status: Online")
        print(f"   📊 Models Ready: {api_data.get('models_ready', False)}")
        print(f"   📋 Classes: {api_data.get('available_classes', 0)}")
    else:
        print(f"   ❌ Status: Offline ({api_data})")
    
    # Check Frontend
    print("\n2. 🎨 Frontend (Port 8501)")
    frontend_ok = check_frontend_status()
    if frontend_ok:
        print("   ✅ Status: Online")
    else:
        print("   ❌ Status: Offline")
    
    # Summary
    print("\n📋 Summary")
    print("-" * 20)
    if api_ok and frontend_ok:
        print("🎉 All systems operational!")
        print("\n🌐 Access URLs:")
        print("   Frontend: http://localhost:8501")
        print("   API Docs: http://localhost:8000/docs")
        print("   API Health: http://localhost:8000/health")
    elif api_ok:
        print("⚠️  API working, Frontend offline")
        print("💡 Try: streamlit run frontend/main_streamlit_app_simple.py --server.port 8501")
    elif frontend_ok:
        print("⚠️  Frontend working, API offline")
        print("💡 Try: python api/main_api_fast.py")
    else:
        print("❌ Both systems offline")
        print("💡 Start API: python api/main_api_fast.py")
        print("💡 Start Frontend: streamlit run frontend/main_streamlit_app_simple.py --server.port 8501")

if __name__ == "__main__":
    main()