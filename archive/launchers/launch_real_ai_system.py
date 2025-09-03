#!/usr/bin/env python3
"""
Amulet-AI System Launcher with Real Trained Model
เปิดใช้งานระบบ AI จริงที่เทรนไว้แล้ว
"""

import os
import sys
import time
import subprocess
import threading
from datetime import datetime

def print_banner():
    """แสดง banner"""
    print("=" * 60)
    print("Amulet-AI System with Real Trained Model")
    print("   ระบบ AI วิเคราะห์พระเครื่องแบบจริง")
    print("=" * 60)
    print()

def check_python_environment():
    """ตรวจสอบ Python environment"""
    print("ตรวจสอบ Python environment...")
    print(f"   Python version: {sys.version}")
    print(f"   Working directory: {os.getcwd()}")

    try:
        import torch  # noqa: F401
        import streamlit  # noqa: F401
        import fastapi  # noqa: F401
        import PIL  # noqa: F401
        print("Dependencies พื้นฐานติดตั้งแล้ว")
        return True
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("   กรุณารัน: pip install -r requirements.txt")
        return False

def install_dependencies():
    """ติดตั้ง dependencies"""
    print("\nกำลังติดตั้ง/อัปเดต dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("Dependencies ติดตั้งเรียบร้อย")
    else:
        print(f"Dependencies warning: {result.stderr}")

def start_real_ai_backend():
    """เริ่ม Real AI Backend"""
    print("\nเริ่มต้น Real AI Backend (Port 8001)...")
    backend_path = os.path.join(os.getcwd(), "backend", "api_with_real_model.py")
    if not os.path.exists(backend_path):
        print(f"ไม่พบไฟล์ backend: {backend_path}")
        return None
    try:
        process = subprocess.Popen([
            sys.executable, backend_path
        ], cwd=os.path.join(os.getcwd(), "backend"))
        print(f"Real AI Backend เริ่มทำงานแล้ว (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"ไม่สามารถเริ่ม backend: {e}")
        return None

def wait_for_backend(max_wait=30):
    """รอให้ backend พร้อมใช้งาน"""
    print(f"รอ Real AI Backend โหลดเสร็จ (สูงสุด {max_wait} วินาที)...")
    import requests
    for i in range(max_wait):
        try:
            response = requests.get("http://127.0.0.1:8001/health", timeout=2)
        except requests.exceptions.RequestException:
            response = None
        if response and response.status_code == 200:
            data = response.json()
            if data.get("ai_service_available", False):
                print("Real AI Backend พร้อมใช้งานแล้ว!")
                print(f"   Model loaded: {data.get('model_status')}")
                print(f"   Classes: {data.get('num_classes')}")
                return True
            else:
                print(f"Model กำลังโหลด... ({i+1}/{max_wait})")
        time.sleep(1)
    print("Backend อาจยังไม่พร้อม กรุณาตรวจสอบด้วยตนเอง")
    return False

def start_streamlit_frontend():
    """เริ่ม Streamlit Frontend"""
    print("\nเริ่ม Streamlit Frontend...")
    frontend_path = os.path.join("frontend", "app_streamlit.py")
    if not os.path.exists(frontend_path):
        print(f"ไม่พบไฟล์ frontend: {frontend_path}")
        return None
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", frontend_path,
            "--server.port", "8501",
            "--server.address", "127.0.0.1",
            "--browser.gatherUsageStats", "false"
        ])
        print("Streamlit Frontend เริ่มทำงานแล้ว!")
        print("   เปิดเบราว์เซอร์ไปที่: http://127.0.0.1:8501")
        return process
    except Exception as e:
        print(f"ไม่สามารถเริ่ม frontend: {e}")
        return None

def main():
    """Main function"""
    print_banner()
    
    # ตรวจสอบ environment
    if not check_python_environment():
        install_dependencies()
    
    print("\n" + "="*60)
    print("เริ่มระบบ Amulet-AI แบบ Real Model...")
    print("="*60)
    
    # เริ่ม backend
    backend_process = start_real_ai_backend()
    if not backend_process:
        print("ไม่สามารถเริ่ม backend ได้")
        return
    
    # รอ backend พร้อม
    backend_ready = wait_for_backend(30)
    
    # เริ่ม frontend
    frontend_process = start_streamlit_frontend()
    if frontend_process:
        print("\n" + "="*60)
        print("ระบบทำงานเรียบร้อยแล้ว!")
        print("   Real AI Backend: http://127.0.0.1:8001")
        print("   Streamlit Frontend: http://127.0.0.1:8501")
        print("   API Documentation: http://127.0.0.1:8001/docs")
        print("="*60)
        print("กด Ctrl+C เพื่อหยุดระบบ")
        try:
            frontend_process.wait()
        except KeyboardInterrupt:
            print("\nกำลังหยุดระบบ...")
        finally:
            if frontend_process:
                frontend_process.terminate()
            if backend_process:
                backend_process.terminate()
            print("หยุดระบบเรียบร้อย")
    else:
        print("ไม่สามารถเริ่ม frontend ได้")
        if backend_process:
            backend_process.terminate()

if __name__ == "__main__":
    main()
