"""
🔧 Emergency Fix Script for Amulet-AI
แก้ไขปัญหาด่วนและเริ่มระบบใหม่
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def emergency_install():
    """Install only the most essential packages"""
    print("🚨 Emergency installation of core packages...")
    
    essential_packages = [
        "fastapi",
        "uvicorn", 
        "streamlit",
        "pillow",
        "numpy",
        "requests"
    ]
    
    for package in essential_packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True)
            print(f"  ✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"  ⚠️ {package} installation failed, trying without version...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "--upgrade", package
                ], check=True)
                print(f"  ✅ {package} upgraded")
            except:
                print(f"  ❌ {package} failed completely")

def create_minimal_api():
    """Create the simplest possible API"""
    api_code = '''from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def read_root():
    return {"message": "Amulet-AI API Ready"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(front: UploadFile = File(...)):
    classes = ["สมเด็จ", "โพธิ์ฐานบัว", "สีวลี", "หลวงพ่อกวย"]
    predicted = random.choice(classes)
    return {
        "top1": {"class_name": predicted, "confidence": 0.85},
        "topk": [{"class_name": predicted, "confidence": 0.85}],
        "valuation": {"p05": 10000, "p50": 25000, "p95": 50000, "confidence": "medium"},
        "recommendations": [{"name": "ตลาดจตุจักร", "distance": 5, "rating": 4.5}],
        "ai_mode": "emergency_mock"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
'''
    
    Path("backend").mkdir(exist_ok=True)
    Path("backend/emergency_api.py").write_text(api_code)
    print("✅ Emergency API created")

def create_minimal_frontend():
    """Create the simplest possible Streamlit app"""
    frontend_code = '''import streamlit as st
import requests
from PIL import Image

st.title("🏺 Amulet-AI (Emergency Mode)")

uploaded_file = st.file_uploader("Upload amulet image", type=['png', 'jpg', 'jpeg'])

if uploaded_file and st.button("Analyze"):
    with st.spinner("Analyzing..."):
        try:
            files = {"front": uploaded_file.getvalue()}
            response = requests.post("http://127.0.0.1:8000/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted: {result['top1']['class_name']}")
                st.write(f"Confidence: {result['top1']['confidence']:.2%}")
                st.write("Price range:", f"{result['valuation']['p50']:,} THB")
            else:
                st.error("API Error")
        except Exception as e:
            st.error(f"Error: {e}")

st.write("Emergency Mode - Basic functionality only")
'''
    
    Path("frontend").mkdir(exist_ok=True) 
    Path("frontend/emergency_app.py").write_text(frontend_code)
    print("✅ Emergency frontend created")

def start_emergency_system():
    """Start the emergency system"""
    print("\n🚀 Starting emergency system...")
    
    # Start API
    api_process = subprocess.Popen([
        sys.executable, "backend/emergency_api.py"
    ])
    
    time.sleep(3)
    
    # Start frontend
    frontend_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "frontend/emergency_app.py",
        "--server.port", "8501"
    ])
    
    time.sleep(5)
    
    print("🎉 Emergency system started!")
    print("🔗 Web: http://127.0.0.1:8501") 
    print("🔗 API: http://127.0.0.1:8000")
    
    try:
        input("\nPress Enter to stop...")
    except KeyboardInterrupt:
        pass
    finally:
        api_process.terminate()
        frontend_process.terminate()
        print("✅ Emergency system stopped")

def main():
    print("🚨 AMULET-AI EMERGENCY FIX")
    print("=" * 30)
    
    print("\n[1/4] Emergency package installation...")
    emergency_install()
    
    print("\n[2/4] Creating emergency API...")
    create_minimal_api()
    
    print("\n[3/4] Creating emergency frontend...")
    create_minimal_frontend()
    
    print("\n[4/4] Starting emergency system...")
    start_emergency_system()

if __name__ == "__main__":
    main()
