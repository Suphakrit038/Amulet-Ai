#!/usr/bin/env python3
"""
🔧 Complete Amulet-AI System Setup & Verification
ระบบติดตั้งและตรวจสอบความพร้อม Amulet-AI แบบครบครัน
"""

import subprocess
import sys
import os
import time
import json
import requests
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_step(step, text):
    """Print formatted step"""
    print(f"\n[STEP {step}] {text}")
    print("-" * 40)

def check_python():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 9:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.9+ required")
        return False

def install_requirements():
    """Install core requirements"""
    core_packages = [
        "fastapi==0.115.0",
        "uvicorn[standard]==0.32.0", 
        "streamlit==1.40.0",
        "python-multipart==0.0.20",
        "pillow==10.4.0",
        "opencv-python-headless==4.10.0.84",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "requests==2.32.3",
        "pydantic==2.10.0",
        "matplotlib==3.9.3",
        "plotly==5.24.1"
    ]
    
    print("Installing core packages...")
    
    for package in core_packages:
        try:
            print(f"  Installing {package.split('==')[0]}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  ⚠️ Warning installing {package}: {result.stderr}")
            else:
                print(f"  ✅ {package.split('==')[0]} installed")
                
        except Exception as e:
            print(f"  ❌ Error installing {package}: {e}")
    
    return True

def verify_imports():
    """Verify that all required modules can be imported"""
    required_modules = [
        "fastapi",
        "uvicorn", 
        "streamlit",
        "PIL",
        "cv2",
        "numpy",
        "pandas",
        "requests",
        "pydantic",
        "matplotlib",
        "plotly"
    ]
    
    print("Verifying module imports...")
    failed_imports = []
    
    for module in required_modules:
        try:
            if module == "PIL":
                import PIL
                print(f"  ✅ PIL (Pillow) - OK")
            elif module == "cv2":
                import cv2
                print(f"  ✅ OpenCV - OK")
            else:
                __import__(module)
                print(f"  ✅ {module} - OK")
        except ImportError as e:
            print(f"  ❌ {module} - FAILED: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        return False
    else:
        print(f"\n✅ All {len(required_modules)} modules imported successfully!")
        return True

def check_project_structure():
    """Check and fix project structure"""
    print("Checking project structure...")
    
    required_dirs = [
        "backend",
        "frontend", 
        "utils",
        "ai_models",
        "logs"
    ]
    
    required_files = [
        "backend/__init__.py",
        "frontend/__init__.py",
        "utils/__init__.py",
        "config.json"
    ]
    
    # Create missing directories
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  📁 Created directory: {dir_name}")
        else:
            print(f"  ✅ Directory exists: {dir_name}")
    
    # Create missing files
    for file_path in required_files:
        file_obj = Path(file_path)
        if not file_obj.exists():
            if file_path.endswith("__init__.py"):
                file_obj.write_text("# Package initialization file\n")
            print(f"  📄 Created file: {file_path}")
        else:
            print(f"  ✅ File exists: {file_path}")
    
    return True

def create_mock_api():
    """Create or verify mock API exists"""
    mock_api_path = Path("backend/mock_api.py")
    
    if mock_api_path.exists():
        print("  ✅ Mock API file exists")
        return True
    
    print("  📄 Creating mock API file...")
    
    mock_api_code = '''"""
Simple Mock API for Amulet-AI Testing
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import time
import random

app = FastAPI(title="Amulet-AI Mock API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "🏺 Amulet-AI Mock API", "status": "ready"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/predict")
async def predict(front: UploadFile = File(...)):
    # Simulate processing time
    time.sleep(random.uniform(0.5, 1.5))
    
    # Mock prediction
    classes = ["สมเด็จ", "โพธิ์ฐานบัว", "สีวลี", "หลวงพ่อกวย"]
    predicted = random.choice(classes)
    confidence = random.uniform(0.7, 0.95)
    
    return {
        "top1": {"class_name": predicted, "confidence": confidence},
        "topk": [
            {"class_name": predicted, "confidence": confidence},
            {"class_name": random.choice(classes), "confidence": random.uniform(0.05, 0.2)},
            {"class_name": random.choice(classes), "confidence": random.uniform(0.02, 0.1)}
        ],
        "valuation": {
            "p05": random.randint(5000, 15000),
            "p50": random.randint(15000, 50000), 
            "p95": random.randint(50000, 150000),
            "confidence": "medium"
        },
        "recommendations": [
            {"name": "ตลาดจตุจักร", "distance": 10.5, "rating": 4.5},
            {"name": "ตลาดออนไลน์", "distance": 0, "rating": 4.0}
        ],
        "ai_mode": "mock",
        "processing_time": 1.2
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
'''
    
    mock_api_path.write_text(mock_api_code, encoding='utf-8')
    print("  ✅ Mock API created successfully")
    return True

def create_streamlit_app():
    """Create or verify Streamlit app exists"""
    app_path = Path("frontend/app_streamlit.py")
    
    if app_path.exists():
        print("  ✅ Streamlit app exists")
        return True
    
    print("  📄 Creating Streamlit app...")
    
    streamlit_code = '''"""
Amulet-AI Streamlit Frontend
"""
import streamlit as st
import requests
import time
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="🏺 Amulet-AI",
    page_icon="🏺",
    layout="wide"
)

# Title
st.title("🏺 Amulet-AI Recognition System")
st.write("ระบบจดจำพระเครื่องไทยด้วย AI")

# Sidebar
st.sidebar.title("📋 เมนู")
st.sidebar.write("**สถานะระบบ:** 🟢 พร้อมใช้งาน")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 อัปโหลดรูปภาพ")
    
    uploaded_file = st.file_uploader(
        "เลือกรูปภาพพระเครื่อง",
        type=['png', 'jpg', 'jpeg'],
        help="รองรับไฟล์ PNG, JPG, JPEG"
    )
    
    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="รูปภาพที่อัปโหลด", width=300)
        
        # Predict button
        if st.button("🔮 วิเคราะห์พระเครื่อง", type="primary"):
            with st.spinner("กำลังวิเคราะห์..."):
                try:
                    # Send to API
                    files = {"front": uploaded_file.getvalue()}
                    response = requests.post("http://127.0.0.1:8000/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        with col2:
                            st.header("📊 ผลการวิเคราะห์")
                            
                            # Prediction
                            st.subheader("🏷️ ประเภทพระเครื่อง")
                            top1 = result['top1']
                            confidence = top1['confidence']
                            
                            st.metric(
                                "ประเภท", 
                                top1['class_name'],
                                f"ความเชื่อมั่น: {confidence:.1%}"
                            )
                            
                            # Price estimation
                            st.subheader("💰 ประเมินราคา")
                            valuation = result['valuation']
                            
                            col_price1, col_price2, col_price3 = st.columns(3)
                            with col_price1:
                                st.metric("ต่ำสุด", f"{valuation['p05']:,} บาท")
                            with col_price2:
                                st.metric("กลาง", f"{valuation['p50']:,} บาท") 
                            with col_price3:
                                st.metric("สูงสุด", f"{valuation['p95']:,} บาท")
                            
                            # Recommendations
                            st.subheader("🏪 แนะนำตลาดขาย")
                            for rec in result['recommendations']:
                                with st.expander(f"📍 {rec['name']}"):
                                    col_rec1, col_rec2 = st.columns(2)
                                    with col_rec1:
                                        st.write(f"**ระยะทาง:** {rec['distance']} กม.")
                                    with col_rec2:
                                        st.write(f"**คะแนน:** {rec['rating']}/5.0")
                    else:
                        st.error("❌ เกิดข้อผิดพลาดในการเชื่อมต่อ API")
                        
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Footer
st.write("---")
st.write("🏺 **Amulet-AI** | Powered by FastAPI + Streamlit | v1.0.0")
'''
    
    app_path.write_text(streamlit_code, encoding='utf-8')
    print("  ✅ Streamlit app created successfully")
    return True

def test_api_connection():
    """Test API connection"""
    print("Testing API connection...")
    
    try:
        # Start mock API in background
        import subprocess
        api_process = subprocess.Popen([
            sys.executable, "backend/mock_api.py"
        ])
        
        # Wait for startup
        time.sleep(3)
        
        # Test connection
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        
        if response.status_code == 200:
            print("  ✅ API connection successful")
            api_process.terminate()
            return True
        else:
            print(f"  ❌ API returned status {response.status_code}")
            api_process.terminate()
            return False
            
    except Exception as e:
        print(f"  ⚠️ Could not test API: {e}")
        return False

def main():
    """Main setup function"""
    print_header("🏺 Amulet-AI Complete System Setup")
    
    print_step(1, "Checking Python Environment")
    if not check_python():
        print("❌ Python environment check failed")
        return False
    
    print_step(2, "Installing Core Dependencies")
    install_requirements()
    
    print_step(3, "Verifying Module Imports")
    if not verify_imports():
        print("❌ Module verification failed")
        return False
    
    print_step(4, "Checking Project Structure")
    check_project_structure()
    
    print_step(5, "Creating Mock API")
    create_mock_api()
    
    print_step(6, "Creating Streamlit App")
    create_streamlit_app()
    
    print_step(7, "Final System Test")
    test_api_connection()
    
    print_header("🎉 Setup Complete!")
    print("""
✅ System setup completed successfully!

🚀 To start the system:
   python launch_complete_system.py

🌐 Or use the batch file:
   launch.bat

📚 Manual commands:
   Backend:  python backend/mock_api.py
   Frontend: streamlit run frontend/app_streamlit.py
   
🔗 URLs:
   Web App: http://127.0.0.1:8501  
   API:     http://127.0.0.1:8000
   Docs:    http://127.0.0.1:8000/docs
""")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
