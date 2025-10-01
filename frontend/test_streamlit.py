#!/usr/bin/env python3
"""
🏺 Amulet-AI - Production Frontend
ระบบจำแนกพระเครื่องอัจฉริยะ
"""

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import json
import time
import base64
from pathlib import Path
import sys
import os
from datetime import datetime
import psutil
import threading
from collections import deque
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import enhanced modules (with fallback)
try:
    from core.error_handling_enhanced import error_handler, validate_image_file
    from core.performance_monitoring import performance_monitor
except:
    # Fallback if modules not available
    def error_handler(error_type="general"):
        def decorator(func):
            return func
        return decorator
    
    class performance_monitor:
        @staticmethod
        def collect_metrics():
            return {}

# Configuration
API_BASE_URL = "http://localhost:8000"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# NEW THEME COLORS - เริ่มใหม่หมด
COLORS = {
    'maroon': '#8B0000',     # แดงเลือดหมู
    'gold': '#DAA520',       # ทอง
    'dark_gold': '#B8860B',  # ทองเข้ม
    'light_gold': '#F4E4BC', # ทองอ่อน
    'green': '#28a745',      # เขียว
    'blue': '#007bff',       # น้ำเงิน
    'yellow': '#ffc107',     # เหลือง
    'red': '#dc3545',        # แดง
    'gray': '#6c757d',       # เทา
    'white': '#ffffff',      # ขาว
    'black': '#000000'       # ดำ
}

# Page Configuration
st.set_page_config(
    page_title="🏺 Amulet-AI - ระบบจำแนกพระเครื่อง",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CLASSIC CSS - สไตล์คลาสสิก ไม่มี Gradient
st.markdown(f"""
<style>
    /* Basic App Styling */
    .stApp {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f5f5f5;
    }}
    
    /* Logo Header Container */
    .logo-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 40px;
        background: white;
        border-bottom: 3px solid {COLORS['maroon']};
        margin-bottom: 30px;
    }}
    
    .logo-left {{
        display: flex;
        align-items: center;
        gap: 15px;
    }}
    
    .logo-right {{
        display: flex;
        align-items: center;
        gap: 20px;
    }}
    
    .logo-img {{
        height: 80px;
        width: auto;
        object-fit: contain;
    }}
    
    .logo-img-small {{
        height: 60px;
        width: auto;
        object-fit: contain;
    }}
    
    /* Main Header */
    .main-header {{
        background-color: {COLORS['maroon']};
        color: white;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        border: 2px solid {COLORS['dark_gold']};
    }}
    
    /* Status Badge */
    .status-badge {{
        display: inline-block;
        padding: 10px 25px;
        border-radius: 5px;
        font-weight: bold;
        margin: 20px 0;
        border: 2px solid;
    }}
    
    .status-online {{
        background-color: #28a745;
        color: white;
        border-color: #1e7e34;
    }}
    
    .status-offline {{
        background-color: #dc3545;
        color: white;
        border-color: #bd2130;
    }}
    
    /* Success Box */
    .success-box {{
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
        padding: 20px;
        margin: 20px 0;
    }}
    
    /* Info Box */
    .info-box {{
        background-color: #d1ecf1;
        border: 2px solid #17a2b8;
        color: #0c5460;
        padding: 20px;
        margin: 20px 0;
    }}
    
    /* Warning Box */
    .warning-box {{
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
        padding: 20px;
        margin: 20px 0;
    }}
    
    /* Error Box */
    .error-box {{
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
        padding: 20px;
        margin: 20px 0;
    }}
    
    /* Card Style */
    .card {{
        background: white;
        padding: 25px;
        border: 1px solid #ddd;
        margin: 20px 0;
    }}
    
    /* Button Styling */
    .stButton > button {{
        background-color: {COLORS['maroon']};
        color: white;
        border: 2px solid {COLORS['dark_gold']};
        padding: 12px 30px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s;
    }}
    
    .stButton > button:hover {{
        background-color: {COLORS['dark_gold']};
        border-color: {COLORS['maroon']};
    }}
    
    /* Section Divider */
    .section-divider {{
        height: 2px;
        background-color: {COLORS['gold']};
        margin: 40px 0;
    }}
    
    /* Tips Card */
    .tips-card {{
        background-color: #fffbf0;
        border: 2px solid {COLORS['gold']};
        padding: 20px;
        margin: 20px 0;
    }}
    
    /* Result Card */
    .result-card {{
        background: white;
        padding: 30px;
        border: 2px solid {COLORS['maroon']};
        margin: 20px 0;
    }}
    
    /* Upload Section */
    .upload-section {{
        background: white;
        padding: 30px;
        border: 2px dashed {COLORS['maroon']};
        text-align: center;
        margin: 20px 0;
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    
    /* Hide Sidebar Completely */
    [data-testid="stSidebar"] {{
        display: none;
    }}
    
    [data-testid="collapsedControl"] {{
        display: none;
    }}
</style>
""", unsafe_allow_html=True)



# Utility Functions
def get_logo_base64():
    """Convert logo image to base64 for embedding"""
    try:
        logo_path = os.path.join(os.path.dirname(__file__), 'imgae', 'Amulet-AI_logo.png')
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return ""
    except:
        return ""

def get_other_logos():
    """Get other partnership logos"""
    try:
        logos = {}
        logo_dir = os.path.join(os.path.dirname(__file__), 'imgae')
        
        # Thai-Austrian Logo
        thai_logo_path = os.path.join(logo_dir, 'Logo Thai-Austrain.gif')
        if os.path.exists(thai_logo_path):
            with open(thai_logo_path, "rb") as f:
                logos["thai_austrian"] = base64.b64encode(f.read()).decode()
        
        # DEPA Logo
        depa_logo_path = os.path.join(logo_dir, 'LogoDEPA-01.png')
        if os.path.exists(depa_logo_path):
            with open(depa_logo_path, "rb") as f:
                logos["depa"] = base64.b64encode(f.read()).decode()
        
        return logos
    except:
        return {}

def check_api_health():
    """ตรวจสอบสถานะ API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_model_status():
    """ตรวจสอบสถานะโมเดล"""
    model_files = [
        "trained_model/classifier.joblib",
        "trained_model/scaler.joblib", 
        "trained_model/label_encoder.joblib"
    ]
    
    missing_files = []
    for file_path in model_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

@error_handler("frontend")
def classify_image(uploaded_file, debug_mode=False):
    """จำแนกรูปภาพ - รองรับทั้ง API และ Local"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Validate image
        try:
            validate_image_file(temp_path)
        except:
            pass  # Continue if validation fails
        
        # Make prediction
        if debug_mode:
            st.write("🔧 Debug: Making API request...")
        
        # Try API first
        try:
            with open(temp_path, "rb") as f:
                files = {"file": f}
                response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                result["method"] = "API"
                return result
        except:
            if debug_mode:
                st.warning("API unavailable, using local prediction...")
        
        # Fallback to local prediction
        result = local_prediction(temp_path)
        result["method"] = "Local"
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "method": "None"
        }

def local_prediction(image_path):
    """การทำนายแบบ local"""
    try:
        import joblib
        
        # Load model components
        classifier = joblib.load("trained_model/classifier.joblib")
        scaler = joblib.load("trained_model/scaler.joblib")
        label_encoder = joblib.load("trained_model/label_encoder.joblib")
        
        # Process image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        features = image_normalized.flatten()
        
        # Make prediction
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        # Load class labels
        try:
            with open("ai_models/labels.json", "r", encoding="utf-8") as f:
                labels = json.load(f)
            thai_name = labels.get("current_classes", {}).get(str(prediction), predicted_class)
        except:
            thai_name = predicted_class
        
        return {
            "status": "success",
            "predicted_class": predicted_class,
            "thai_name": thai_name,
            "confidence": confidence,
            "probabilities": {
                label_encoder.classes_[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def predict_image_api(uploaded_file):
    """ส่งรูปภาพไป API เพื่อทำนาย (สำหรับ dual image mode)"""
    try:
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
            
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def display_classification_result(result, show_confidence=True, show_probabilities=True):
    """แสดงผลการจำแนก"""
    if result.get("status") == "success" or "predicted_class" in result:
        # Success result
        predicted_class = result.get('predicted_class', result.get('class', 'Unknown'))
        thai_name = result.get('thai_name', predicted_class)
        confidence = result.get('confidence', 0)
        
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ ผลการจำแนก</h3>
            <p><strong>ประเภทพระเครื่อง:</strong> {predicted_class}</p>
            <p><strong>ชื่อภาษาไทย:</strong> {thai_name}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence
        if show_confidence and confidence > 0:
            st.metric("ความเชื่อมั่น", f"{confidence:.2%}")
            st.progress(confidence)
            
            # Confidence interpretation
            if confidence >= 0.9:
                st.success("🎯 ความเชื่อมั่นสูงมาก - ผลลัพธ์น่าเชื่อถือ")
            elif confidence >= 0.7:
                st.warning("⚠️ ความเชื่อมั่นปานกลาง - ควรตรวจสอบเพิ่มเติม")
            else:
                st.error("⚡ ความเชื่อมั่นต่ำ - ควรถ่ายรูปใหม่หรือปรึกษาผู้เชี่ยวชาญ")
        
        # All probabilities
        if show_probabilities and 'probabilities' in result:
            st.subheader("📊 ความน่าจะเป็นทั้งหมด")
            probs = result['probabilities']
            
            for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{class_name}:** {prob:.2%}")
                st.progress(prob)
        
        # Method used
        method = result.get('method', 'Unknown')
        st.caption(f"วิธีการทำนาย: {method}")
        
    else:
        # Error result
        error_msg = result.get('error', 'เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ')
        st.markdown(f"""
        <div class="error-box">
            <h3>❌ เกิดข้อผิดพลาด</h3>
            <p>{error_msg}</p>
        </div>
        """, unsafe_allow_html=True)

def show_system_status():
    """แสดงสถานะระบบ"""
    try:
        # Check API status
        api_healthy = check_api_health()
        api_status = "🟢 Online" if api_healthy else "🔴 Offline"
        
        # Check model status
        model_ready, missing_files = check_model_status()
        model_status = "🟢 Ready" if model_ready else "🟡 Incomplete"
        
        # Overall status
        overall_status = "🟢 Operational" if api_healthy and model_ready else "⚠️ Partial"
        
        st.markdown(f"""
        **API Server:** {api_status}  
        **Model:** {model_status}  
        **Status:** {overall_status}
        """)
        
        if missing_files:
            st.warning(f"Missing files: {', '.join(missing_files)}")
        
    except Exception as e:
        st.error(f"Error checking status: {e}")

def show_quick_stats():
    """แสดงสถิติด่วน"""
    try:
        # Performance metrics
        metrics = performance_monitor.collect_metrics()
        
        if metrics and "system" in metrics:
            st.metric("CPU Usage", f"{metrics['system']['cpu_percent']:.1f}%")
            st.metric("Memory Usage", f"{metrics['system']['memory_percent']:.1f}%")
        
        # Model info
        model_info_path = Path("trained_model/model_info.json")
        if model_info_path.exists():
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            st.metric("Model Version", model_info.get("version", "Unknown"))
            st.metric("Accuracy", f"{model_info.get('training_results', {}).get('test_accuracy', 0):.1%}")
        
    except Exception as e:
        st.warning(f"Stats unavailable: {e}")

def run_health_check():
    """รันการตรวจสุขภาพระบบ"""
    try:
        health = {
            "timestamp": time.time(),
            "api_status": "checking...",
            "model_status": "checking...",
            "disk_space": "checking..."
        }
        
        # Check API
        api_healthy = check_api_health()
        health["api_status"] = "online" if api_healthy else "offline"
        
        # Check model files
        model_ready, missing_files = check_model_status()
        health["model_status"] = "ready" if model_ready else f"missing: {missing_files}"
        
        # Check disk space
        try:
            disk_usage = psutil.disk_usage('.')
            health["disk_space"] = f"{disk_usage.free / (1024**3):.1f} GB free"
        except:
            health["disk_space"] = "unknown"
        
        return health
        
    except Exception as e:
        return {"error": str(e)}

def main():
    # Initialize session state
    if 'front_camera_image' not in st.session_state:
        st.session_state.front_camera_image = None
    if 'back_camera_image' not in st.session_state:
        st.session_state.back_camera_image = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Get logos
    amulet_logo = get_logo_base64()
    other_logos = get_other_logos()
    
    # Build logos HTML for new layout
    amulet_logo_html = ""
    if amulet_logo:
        amulet_logo_html = f'<img src="data:image/png;base64,{amulet_logo}" class="logo-img" alt="Amulet-AI Logo">'
    
    partner_logos_html = ""
    if 'thai_austrian' in other_logos:
        partner_logos_html += f'<img src="data:image/gif;base64,{other_logos["thai_austrian"]}" class="logo-img-small" alt="Thai-Austrian Logo">'
    
    if 'depa' in other_logos:
        partner_logos_html += f'<img src="data:image/png;base64,{other_logos["depa"]}" class="logo-img-small" alt="DEPA Logo">'
    
    # Header with new logo layout - Amulet left, Partners right
    st.markdown(f"""
    <div class="logo-header">
        <div class="logo-left">
            {amulet_logo_html}
            <div>
                <h2 style="margin: 0; color: {COLORS['maroon']};">Amulet-AI</h2>
                <p style="margin: 0; color: {COLORS['gray']}; font-size: 0.9rem;">ระบบจำแนกพระเครื่องอัจฉริยะ</p>
            </div>
        </div>
        <div class="logo-right">
            {partner_logos_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown(f"""
    <div class="main-header">
        <h1 style="font-size: 3rem; font-weight: 900; margin: 0;">
            🏺 ระบบจำแนกพระเครื่องด้วย AI
        </h1>
        <p style="font-size: 1.3rem; margin-top: 15px;">
            Thai Amulet Classification System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Status Check
    api_healthy = check_api_health()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        status_class = "status-online" if api_healthy else "status-offline"
        status_text = "ระบบพร้อมใช้งาน" if api_healthy else "ใช้โหมด Local"
        status_icon = "🟢" if api_healthy else "🔴"
        
        st.markdown(f"""
        <div class="{status_class} status-badge">
            {status_icon} {status_text}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Control Panel (แทน Sidebar)
    with st.expander("⚙️ ตัวเลือกการใช้งาน", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎯 โหมดการวิเคราะห์")
            analysis_mode = st.radio(
                "เลือกโหมด:",
                ["รูปเดียว", "สองด้าน (หน้า+หลัง)"],
                index=1
            )
        
        with col2:
            st.markdown("### 🔧 การแสดงผล")
            show_confidence = st.checkbox("แสดงค่าความเชื่อมั่น", value=True)
            show_probabilities = st.checkbox("แสดงความน่าจะเป็นทั้งหมด", value=True)
        
        with col3:
            st.markdown("### ⚡ คำสั่งด่วน")
            if st.button("🔄 รีเฟรชระบบ", use_container_width=True):
                st.rerun()
            
            if st.button("🗑️ ล้างประวัติ", use_container_width=True):
                st.session_state.analysis_history = []
                st.session_state.front_camera_image = None
                st.session_state.back_camera_image = None
                st.success("ล้างประวัติเรียบร้อย!")
                time.sleep(1)
                st.rerun()
    
    # Get values from expander or use defaults
    if 'analysis_mode' not in locals():
        analysis_mode = "สองด้าน (หน้า+หลัง)"
        show_confidence = True
        show_probabilities = True
    
    # Main content - Image Classification
    if analysis_mode == "รูปเดียว":
        image_classification_tab("Single Image", show_confidence, show_probabilities, False)
    else:
        image_classification_tab("Dual Image (Front + Back)", show_confidence, show_probabilities, False)

def image_classification_tab(analysis_mode, show_confidence, show_probabilities, debug_mode):
    """Tab สำหรับการจำแนกรูปภาพ"""
    
    st.header("🖼️ Upload & Classify Amulet Images")
    
    if analysis_mode == "Single Image":
        # Single image mode
        st.markdown("### อัปโหลดรูปพระเครื่อง")
        
        # Upload methods
        upload_method = st.radio("เลือกวิธีการอัปโหลด:", ["📁 อัปโหลดไฟล์", "📸 ถ่ายรูป"])
        
        uploaded_image = None
        
        if upload_method == "📁 อัปโหลดไฟล์":
            uploaded_image = st.file_uploader(
                "เลือกรูปพระเครื่อง",
                type=['jpg', 'jpeg', 'png'],
                help="อัปโหลดรูปภาพที่ชัดเจนของพระเครื่อง"
            )
        
        elif upload_method == "📸 ถ่ายรูป":
            st.info("📷 กรุณาอนุญาตให้เข้าถึงกล้องในเบราว์เซอร์ของคุณ")
            uploaded_image = st.camera_input("ถ่ายรูปพระเครื่อง")
        
        # Display and analyze
        if uploaded_image is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(uploaded_image, caption="รูปที่อัปโหลด", use_column_width=True)
            
            with col2:
                if st.button("🔍 เริ่มการวิเคราะห์", type="primary"):
                    with st.spinner("AI กำลังวิเคราะห์รูปภาพ... กรุณารอสักครู่"):
                        start_time = time.time()
                        result = classify_image(uploaded_image, debug_mode)
                        processing_time = time.time() - start_time
                        
                        st.success("✅ การวิเคราะห์เสร็จสิ้น!")
                        st.metric("เวลาประมวลผล", f"{processing_time:.2f} วินาที")
                        
                        display_classification_result(result, show_confidence, show_probabilities)
                        
                        # Add to history
                        st.session_state.analysis_history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "result": result,
                            "processing_time": processing_time
                        })
    
    else:
        # Dual image mode
        st.markdown("### อัปโหลดรูปพระเครื่อง (ด้านหน้า และ ด้านหลัง)")
        
        col1, col2 = st.columns(2)
        
        # Front image
        with col1:
            st.markdown("#### รูปด้านหน้า")
            
            front_tab1, front_tab2 = st.tabs(["📁 อัปโหลดไฟล์", "📸 ถ่ายรูป"])
            
            front_image = None
            
            with front_tab1:
                front_image = st.file_uploader(
                    "เลือกรูปด้านหน้า",
                    type=['jpg', 'jpeg', 'png'],
                    key="front_upload"
                )
            
            with front_tab2:
                st.info("📷 เปิดกล้องถ่ายรูปด้านหน้า")
                camera_front = st.camera_input("ถ่ายรูปด้านหน้า", key="front_camera")
                if camera_front:
                    st.session_state.front_camera_image = camera_front
                    st.success("✅ ถ่ายรูปด้านหน้าสำเร็จ!")
            
            # Display front image
            display_front_image = front_image or st.session_state.front_camera_image
            if display_front_image:
                st.image(display_front_image, caption="รูปด้านหน้า", use_column_width=True)
        
        # Back image
        with col2:
            st.markdown("#### รูปด้านหลัง")
            
            back_tab1, back_tab2 = st.tabs(["📁 อัปโหลดไฟล์", "📸 ถ่ายรูป"])
            
            back_image = None
            
            with back_tab1:
                back_image = st.file_uploader(
                    "เลือกรูปด้านหลัง",
                    type=['jpg', 'jpeg', 'png'],
                    key="back_upload"
                )
            
            with back_tab2:
                st.info("📷 เปิดกล้องถ่ายรูปด้านหลัง")
                camera_back = st.camera_input("ถ่ายรูปด้านหลัง", key="back_camera")
                if camera_back:
                    st.session_state.back_camera_image = camera_back
                    st.success("✅ ถ่ายรูปด้านหลังสำเร็จ!")
            
            # Display back image
            display_back_image = back_image or st.session_state.back_camera_image
            if display_back_image:
                st.image(display_back_image, caption="รูปด้านหลัง", use_column_width=True)
        
        # Analysis section for dual images
        final_front_image = front_image or st.session_state.front_camera_image
        final_back_image = back_image or st.session_state.back_camera_image
        
        if final_front_image and final_back_image:
            st.markdown("---")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### 🎯 เริ่มการวิเคราะห์")
                st.info("✅ มีรูปภาพทั้งสองด้านแล้ว พร้อมเริ่มการวิเคราะห์")
            
            with col2:
                analyze_button = st.button("🚀 เริ่มการวิเคราะห์ด้วย AI", type="primary")
            
            if analyze_button:
                with st.spinner("🤖 AI กำลังวิเคราะห์รูปภาพทั้งสองด้าน..."):
                    start_time = time.time()
                    
                    # Analyze front image (primary)
                    front_result = classify_image(final_front_image, debug_mode)
                    
                    # Analyze back image for comparison
                    back_result = classify_image(final_back_image, debug_mode)
                    
                    processing_time = time.time() - start_time
                    
                    st.success("✅ การวิเคราะห์เสร็จสิ้น!")
                    st.metric("เวลาประมวลผล", f"{processing_time:.2f} วินาที")
                    
                    # Display results
                    st.subheader("📊 ผลการวิเคราะห์")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🖼️ ผลจากรูปด้านหน้า")
                        display_classification_result(front_result, show_confidence, show_probabilities)
                    
                    with col2:
                        st.markdown("#### 🖼️ ผลจากรูปด้านหลัง")
                        display_classification_result(back_result, show_confidence, show_probabilities)
                    
                    # Comparison
                    if (front_result.get("status") == "success" and 
                        back_result.get("status") == "success"):
                        
                        front_class = front_result.get("predicted_class", "")
                        back_class = back_result.get("predicted_class", "")
                        
                        if front_class == back_class:
                            st.success("🎯 ผลการวิเคราะห์จากทั้งสองด้านสอดคล้องกัน!")
                        else:
                            st.warning("⚠️ ผลการวิเคราะห์จากทั้งสองด้านไม่สอดคล้องกัน ควรตรวจสอบเพิ่มเติม")
                    
                    # Add to history
                    st.session_state.analysis_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "front_result": front_result,
                        "back_result": back_result,
                        "processing_time": processing_time,
                        "mode": "dual"
                    })
        
        else:
            st.markdown("""
            <div class="upload-section">
                <h3>พร้อมเริ่มการวิเคราะห์แล้วหรือยัง?</h3>
                <p>กรุณาอัปโหลดรูปภาพทั้งสองด้าน (หน้า และ หลัง) เพื่อเริ่มการวิเคราะห์ด้วย AI</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tips section
    st.markdown("---")
    st.markdown("## 💡 เคล็ดลับการใช้งาน")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📸 การถ่ายรูปที่ดี
        - ใช้แสงสว่างเพียงพอ
        - รูปภาพชัดเจน ไม่เบลอ
        - พื้นหลังสีเรียบ
        - ขนาดไฟล์ไม่เกิน 10MB
        - ถ่ายให้เห็นรายละเอียดของพระเครื่อง
        """)
    
    with col2:
        st.markdown("""
        ### 📊 การตีความผลลัพธ์
        - >90%: ความเชื่อมั่นสูงมาก
        - 70-90%: ความเชื่อมั่นปานกลาง
        - <70%: ความเชื่อมั่นต่ำ
        - เวลาประมวลผล: 2-10 วินาที
        - ใช้ข้อมูลร่วมกับความรู้ของผู้เชี่ยวชาญ
        """)



if __name__ == "__main__":
    main()