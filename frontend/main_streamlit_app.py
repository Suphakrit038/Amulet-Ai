#!/usr/bin/env python3
"""
🏺 Amulet-AI Unified Frontend
ระบบ Frontend รวมฟีเจอร์ทั้งหมดในไฟล์เดียว - สะดวกใช้งาน
รวมฟีเจอร์จาก enhanced, simple, และ production version
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

# Theme Colors
THEME_COLORS = {
    'primary': '#800000',   # แดงเลือดหมู
    'accent': '#B8860B',    # ทอง
    'gold': '#D4AF37',      # ทองสว่าง
    'success': '#10b981',
    'warning': '#f59e0b',
    'error': '#ef4444',
    'info': '#3b82f6'
}

# Page Configuration
st.set_page_config(
    page_title="🏺 Amulet-AI Unified System",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS Styling
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700;800;900&display=swap');
    
    .stApp {{
        font-family: 'Prompt', 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #f8f5f0 0%, #ffffff 50%, #faf7f2 100%);
        border: 4px solid {THEME_COLORS['primary']};
        margin: 10px;
        box-shadow: 0 8px 32px rgba(128, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }}
    
    /* Wave Animation Background */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, 
            rgba(128, 0, 0, 0.05) 0%, 
            rgba(184, 134, 11, 0.05) 25%, 
            rgba(128, 0, 0, 0.03) 50%, 
            rgba(184, 134, 11, 0.05) 75%, 
            rgba(128, 0, 0, 0.05) 100%);
        background-size: 400% 400%;
        animation: waveMove 15s ease-in-out infinite;
        z-index: -1;
        pointer-events: none;
    }}
    
    @keyframes waveMove {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    
    .main-header {{
        background: linear-gradient(135deg, {THEME_COLORS['primary']} 0%, {THEME_COLORS['accent']} 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(128, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: headerFloat 8s ease-in-out infinite;
    }}
    
    @keyframes headerFloat {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-15px) rotate(3deg); }}
    }}
    
    .metric-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid {THEME_COLORS['primary']};
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }}
    
    .success-box {{
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid {THEME_COLORS['success']};
    }}
    
    .warning-box {{
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid {THEME_COLORS['warning']};
    }}
    
    .error-box {{
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid {THEME_COLORS['error']};
    }}
    
    .upload-section {{
        background: linear-gradient(135deg, rgba(128, 0, 0, 0.05) 0%, rgba(184, 134, 11, 0.05) 100%);
        padding: 2rem;
        border: 3px solid {THEME_COLORS['primary']};
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }}
    
    .upload-section:hover {{
        border-color: {THEME_COLORS['accent']};
        background: linear-gradient(135deg, rgba(128, 0, 0, 0.08) 0%, rgba(184, 134, 11, 0.08) 100%);
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, {THEME_COLORS['primary']} 0%, {THEME_COLORS['accent']} 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(128, 0, 0, 0.3);
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(128, 0, 0, 0.4);
    }}
    
    .logo-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 2rem;
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }}
    
    .logo-img {{
        height: 120px;
        max-width: 250px;
        object-fit: contain;
        padding: 10px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }}
    
    .logo-img:hover {{
        transform: scale(1.05);
    }}
    
    .status-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }}
    
    .status-online {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        color: #059669;
        border: 2px solid rgba(16, 185, 129, 0.3);
    }}
    
    .status-offline {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        color: #dc2626;
        border: 2px solid rgba(239, 68, 68, 0.3);
    }}
    
    .section-divider {{
        height: 3px;
        background: linear-gradient(90deg, {THEME_COLORS['primary']} 0%, {THEME_COLORS['accent']} 50%, {THEME_COLORS['primary']} 100%);
        margin: 3rem 0;
        border-radius: 2px;
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: #f5f5f5;
        border-radius: 10px;
        padding: 4px;
        border: 2px solid #ddd;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 3rem;
        background: transparent;
        border-radius: 8px;
        color: #666;
        font-weight: 600;
        padding: 0 1.5rem;
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: #e0e0e0;
        color: {THEME_COLORS['primary']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {THEME_COLORS['primary']} 0%, {THEME_COLORS['accent']} 100%) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(128, 0, 0, 0.3);
    }}
    
    .stTabs [data-baseweb="tab-panel"] {{
        padding: 1.5rem;
    }}
    
    /* File uploader styling */
    .stFileUploader {{
        border: 2px dashed #ddd;
        border-radius: 10px;
        padding: 1.5rem;
        background: white;
        transition: all 0.3s ease;
    }}
    
    .stFileUploader:hover {{
        border-color: {THEME_COLORS['primary']};
        background: #fafafa;
    }}
    
    /* Camera input styling */
    [data-testid="stCameraInput"] {{
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        background: white;
        transition: all 0.3s ease;
    }}
    
    [data-testid="stCameraInput"]:hover {{
        border-color: {THEME_COLORS['primary']};
        box-shadow: 0 4px 12px rgba(128, 0, 0, 0.1);
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
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    # Get logos
    amulet_logo = get_logo_base64()
    other_logos = get_other_logos()
    
    # Build logos HTML
    logos_html = ""
    if amulet_logo:
        logos_html += f'<img src="data:image/png;base64,{amulet_logo}" class="logo-img" alt="Amulet-AI Logo">'
    
    partner_logos = ""
    if 'thai_austrian' in other_logos:
        partner_logos += f'<img src="data:image/gif;base64,{other_logos["thai_austrian"]}" class="logo-img" alt="Thai-Austrian Logo">'
    
    if 'depa' in other_logos:
        partner_logos += f'<img src="data:image/png;base64,{other_logos["depa"]}" class="logo-img" alt="DEPA Logo">'
    
    # Header with logos
    st.markdown(f"""
    <div class="logo-container">
        {logos_html}
        {partner_logos}
    </div>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <div style="position: relative; z-index: 2;">
            <h1 style="font-size: 4rem; font-weight: 900; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.4);">
                🏺 AMULET-AI UNIFIED
            </h1>
            <h2 style="font-size: 1.8rem; margin: 1rem 0 0 0; opacity: 0.95; font-weight: 600; text-shadow: 1px 1px 3px rgba(0,0,0,0.3);">
                ระบบจำแนกพระเครื่องอัจฉริยะ - รวมฟีเจอร์ครบครัน
            </h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # System status
        st.subheader("📊 System Status")
        show_system_status()
        
        st.subheader("📈 Quick Stats")
        show_quick_stats()
        
        st.subheader("🔧 Advanced Options")
        debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        st.session_state.debug_mode = debug_mode
        
        show_confidence = st.checkbox("Show Confidence Details", value=True)
        show_probabilities = st.checkbox("Show All Probabilities", value=True)
        
        # Analysis mode
        st.subheader("🎯 Analysis Mode")
        analysis_mode = st.radio(
            "Select mode:",
            ["Single Image", "Dual Image (Front + Back)"],
            index=1
        )
    
    # API Status Check
    api_healthy = check_api_health()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        status_class = "status-online" if api_healthy else "status-offline"
        status_text = "API เชื่อมต่อสำเร็จ - ระบบพร้อมใช้งาน" if api_healthy else "API ไม่พร้อมใช้งาน - ใช้โหมด Local"
        status_indicator = "●" if api_healthy else "●"
        
        st.markdown(f"""
        <div class="{status_class}">
            {status_indicator} {status_text}
        </div>
        """, unsafe_allow_html=True)
    
    # Section divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Image Classification", 
        "📊 System Analytics", 
        "📚 Documentation", 
        "🔧 System Tools"
    ])
    
    with tab1:
        image_classification_tab(analysis_mode, show_confidence, show_probabilities, debug_mode)
    
    with tab2:
        system_analytics_tab()
    
    with tab3:
        documentation_tab()
    
    with tab4:
        system_tools_tab()

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

def system_analytics_tab():
    """Tab สำหรับ analytics"""
    st.header("📊 System Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Performance Metrics")
        try:
            metrics = performance_monitor.collect_metrics()
            if metrics and "system" in metrics:
                st.json(metrics)
            else:
                st.info("ไม่มีข้อมูล performance")
        except:
            st.warning("Performance monitoring ไม่พร้อมใช้งาน")
    
    with col2:
        st.subheader("🤖 Model Statistics")
        try:
            model_info_path = Path("trained_model/model_info.json")
            if model_info_path.exists():
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                
                st.metric("Training Accuracy", f"{model_info.get('training_results', {}).get('train_accuracy', 0):.1%}")
                st.metric("Validation Accuracy", f"{model_info.get('training_results', {}).get('val_accuracy', 0):.1%}")
                st.metric("Test Accuracy", f"{model_info.get('training_results', {}).get('test_accuracy', 0):.1%}")
            else:
                st.info("ไม่มีข้อมูล model statistics")
                
        except Exception as e:
            st.error(f"ไม่สามารถโหลดข้อมูล model: {e}")
    
    # Analysis history
    if st.session_state.analysis_history:
        st.subheader("📈 Analysis History")
        
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-10:])):
            with st.expander(f"Analysis #{len(st.session_state.analysis_history)-i}: {analysis['timestamp']}"):
                if analysis.get("mode") == "dual":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Front Result:**")
                        st.json(analysis["front_result"])
                    with col2:
                        st.write("**Back Result:**")
                        st.json(analysis["back_result"])
                else:
                    st.json(analysis["result"])
                st.write(f"**Processing Time:** {analysis['processing_time']:.2f} seconds")

def documentation_tab():
    """Tab สำหรับเอกสาร"""
    st.header("📚 Documentation")
    
    st.markdown("""
    ## 🏺 About Amulet-AI Unified
    
    Amulet-AI Unified เป็นระบบจำแนกพระเครื่องไทยที่รวมฟีเจอร์ทั้งหมดไว้ในไฟล์เดียว 
    เพื่อความสะดวกในการใช้งานและการบำรุงรักษา
    
    ### 🎯 Supported Amulet Types
    - **พระศิวลี** (Phra Sivali) - พระป้องกันภัยอันตราย
    - **พระสมเด็จ** (Phra Somdej) - พระยอดนิยมที่สุด
    - **ปรกโพธิ์ 9 ใบ** (Prok Bodhi 9 Leaves) - พระคุ้มครองจากภัยธรรมชาติ
    - **แหวกม่าน** (Waek Man) - พระแคล้วคลาดภัยอันตราย
    - **หลังรูปเหมือน** (Portrait Back) - พระที่มีรูปเหมือนหลัง
    - **วัดหนองอีดุก** (Wat Nong E Duk) - พระจากวัดหนองอีดุก
    
    ### 🔧 Technical Specifications
    - **Model Type:** Random Forest Classifier
    - **Image Processing:** OpenCV + PIL
    - **Input Size:** 224x224 pixels
    - **Features:** 150,528 (flattened pixels)
    - **Accuracy:** ~72% on test dataset
    - **API Framework:** FastAPI
    - **Frontend:** Streamlit
    
    ### 🚀 Features Included
    
    #### 📱 From Enhanced Version
    - Multi-tab interface
    - System status monitoring
    - Performance analytics
    - Debug mode
    - Enhanced error handling
    
    #### 🎨 From Simple Version  
    - Thai-themed UI design
    - Camera support
    - Loading animations
    - Partnership logos
    - Responsive design
    
    #### ⚡ From Production Version
    - Modular components
    - Performance monitoring
    - Advanced image processing
    - Comprehensive error handling
    - Production-ready architecture
    
    ### 🎯 Analysis Modes
    
    #### Single Image Mode
    - อัปโหลดรูปเดียว
    - เหมาะสำหรับการทดสอบเร็ว
    - รองรับ file upload และ camera
    
    #### Dual Image Mode  
    - อัปโหลดรูปทั้งหน้าและหลัง
    - การวิเคราะห์ที่แม่นยำขึ้น
    - เปรียบเทียบผลลัพธ์จากทั้งสองด้าน
    
    ### 🔄 How It Works
    
    1. **Image Upload** - อัปโหลดรูปหรือถ่ายด้วยกล้อง
    2. **Preprocessing** - ปรับขนาดเป็น 224x224, normalize
    3. **Feature Extraction** - แปลงเป็น feature vector
    4. **Classification** - ใช้ Random Forest predict
    5. **Post-processing** - แปลงผลลัพธ์เป็นชื่อไทย
    
    ### 🛠️ API Endpoints
    - `GET /health` - ตรวจสอบสถานะ API
    - `POST /predict` - ทำนายรูปภาพ  
    - `GET /classes` - ดูรายการคลาส
    - `GET /model-info` - ข้อมูลโมเดล
    
    ### ⚠️ Limitations
    - รองรับเฉพาะพระเครื่องไทย 6 ประเภท
    - ต้องการรูปภาพที่ชัดเจน
    - ความแม่นยำขึ้นอยู่กับคุณภาพรูปภาพ
    - ไม่ใช่การประเมินราคา
    
    ### 🔧 System Requirements
    - Python 3.8+
    - RAM: 4GB+
    - Storage: 2GB+
    - Browser: Chrome, Firefox, Safari
    - Internet connection (for API mode)
    """)

def system_tools_tab():
    """Tab สำหรับเครื่องมือระบบ"""
    st.header("🔧 System Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 Health Checks")
        if st.button("🩺 Run Complete Health Check"):
            with st.spinner("กำลังตรวจสอบสุขภาพระบบ..."):
                health_results = run_health_check()
                st.json(health_results)
        
        st.subheader("🔄 System Actions")
        if st.button("🔄 Refresh System Status"):
            st.rerun()
        
        if st.button("🧹 Clear Analysis History"):
            st.session_state.analysis_history = []
            st.success("ล้างประวัติการวิเคราะห์เรียบร้อย!")
    
    with col2:
        st.subheader("💾 Cache Management")
        if st.button("🗑️ Clear Cache"):
            try:
                # Clear Streamlit cache
                st.cache_data.clear()
                st.success("ล้าง cache เรียบร้อย!")
            except:
                st.warning("ไม่สามารถล้าง cache ได้")
        
        st.subheader("📊 System Information")
        try:
            sys_info = {
                "Python Version": sys.version,
                "Streamlit Version": st.__version__,
                "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Working Directory": os.getcwd()
            }
            st.json(sys_info)
        except Exception as e:
            st.error(f"ไม่สามารถดึงข้อมูลระบบได้: {e}")
    
    # Export/Import settings
    st.subheader("⚙️ Settings Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Settings"):
            settings = {
                "debug_mode": st.session_state.get("debug_mode", False),
                "analysis_history_count": len(st.session_state.get("analysis_history", [])),
                "export_time": datetime.now().isoformat()
            }
            st.json(settings)
            st.success("Settings exported!")
    
    with col2:
        if st.button("📥 Reset to Defaults"):
            # Reset session state
            for key in list(st.session_state.keys()):
                if key.startswith(('front_', 'back_', 'debug_', 'analysis_')):
                    del st.session_state[key]
            st.success("รีเซ็ตการตั้งค่าเรียบร้อย!")
            st.rerun()

if __name__ == "__main__":
    main()