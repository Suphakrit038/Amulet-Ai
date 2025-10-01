#!/usr/bin/env python3
"""
🏺 Amulet-AI Unified Frontend Application
ระบบ Frontend แบบครบครันที่รวมฟีเจอร์ทั้งหมดเข้าด้วยกัน
Version: 3.0 Unified Edition
"""

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import json
import time
import base64
import os
import sys
from pathlib import Path
from datetime import datetime
import psutil
from io import BytesIO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import enhanced modules with fallback
try:
    from core.error_handling_enhanced import error_handler, validate_image_file
    from core.performance_monitoring import performance_monitor
except ImportError:
    # Fallback implementations
    def error_handler(error_type="general"):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    st.error(f"Error in {func.__name__}: {str(e)}")
                    return None
            return wrapper
        return decorator
    
    class performance_monitor:
        @staticmethod
        def collect_metrics():
            try:
                return {
                    "system": {
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
                    }
                }
            except:
                return {}

# Page Configuration
st.set_page_config(
    page_title="🏺 Amulet-AI - ระบบวิเคราะห์พระเครื่องด้วย AI",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Enhanced CSS with Thai Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700;800;900&display=swap');
    
    .stApp {
        font-family: 'Prompt', 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #f8f5f0 0%, #ffffff 50%, #faf7f2 100%);
        border: 4px solid #800000;
        margin: 10px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(128, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    /* Animated background */
    .stApp::before {
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
        animation: gradientMove 15s ease-in-out infinite;
        z-index: -1;
        pointer-events: none;
    }
    
    @keyframes gradientMove {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .main-header {
        background: linear-gradient(135deg, #800000 0%, #B8860B 100%);
        padding: 0;
        margin-bottom: 2rem;
        color: white;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(128, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: headerFloat 8s ease-in-out infinite;
    }
    
    @keyframes headerFloat {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(3deg); }
    }
    
    .status-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #800000;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .status-online {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        color: #059669;
        border: 2px solid rgba(16, 185, 129, 0.3);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .status-offline {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        color: #dc2626;
        border: 2px solid rgba(239, 68, 68, 0.3);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .upload-section {
        background: linear-gradient(135deg, rgba(128, 0, 0, 0.05) 0%, rgba(184, 134, 11, 0.05) 100%);
        padding: 3rem 2rem;
        border: 3px solid #800000;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #B8860B;
        background: linear-gradient(135deg, rgba(128, 0, 0, 0.08) 0%, rgba(184, 134, 11, 0.08) 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #800000 0%, #B8860B 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(128, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(128, 0, 0, 0.4);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #800000;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 2px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.1) 100%);
        border: 2px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        border: 2px solid rgba(239, 68, 68, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 2rem;
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .logo-img {
        height: 80px;
        max-width: 200px;
        object-fit: contain;
        padding: 10px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .logo-img:hover {
        transform: scale(1.05);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f5f5f5;
        border-radius: 10px;
        padding: 4px;
        border: 2px solid #ddd;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background: #f5f5f5;
        border: none;
        border-radius: 8px;
        color: #666;
        font-weight: 600;
        padding: 0 2rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e0e0e0;
        color: #800000;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #800000 0%, #B8860B 100%) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(128, 0, 0, 0.3);
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        border: 2px solid #ddd;
        border-top: none;
        padding: 1.5rem;
        background: white;
        border-radius: 0 0 10px 10px;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #ddd !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        background: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader:hover {
        border-color: #800000 !important;
        background: #fafafa !important;
    }
    
    /* Camera input styling */
    [data-testid="stCameraInput"] {
        border: 2px solid #ddd !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        background: white !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stCameraInput"]:hover {
        border-color: #800000 !important;
        box-shadow: 0 4px 12px rgba(128, 0, 0, 0.1) !important;
    }
    
    /* Image styling */
    .stImage img {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #800000 0%, #B8860B 100%);
    }
    
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #800000 0%, #B8860B 50%, #800000 100%);
        margin: 3rem 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Utility Functions
def get_logo_base64(logo_path):
    """Convert logo to base64"""
    try:
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
    except:
        pass
    return ""

def load_logos():
    """Load all logos"""
    logo_dir = Path(__file__).parent / "imgae"
    logos = {}
    
    # Main logo
    amulet_logo_path = logo_dir / "Amulet-AI_logo.png"
    if amulet_logo_path.exists():
        logos["amulet"] = get_logo_base64(amulet_logo_path)
    
    # Partner logos
    thai_logo_path = logo_dir / "Logo Thai-Austrain.gif"
    if thai_logo_path.exists():
        logos["thai_austrian"] = get_logo_base64(thai_logo_path)
    
    depa_logo_path = logo_dir / "LogoDEPA-01.png"
    if depa_logo_path.exists():
        logos["depa"] = get_logo_base64(depa_logo_path)
    
    return logos

@error_handler("api")
def check_api_health():
    """Check API server status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

@error_handler("prediction")
def predict_image(front_image, back_image=None):
    """Make prediction via API or local fallback"""
    try:
        # Prepare files for API
        files = {}
        
        if front_image:
            if hasattr(front_image, 'getvalue'):
                files['front'] = ('front.jpg', front_image.getvalue(), 'image/jpeg')
            else:
                files['front'] = ('front.jpg', front_image, 'image/jpeg')
        
        if back_image:
            if hasattr(back_image, 'getvalue'):
                files['back'] = ('back.jpg', back_image.getvalue(), 'image/jpeg')
            else:
                files['back'] = ('back.jpg', back_image, 'image/jpeg')
        
        # Try API prediction
        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            result["method"] = "API"
            return result
        else:
            return {"status": "error", "error": f"API Error: {response.status_code}", "method": "API"}
    
    except Exception as e:
        # Fallback to local prediction
        return local_prediction(front_image, back_image)

def local_prediction(front_image, back_image=None):
    """Local prediction fallback"""
    try:
        import joblib
        
        # Load model components
        model_dir = Path("trained_model")
        classifier = joblib.load(model_dir / "classifier.joblib")
        scaler = joblib.load(model_dir / "scaler.joblib") 
        label_encoder = joblib.load(model_dir / "label_encoder.joblib")
        
        # Process front image
        if hasattr(front_image, 'getvalue'):
            image_data = front_image.getvalue()
        else:
            image_data = front_image
        
        # Convert to numpy array
        image = Image.open(BytesIO(image_data))
        image = image.convert('RGB')
        image_array = np.array(image)
        
        # Resize and normalize
        image_resized = cv2.resize(image_array, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        features = image_normalized.flatten()
        
        # Scale and predict
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        
        # Get class name
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        # Load Thai names
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
            },
            "method": "Local"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "method": "Local"
        }

def show_system_status():
    """Display system status in sidebar"""
    try:
        # API Status
        api_healthy = check_api_health()
        api_status = "🟢 Online" if api_healthy else "🔴 Offline"
        
        # Model Status
        model_files = [
            "trained_model/classifier.joblib",
            "trained_model/scaler.joblib", 
            "trained_model/label_encoder.joblib"
        ]
        
        missing_files = [f for f in model_files if not Path(f).exists()]
        model_status = "🟢 Ready" if not missing_files else "🟡 Incomplete"
        
        # Overall status
        overall_status = "🟢 Operational" if api_healthy and not missing_files else "⚠️ Partial"
        
        st.markdown(f"""
        **API Server:** {api_status}  
        **Model:** {model_status}  
        **Overall:** {overall_status}
        """)
        
        # Performance metrics
        try:
            metrics = performance_monitor.collect_metrics()
            if metrics and "system" in metrics:
                st.metric("CPU Usage", f"{metrics['system']['cpu_percent']:.1f}%")
                st.metric("Memory Usage", f"{metrics['system']['memory_percent']:.1f}%")
        except:
            pass
            
    except Exception as e:
        st.error(f"Status check error: {e}")

def main():
    """Main application function"""
    
    # Initialize session state
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'front_camera_image' not in st.session_state:
        st.session_state.front_camera_image = None
    if 'back_camera_image' not in st.session_state:
        st.session_state.back_camera_image = None
    
    # Load logos
    logos = load_logos()
    
    # === HEADER SECTION ===
    # Build header with logos
    logos_html = ""
    if "amulet" in logos:
        logos_html += f'<img src="data:image/png;base64,{logos["amulet"]}" class="logo-img" alt="Amulet-AI Logo">'
    
    partner_logos = ""
    if "thai_austrian" in logos:
        partner_logos += f'<img src="data:image/gif;base64,{logos["thai_austrian"]}" class="logo-img" alt="Thai-Austrian Logo">'
    
    if "depa" in logos:
        partner_logos += f'<img src="data:image/png;base64,{logos["depa"]}" class="logo-img" alt="DEPA Logo">'
    
    # Main header
    header_html = f"""
    <div class="main-header">
        <div style="position: relative; z-index: 2; display: flex; align-items: center; justify-content: space-between; gap: 2rem; padding: 2rem;">
            <div style="display: flex; align-items: center; gap: 2rem;">
                {logos_html}
                <div>
                    <h1 style="font-size: 4rem; font-weight: 900; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.4); 
                               color: white; letter-spacing: 2px;">
                        AMULET-AI
                    </h1>
                    <h2 style="font-size: 1.8rem; margin: 1rem 0 0 0; opacity: 0.95; font-weight: 600; 
                               text-shadow: 1px 1px 3px rgba(0,0,0,0.3); color: white;">
                        ระบบวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์
                    </h2>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                {partner_logos}
            </div>
        </div>
    </div>
    """
    
    st.components.v1.html(header_html, height=280)
    
    # === SIDEBAR ===
    with st.sidebar:
        st.header("⚙️ System Control")
        show_system_status()
        
        st.header("🔧 Options")
        debug_mode = st.checkbox("Debug Mode")
        show_confidence = st.checkbox("Show Confidence Details", value=True)
        show_probabilities = st.checkbox("Show All Probabilities", value=True)
        
        st.header("📊 Quick Actions")
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    # === STATUS SECTION ===
    api_healthy = check_api_health()
    status_class = "status-online" if api_healthy else "status-offline"
    status_text = "🟢 API เชื่อมต่อสำเร็จ - ระบบพร้อมใช้งาน" if api_healthy else "🔴 API ไม่พร้อมใช้งาน - ใช้โหมด Local"
    
    st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
    
    # === MAIN CONTENT TABS ===
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Classification", "📊 Analytics", "📚 Documentation", "🔧 Tools"])
    
    with tab1:
        st.header("🖼️ Upload & Classify Amulet Images")
        
        # Upload section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("รูปด้านหน้า")
            tab_file1, tab_cam1 = st.tabs(["📁 อัปโหลดไฟล์", "📸 ถ่ายรูป"])
            
            front_image = None
            with tab_file1:
                front_image = st.file_uploader(
                    "เลือกรูปด้านหน้า",
                    type=['jpg', 'jpeg', 'png'],
                    key="front_upload"
                )
            
            with tab_cam1:
                front_camera = st.camera_input("ถ่ายรูปด้านหน้า", key="front_camera")
                if front_camera:
                    st.session_state.front_camera_image = front_camera
                    st.success("✅ ถ่ายรูปด้านหน้าสำเร็จ!")
            
            # Display front image
            display_front = front_image or st.session_state.front_camera_image
            if display_front:
                st.image(display_front, caption="รูปด้านหน้า", use_column_width=True)
        
        with col2:
            st.subheader("รูปด้านหลัง")
            tab_file2, tab_cam2 = st.tabs(["📁 อัปโหลดไฟล์", "📸 ถ่ายรูป"])
            
            back_image = None
            with tab_file2:
                back_image = st.file_uploader(
                    "เลือกรูปด้านหลัง",
                    type=['jpg', 'jpeg', 'png'],
                    key="back_upload"
                )
            
            with tab_cam2:
                back_camera = st.camera_input("ถ่ายรูปด้านหลัง", key="back_camera")
                if back_camera:
                    st.session_state.back_camera_image = back_camera
                    st.success("✅ ถ่ายรูปด้านหลังสำเร็จ!")
            
            # Display back image
            display_back = back_image or st.session_state.back_camera_image
            if display_back:
                st.image(display_back, caption="รูปด้านหลัง", use_column_width=True)
        
        # Analysis section
        final_front = front_image or st.session_state.front_camera_image
        final_back = back_image or st.session_state.back_camera_image
        
        if final_front:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("🎯 เริ่มการวิเคราะห์")
                
            with col2:
                analyze_button = st.button("🔍 วิเคราะห์พระเครื่อง", type="primary")
            
            if analyze_button:
                with st.spinner("🤖 AI กำลังวิเคราะห์รูปภาพ... กรุณารอสักครู่"):
                    start_time = time.time()
                    
                    # Make prediction
                    result = predict_image(final_front, final_back)
                    
                    processing_time = time.time() - start_time
                    
                    # Display results
                    if result["status"] == "success":
                        # Success result
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ การวิเคราะห์เสร็จสิ้น!</h3>
                            <p><strong>ประเภทพระเครื่อง:</strong> {result.get('predicted_class', 'Unknown')}</p>
                            <p><strong>ชื่อไทย:</strong> {result.get('thai_name', result.get('predicted_class', 'Unknown'))}</p>
                            <p><strong>วิธีการ:</strong> {result.get('method', 'Unknown')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            confidence = result.get('confidence', 0)
                            st.metric("ความเชื่อมั่น", f"{confidence:.1%}")
                            st.progress(confidence)
                        
                        with col2:
                            st.metric("เวลาประมวลผล", f"{processing_time:.2f} วินาที")
                        
                        # Confidence interpretation
                        if confidence > 0.9:
                            st.success("🎯 ความเชื่อมั่นสูงมาก - ผลลัพธ์น่าเชื่อถือ")
                        elif confidence > 0.7:
                            st.warning("⚠️ ความเชื่อมั่นปานกลาง - ควรตรวจสอบเพิ่มเติม")
                        else:
                            st.error("⚡ ความเชื่อมั่นต่ำ - ควรถ่ายรูปใหม่หรือปรึกษาผู้เชี่ยวชาญ")
                        
                        # Show all probabilities if requested
                        if show_probabilities and 'probabilities' in result:
                            st.subheader("📊 ความน่าจะเป็นทั้งหมด")
                            probs = result['probabilities']
                            
                            for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                                st.write(f"**{class_name}:** {prob:.1%}")
                                st.progress(prob)
                        
                        # Add to history
                        st.session_state.analysis_history.append({
                            "timestamp": datetime.now(),
                            "result": result,
                            "processing_time": processing_time
                        })
                        
                    else:
                        # Error result
                        st.markdown(f"""
                        <div class="error-box">
                            <h3>❌ เกิดข้อผิดพลาด</h3>
                            <p>{result.get('error', 'Unknown error occurred')}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="upload-section">
                <h3 style="color: #800000; margin-bottom: 1rem;">พร้อมเริ่มการวิเคราะห์แล้วหรือยัง?</h3>
                <p style="color: #666; font-size: 1.1rem;">
                    กรุณาอัปโหลดรูปภาพด้านหน้าอย่างน้อย 1 รูป เพื่อเริ่มการวิเคราะห์ด้วย AI<br/>
                    (รูปด้านหลังเป็นทางเลือก แต่จะช่วยเพิ่มความแม่นยำ)
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("📊 System Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            try:
                metrics = performance_monitor.collect_metrics()
                if metrics and "system" in metrics:
                    st.json(metrics)
                else:
                    st.info("Performance data not available")
            except:
                st.warning("Performance monitoring not available")
        
        with col2:
            st.subheader("Analysis History")
            if st.session_state.analysis_history:
                st.write(f"Total analyses: {len(st.session_state.analysis_history)}")
                
                for i, entry in enumerate(reversed(st.session_state.analysis_history[-5:])):
                    with st.expander(f"Analysis {len(st.session_state.analysis_history)-i}"):
                        st.write(f"**Time:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Result:** {entry['result'].get('predicted_class', 'Error')}")
                        st.write(f"**Confidence:** {entry['result'].get('confidence', 0):.1%}")
                        st.write(f"**Processing Time:** {entry['processing_time']:.2f}s")
            else:
                st.info("No analysis history yet")
    
    with tab3:
        st.header("📚 Documentation")
        
        st.markdown("""
        ## 🏺 About Amulet-AI Unified
        
        Amulet-AI Unified เป็นระบบจำแนกพระเครื่องอัจฉริยะที่รวมฟีเจอร์ทั้งหมดเข้าด้วยกัน
        
        ### 🎯 Supported Amulet Types
        - พระศิวลี (Phra Sivali)
        - พระสมเด็จ (Phra Somdej)  
        - ปรกโพธิ์ 9 ใบ (Prok Bodhi 9 Leaves)
        - แหวกม่าน (Waek Man)
        - หลังรูปเหมือน (Portrait Back)
        - วัดหนองอีดุก (Wat Nong E Duk)
        
        ### 🔧 Technical Features
        - **Dual Mode:** API และ Local prediction
        - **Multiple Upload:** ไฟล์อัปโหลดและกล้อง
        - **Real-time Status:** ตรวจสอบสถานะระบบ
        - **Performance Monitor:** ติดตามประสิทธิภาพ
        - **Analysis History:** บันทึกประวัติการวิเคราะห์
        
        ### 🚀 How to Use
        1. อัปโหลดรูปพระเครื่อง (หน้า + หลัง หรือ หน้าอย่างเดียว)
        2. คลิก "วิเคราะห์พระเครื่อง"
        3. ดูผลลัพธ์และค่าความเชื่อมั่น
        4. ตรวจสอบประวัติการวิเคราะห์ใน Analytics tab
        
        ### ⚡ Performance Tips
        - ใช้แสงสว่างเพียงพอ
        - รูปภาพชัดเจน ไม่เบลอ
        - หลีกเลี่ยงเงาหรือแสงสะท้อน
        - ขนาดไฟล์ไม่เกิน 10MB
        """)
    
    with tab4:
        st.header("🔧 System Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Health Checks")
            if st.button("🏥 Run Full Health Check"):
                with st.spinner("Checking system health..."):
                    health_results = {
                        "timestamp": datetime.now().isoformat(),
                        "api_status": "online" if check_api_health() else "offline",
                        "model_files": [],
                        "missing_files": []
                    }
                    
                    # Check model files
                    model_files = [
                        "trained_model/classifier.joblib",
                        "trained_model/scaler.joblib",
                        "trained_model/label_encoder.joblib"
                    ]
                    
                    for file_path in model_files:
                        if Path(file_path).exists():
                            health_results["model_files"].append(file_path)
                        else:
                            health_results["missing_files"].append(file_path)
                    
                    st.json(health_results)
        
        with col2:
            st.subheader("Cache & Storage")
            if st.button("🗑️ Clear All Cache"):
                st.cache_data.clear()
                if 'analysis_history' in st.session_state:
                    st.session_state.analysis_history = []
                st.success("All cache cleared!")
            
            if st.button("💾 Export Analysis History"):
                if st.session_state.analysis_history:
                    import json
                    history_json = json.dumps(
                        st.session_state.analysis_history, 
                        default=str, 
                        indent=2, 
                        ensure_ascii=False
                    )
                    st.download_button(
                        "📥 Download History",
                        history_json,
                        "analysis_history.json",
                        "application/json"
                    )
                else:
                    st.info("No history to export")
    
    # === FOOTER ===
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Tips Section
    st.markdown("## 💡 เคล็ดลับการใช้งาน")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### การถ่ายรูปที่ดี
        - ใช้แสงสว่างเพียงพอ
        - รูปภาพชัดเจน ไม่เบลอ
        - พื้นหลังสีเรียบ
        - มุมกล้องตั้งฉาก
        - ขนาดไฟล์ไม่เกิน 10MB
        """)
    
    with col2:
        st.markdown("""
        ### การตีความผลลัพธ์
        - >90%: ความเชื่อมั่นสูงมาก
        - 70-90%: ความเชื่อมั่นปานกลาง
        - <70%: ความเชื่อมั่นต่ำ
        - API: เร็วกว่า, Local: ปลอดภัยกว่า
        - เวลาประมวลผล: 2-10 วินาที
        """)
    
    # Final footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%); 
                border-radius: 10px; margin-top: 2rem;">
        <h3 style="margin: 0; color: #800000;">🏺 Amulet-AI Unified v3.0</h3>
        <p style="margin: 0.5rem 0 0 0; color: #666;">
            © 2025 Amulet-AI Project | Powered by Advanced AI Technology
        </p>
        <p style="margin: 0.5rem 0 0 0; color: #888; font-size: 0.9rem;">
            รวมฟีเจอร์ทั้งหมดในไฟล์เดียว - ใช้งานง่าย ครบครัน
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()