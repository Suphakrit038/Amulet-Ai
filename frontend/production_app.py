#!/usr/bin/env python3
"""
🏺 Amulet-AI - Production Frontend
ระบบจำแนกพระเครื่องอัจฉริยะ
Thai Amulet Classification System
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
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import     # Main Header
    st.markdown(f"""
    <div class="main-header">
        <h1 style="font-size: 3rem; font-weight: 900; margin: 0;">
            ระบบจำแนกพระเครื่องด้วย AI
        </h1>
        <p style="font-size: 1.3rem; margin-top: 15px;">
            Thai Amulet Classification System
        </p>
    </div>
    """, unsafe_allow_html=True)dules (with fallback)
try:
    from core.error_handling_enhanced import error_handler, validate_image_file
    from core.performance_monitoring import performance_monitor
except:
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
COLORS = {
    'maroon': '#8B0000',
    'gold': '#DAA520',
    'dark_gold': '#B8860B',
    'light_gold': '#F4E4BC',
    'green': '#28a745',
    'blue': '#007bff',
    'yellow': '#ffc107',
    'red': '#dc3545',
    'gray': '#6c757d',
    'white': '#ffffff',
    'black': '#000000'
}

# Page Configuration
st.set_page_config(
    page_title="Amulet-AI - ระบบจำแนกพระเครื่อง",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Production CSS
st.markdown(f"""
<style>
    /* Basic App Styling */
    .stApp {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }}
    
    /* Logo Header Container */
    .logo-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 40px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 30px;
        border-radius: 10px;
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
        background: linear-gradient(135deg, {COLORS['maroon']} 0%, {COLORS['dark_gold']} 100%);
        color: white;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }}
    
    /* Status Badge */
    .status-badge {{
        display: inline-block;
        padding: 10px 25px;
        border-radius: 25px;
        font-weight: bold;
        margin: 20px 0;
        font-size: 1.1rem;
    }}
    
    .status-online {{
        background-color: #28a745;
        color: white;
    }}
    
    .status-offline {{
        background-color: #dc3545;
        color: white;
    }}
    
    /* Success Box */
    .success-box {{
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    /* Info Box */
    .info-box {{
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 5px solid #17a2b8;
        color: #0c5460;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    /* Warning Box */
    .warning-box {{
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
        color: #856404;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    /* Error Box */
    .error-box {{
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    /* Card Style */
    .card {{
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 20px 0;
    }}
    
    /* Button Styling */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['maroon']} 0%, {COLORS['dark_gold']} 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }}
    
    /* Section Divider */
    .section-divider {{
        height: 3px;
        background: linear-gradient(90deg, transparent 0%, {COLORS['gold']} 50%, transparent 100%);
        margin: 40px 0;
    }}
    
    /* Tips Card */
    .tips-card {{
        background: linear-gradient(135deg, #fff9e6 0%, #ffe6cc 100%);
        border-left: 5px solid {COLORS['gold']};
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }}
    
    /* Result Card */
    .result-card {{
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        margin: 20px 0;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['maroon']} 0%, {COLORS['dark_gold']} 100%);
    }}
    
    [data-testid="stSidebar"] .card {{
        background: rgba(255, 255, 255, 0.95);
    }}
    
    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
</style>
""", unsafe_allow_html=True)

# Utility Functions
def get_logo_base64():
    """Convert logo image to base64"""
    try:
        logo_path = os.path.join(os.path.dirname(__file__), 'imgae', 'Amulet-AI_logo.png')
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return ""

def get_other_logos():
    """Get partnership logos"""
    logos = {}
    try:
        logo_dir = os.path.join(os.path.dirname(__file__), 'imgae')
        
        thai_logo = os.path.join(logo_dir, 'Logo Thai-Austrain.gif')
        if os.path.exists(thai_logo):
            with open(thai_logo, "rb") as f:
                logos["thai_austrian"] = base64.b64encode(f.read()).decode()
        
        depa_logo = os.path.join(logo_dir, 'LogoDEPA-01.png')
        if os.path.exists(depa_logo):
            with open(depa_logo, "rb") as f:
                logos["depa"] = base64.b64encode(f.read()).decode()
    except:
        pass
    return logos

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
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

@error_handler("frontend")
def classify_image(uploaded_file, debug_mode=False):
    """จำแนกรูปภาพ"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Try API first
        try:
            with open(temp_path, "rb") as f:
                files = {"file": f}
                response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                result["method"] = "API"
                Path(temp_path).unlink(missing_ok=True)
                return result
        except:
            pass
        
        # Fallback to local prediction
        result = local_prediction(temp_path)
        result["method"] = "Local"
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
        
        # Load model
        classifier = joblib.load(str(project_root / "trained_model/classifier.joblib"))
        scaler = joblib.load(str(project_root / "trained_model/scaler.joblib"))
        label_encoder = joblib.load(str(project_root / "trained_model/label_encoder.joblib"))
        
        # Process image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        features = image_normalized.flatten()
        
        # Prediction
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        # Load labels
        try:
            labels_path = project_root / "ai_models/labels.json"
            with open(labels_path, "r", encoding="utf-8") as f:
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

def display_classification_result(result, show_confidence=True, show_probabilities=True):
    """แสดงผลการจำแนก"""
    if result.get("status") == "success" or "predicted_class" in result:
        predicted_class = result.get('predicted_class', 'Unknown')
        thai_name = result.get('thai_name', predicted_class)
        confidence = result.get('confidence', 0)
        
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ ผลการจำแนก</h3>
            <p style="font-size: 1.2rem;"><strong>ประเภทพระเครื่อง:</strong> {predicted_class}</p>
            <p style="font-size: 1.2rem;"><strong>ชื่อภาษาไทย:</strong> {thai_name}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if show_confidence and confidence > 0:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("ความเชื่อมั่น", f"{confidence:.1%}")
            with col2:
                st.progress(confidence)
            
            if confidence >= 0.9:
                st.success("🎯 ความเชื่อมั่นสูงมาก - ผลลัพธ์น่าเชื่อถือ")
            elif confidence >= 0.7:
                st.warning("⚠️ ความเชื่อมั่นปานกลาง - ควรตรวจสอบเพิ่มเติม")
            else:
                st.error("⚡ ความเชื่อมั่นต่ำ - ควรถ่ายรูปใหม่หรือปรึกษาผู้เชี่ยวชาญ")
        
        if show_probabilities and 'probabilities' in result:
            with st.expander("📊 ดูความน่าจะเป็นทั้งหมด"):
                probs = result['probabilities']
                for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**{class_name}**")
                    with col2:
                        st.write(f"{prob:.1%}")
                    st.progress(prob)
        
        st.caption(f"🔧 วิธีการทำนาย: {result.get('method', 'Unknown')}")
        
    else:
        error_msg = result.get('error', 'เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ')
        st.markdown(f"""
        <div class="error-box">
            <h3>❌ เกิดข้อผิดพลาด</h3>
            <p>{error_msg}</p>
        </div>
        """, unsafe_allow_html=True)

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
    
    # Logo Header
    logo_left_html = ""
    if amulet_logo:
        logo_left_html = f'<img src="data:image/png;base64,{amulet_logo}" class="logo-img" alt="Amulet-AI">'
    
    logo_right_html = ""
    if 'thai_austrian' in other_logos:
        logo_right_html += f'<img src="data:image/gif;base64,{other_logos["thai_austrian"]}" class="logo-img-small" alt="Thai-Austrian">'
    if 'depa' in other_logos:
        logo_right_html += f'<img src="data:image/png;base64,{other_logos["depa"]}" class="logo-img-small" alt="DEPA">'
    
    st.markdown(f"""
    <div class="logo-header">
        <div class="logo-left">
            {logo_left_html}
            <div>
                <h2 style="margin: 0; color: {COLORS['maroon']};">Amulet-AI</h2>
                <p style="margin: 0; color: {COLORS['gray']}; font-size: 0.9rem;">ระบบจำแนกพระเครื่องอัจฉริยะ</p>
            </div>
        </div>
        <div class="logo-right">
            {logo_right_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Header
    st.markdown(f"""
    <div class="main-header">
        <h1 style="font-size: 3rem; font-weight: 900; margin: 0;">
            🏺 ระบบจำแนกพระเครื่องด้วย AI
        </h1>
        <p style="font-size: 1.3rem; margin-top: 15px; opacity: 0.95;">
            Thai Amulet Classification System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Settings in expander (no sidebar)
    with st.expander("การตั้งค่าระบบ", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### โหมดการวิเคราะห์")
            analysis_mode = st.radio(
                "เลือกโหมด:",
                ["รูปเดียว", "สองด้าน (หน้า+หลัง)"],
                index=1,
                help="เลือกว่าต้องการวิเคราะห์รูปเดียวหรือทั้งสองด้าน"
            )
        
        with col2:
            st.markdown("### ตัวเลือกการแสดงผล")
            show_confidence = st.checkbox("แสดงค่าความเชื่อมั่น", value=True)
            show_probabilities = st.checkbox("แสดงความน่าจะเป็นทั้งหมด", value=True)
        
        with col3:
            st.markdown("### คำสั่งด่วน")
            if st.button("รีเฟรชระบบ", use_container_width=True):
                st.rerun()
            
            if st.button("ล้างประวัติ", use_container_width=True):
                st.session_state.analysis_history = []
                st.session_state.front_camera_image = None
                st.session_state.back_camera_image = None
                st.success("ล้างประวัติเรียบร้อย!")
                time.sleep(1)
                st.rerun()
    
    # Default values if expander is closed
    if 'analysis_mode' not in locals():
        analysis_mode = "สองด้าน (หน้า+หลัง)"
        show_confidence = True
        show_probabilities = True
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Main Content
    image_classification_section(analysis_mode, show_confidence, show_probabilities)
    
    # Tips Section
    show_tips_section()
    
    # History Section
    if st.session_state.analysis_history:
        show_history_section()
    
    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; color: {COLORS['gray']};">
        <p>© 2025 Amulet-AI | พัฒนาโดยความร่วมมือกับ Thai-Austrian และ DEPA</p>
        <p style="font-size: 0.9rem;">ระบบนี้ใช้ AI ช่วยในการจำแนกพระเครื่อง ผลลัพธ์ควรใช้ประกอบการตัดสินใจเท่านั้น</p>
    </div>
    """, unsafe_allow_html=True)

def image_classification_section(analysis_mode, show_confidence, show_probabilities):
    """ส่วนหลักสำหรับการจำแนกรูปภาพ"""
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("อัปโหลดรูปพระเครื่อง")
    
    if analysis_mode == "รูปเดียว":
        single_image_mode(show_confidence, show_probabilities)
    else:
        dual_image_mode(show_confidence, show_probabilities)
    
    st.markdown('</div>', unsafe_allow_html=True)

def single_image_mode(show_confidence, show_probabilities):
    """โหมดรูปเดียว"""
    
    st.markdown("### เลือกวิธีอัปโหลด")
    
    upload_tab, camera_tab = st.tabs(["อัปโหลดไฟล์", "ถ่ายรูป"])
    
    uploaded_image = None
    
    with upload_tab:
        uploaded_image = st.file_uploader(
            "เลือกรูปพระเครื่อง (JPG, JPEG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="อัปโหลดรูปภาพที่ชัดเจน ขนาดไม่เกิน 10MB"
        )
    
    with camera_tab:
        st.info("กรุณาอนุญาตให้เข้าถึงกล้องในเบราว์เซอร์")
        uploaded_image = st.camera_input("ถ่ายรูปพระเครื่อง")
    
    if uploaded_image is not None:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.image(uploaded_image, caption="รูปที่อัปโหลด", use_column_width=True)
        
        with col2:
            st.markdown("### พร้อมวิเคราะห์")
            file_size = len(uploaded_image.getvalue()) / 1024
            st.info(f"""
            **ขนาดไฟล์:** {file_size:.1f} KB  
            **ชนิดไฟล์:** {uploaded_image.type}
            """)
            
            if st.button("เริ่มการวิเคราะห์", type="primary", use_container_width=True):
                with st.spinner("🤖 AI กำลังวิเคราะห์..."):
                    start_time = time.time()
                    result = classify_image(uploaded_image)
                    processing_time = time.time() - start_time
                    
                    st.success(f"✅ เสร็จสิ้น! ({processing_time:.2f}s)")
                    
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    display_classification_result(result, show_confidence, show_probabilities)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add to history
                    st.session_state.analysis_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "result": result,
                        "processing_time": processing_time,
                        "mode": "single"
                    })

def dual_image_mode(show_confidence, show_probabilities):
    """โหมดสองด้าน"""
    
    st.markdown("### 📤 อัปโหลดรูปทั้งสองด้าน")
    
    col1, col2 = st.columns(2)
    
    # Front image
    with col1:
        st.markdown("#### 🖼️ ด้านหน้า")
        
        front_upload, front_camera = st.tabs(["📁 ไฟล์", "📸 กล้อง"])
        
        front_image = None
        
        with front_upload:
            front_image = st.file_uploader("เลือกรูปด้านหน้า", type=['jpg', 'jpeg', 'png'], key="front_upload")
        
        with front_camera:
            camera_front = st.camera_input("ถ่ายรูปด้านหน้า", key="front_camera")
            if camera_front:
                st.session_state.front_camera_image = camera_front
                st.success("✅ ถ่ายรูปสำเร็จ!")
        
        display_front = front_image or st.session_state.front_camera_image
        if display_front:
            st.image(display_front, caption="รูปด้านหน้า", use_column_width=True)
    
    # Back image
    with col2:
        st.markdown("#### 🖼️ ด้านหลัง")
        
        back_upload, back_camera = st.tabs(["📁 ไฟล์", "📸 กล้อง"])
        
        back_image = None
        
        with back_upload:
            back_image = st.file_uploader("เลือกรูปด้านหลัง", type=['jpg', 'jpeg', 'png'], key="back_upload")
        
        with back_camera:
            camera_back = st.camera_input("ถ่ายรูปด้านหลัง", key="back_camera")
            if camera_back:
                st.session_state.back_camera_image = camera_back
                st.success("✅ ถ่ายรูปสำเร็จ!")
        
        display_back = back_image or st.session_state.back_camera_image
        if display_back:
            st.image(display_back, caption="รูปด้านหลัง", use_column_width=True)
    
    st.markdown("---")
    
    final_front = front_image or st.session_state.front_camera_image
    final_back = back_image or st.session_state.back_camera_image
    
    if final_front and final_back:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.success("✅ มีรูปภาพทั้งสองด้านแล้ว!")
            
            if st.button("🚀 เริ่มการวิเคราะห์ทั้งสองด้าน", type="primary", use_container_width=True):
                with st.spinner("🤖 AI กำลังวิเคราะห์ทั้งสองด้าน..."):
                    start_time = time.time()
                    
                    front_result = classify_image(final_front)
                    back_result = classify_image(final_back)
                    
                    processing_time = time.time() - start_time
                    
                    st.success(f"✅ เสร็จสิ้น! ({processing_time:.2f}s)")
                    
                    st.markdown("### 📊 ผลการวิเคราะห์")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown("#### 🖼️ ด้านหน้า")
                        display_classification_result(front_result, show_confidence, show_probabilities)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown("#### 🖼️ ด้านหลัง")
                        display_classification_result(back_result, show_confidence, show_probabilities)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Comparison
                    if (front_result.get("status") == "success" and back_result.get("status") == "success"):
                        front_class = front_result.get("predicted_class", "")
                        back_class = back_result.get("predicted_class", "")
                        front_conf = front_result.get("confidence", 0)
                        back_conf = back_result.get("confidence", 0)
                        
                        st.markdown("### 🔍 การเปรียบเทียบ")
                        
                        if front_class == back_class:
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>🎯 ผลลัพธ์สอดคล้องกัน!</h4>
                                <p style="font-size: 1.1rem;"><strong>ทั้งสองด้านระบุเป็น:</strong> {front_class}</p>
                                <p><strong>ความเชื่อมั่นเฉลี่ย:</strong> {(front_conf + back_conf) / 2:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="warning-box">
                                <h4>⚠️ ผลลัพธ์ไม่สอดคล้องกัน</h4>
                                <p><strong>ด้านหน้า:</strong> {front_class} ({front_conf:.1%})</p>
                                <p><strong>ด้านหลัง:</strong> {back_class} ({back_conf:.1%})</p>
                                <p>⚡ แนะนำให้ปรึกษาผู้เชี่ยวชาญเพิ่มเติม</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
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
        <div class="info-box">
            <h3>📋 คำแนะนำ</h3>
            <p>กรุณาอัปโหลดรูปภาพทั้งสองด้าน (หน้า และ หลัง)</p>
            <p>การวิเคราะห์ทั้งสองด้านจะช่วยเพิ่มความแม่นยำ</p>
        </div>
        """, unsafe_allow_html=True)

def show_tips_section():
    """แสดงเคล็ดลับ"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 💡 เคล็ดลับสำหรับผลลัพธ์ที่ดีที่สุด")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="tips-card">
            <h3>📸 การถ่ายรูปที่ดี</h3>
            <ul>
                <li>ใช้แสงสว่างเพียงพอและสม่ำเสมอ</li>
                <li>รูปภาพชัดเจน ไม่เบลอ</li>
                <li>พื้นหลังสีเรียบ</li>
                <li>ถ่ายให้เห็นรายละเอียด</li>
                <li>หลีกเลี่ยงเงาบนพระเครื่อง</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tips-card">
            <h3>📊 การตีความผล</h3>
            <ul>
                <li><strong>>90%</strong>: เชื่อถือได้สูงมาก</li>
                <li><strong>70-90%</strong>: เชื่อถือได้ดี</li>
                <li><strong>50-70%</strong>: ควรตรวจสอบเพิ่ม</li>
                <li><strong><50%</strong>: ควรถ่ายใหม่</li>
                <li>ใช้ข้อมูลประกอบเท่านั้น</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="tips-card">
            <h3>🎯 ประเภทที่รองรับ</h3>
            <ul>
                <li>พระศิวลี</li>
                <li>พระสมเด็จ</li>
                <li>ปรกโพธิ์ 9 ใบ</li>
                <li>แหวกม่าน</li>
                <li>หลังรูปเหมือน</li>
                <li>วัดหนองอีดุก</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_history_section():
    """แสดงประวัติ"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 📈 ประวัติการวิเคราะห์")
    
    recent = list(reversed(st.session_state.analysis_history[-5:]))
    
    for i, analysis in enumerate(recent):
        with st.expander(f"📋 #{len(st.session_state.analysis_history)-i}: {analysis['timestamp']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if analysis.get("mode") == "dual":
                    st.write("**โหมด:** สองด้าน")
                    st.write(f"**ด้านหน้า:** {analysis['front_result'].get('predicted_class', 'N/A')}")
                    st.write(f"**ด้านหลัง:** {analysis['back_result'].get('predicted_class', 'N/A')}")
                else:
                    st.write("**โหมด:** รูปเดียว")
                    st.write(f"**ผลลัพธ์:** {analysis['result'].get('predicted_class', 'N/A')}")
            
            with col2:
                st.metric("เวลา", f"{analysis['processing_time']:.2f}s")

if __name__ == "__main__":
    main()
