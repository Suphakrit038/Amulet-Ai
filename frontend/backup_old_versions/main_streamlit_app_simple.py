#!/usr/bin/env python3
"""
Amulet-AI Frontend - Simple Version
ระบบ Frontend แบบเรียบง่ายสำหรับการจำแนกพระเครื่อง
"""

import streamlit as st
import os
import sys
import requests
import time
import numpy as np
from PIL import Image
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
API_BASE_URL = "http://localhost:8000"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Amulet-AI - ระบบวิเคราะห์พระเครื่องด้วย AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Prompt', sans-serif;
        background: linear-gradient(135deg, #f8f5f0 0%, #ffffff 50%, #faf7f2 100%);
        border: 3px solid #800000;
        border-radius: 20px;
        margin: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(128, 0, 0, 0.15);
        min-height: calc(100vh - 40px);
    }
    
    .main-header {
        background: linear-gradient(135deg, #800000 0%, #B8860B 100%);
        padding: 3rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(128, 0, 0, 0.3);
        min-height: 250px;
        display: flex;
        align-items: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #800000 0%, #B8860B 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(128, 0, 0, 0.4);
    }
    
    .upload-section {
        background: linear-gradient(135deg, rgba(128, 0, 0, 0.05) 0%, rgba(184, 134, 11, 0.05) 100%);
        padding: 2rem;
        border: 2px solid #800000;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f5f5f5;
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background: transparent;
        border-radius: 8px;
        color: #666;
        font-weight: 600;
        padding: 0 1.5rem;
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
        padding: 1.5rem;
    }
    
    /* Camera input styling */
    [data-testid="stCameraInput"] {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        background: white;
        transition: all 0.3s ease;
    }
    
    [data-testid="stCameraInput"]:hover {
        border-color: #800000;
        box-shadow: 0 4px 12px rgba(128, 0, 0, 0.1);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #ddd;
        border-radius: 10px;
        padding: 1.5rem;
        background: white;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #800000;
        background: #fafafa;
    }
    
    /* Logo hover effects */
    .main-header img {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.1);
        padding: 8px;
    }
    
    .main-header img:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.3);
    }
    
    /* Loading Spinner */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.9);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        backdrop-filter: blur(5px);
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 6px solid #f3f3f3;
        border-top: 6px solid #800000;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        box-shadow: 0 4px 20px rgba(128, 0, 0, 0.3);
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        margin-top: 20px;
        color: #800000;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
    }
    
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """ตรวจสอบสถานะ API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_image(uploaded_file):
    """ส่งรูปภาพไป API เพื่อทำนาย"""
    try:
        # เตรียมไฟล์สำหรับ API
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

def get_logo_base64():
    """Convert logo image to base64 for embedding"""
    try:
        logo_path = os.path.join(os.path.dirname(__file__), 'imgae', 'Amulet-AI_logo.png')
        if os.path.exists(logo_path):
            import base64
            with open(logo_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode()
        return ""
    except:
        return ""

def get_other_logos():
    """Get other partnership logos"""
    try:
        import base64
        logos = {}
        logo_dir = os.path.join(os.path.dirname(__file__), 'imgae')
        
        # Thai-Austrian Logo
        thai_logo_path = os.path.join(logo_dir, 'Logo Thai-Austrain.gif')
        if os.path.exists(thai_logo_path):
            with open(thai_logo_path, 'rb') as f:
                logos['thai_austrian'] = base64.b64encode(f.read()).decode()
        
        # DEPA Logo
        depa_logo_path = os.path.join(logo_dir, 'LogoDEPA-01.png')
        if os.path.exists(depa_logo_path):
            with open(depa_logo_path, 'rb') as f:
                logos['depa'] = base64.b64encode(f.read()).decode()
        
        return logos
    except:
        return {}

def main():
    # Initialize session state for camera images
    if 'front_camera_image' not in st.session_state:
        st.session_state.front_camera_image = None
    if 'back_camera_image' not in st.session_state:
        st.session_state.back_camera_image = None
    if 'show_front_camera' not in st.session_state:
        st.session_state.show_front_camera = False
    if 'show_back_camera' not in st.session_state:
        st.session_state.show_back_camera = False
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False
    if 'loading_message' not in st.session_state:
        st.session_state.loading_message = ""
    
    # Show loading overlay if needed
    if st.session_state.is_loading:
        st.markdown(f"""
        <div class="loading-overlay">
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">{st.session_state.loading_message}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Auto-hide loading after a short time
        time.sleep(1)
        st.session_state.is_loading = False
        st.rerun()
    
    # Get logos
    amulet_logo = get_logo_base64()
    other_logos = get_other_logos()
    
    # Build header with logos
    logos_html = ""
    
    # Add Amulet logo on the left (ขยายขนาดเป็น 200px)
    if amulet_logo:
        logos_html += f'<img src="data:image/png;base64,{amulet_logo}" style="height: 200px; width: auto; margin-right: 3rem;" alt="Amulet-AI Logo">'
    
    # Add partnership logos on the right (ขยายขนาดเป็น 200px)
    partner_logos = ""
    if 'thai_austrian' in other_logos:
        partner_logos += f'<img src="data:image/gif;base64,{other_logos["thai_austrian"]}" style="height: 200px; width: auto; margin: 0 15px;" alt="Thai-Austrian Logo">'
    
    if 'depa' in other_logos:
        partner_logos += f'<img src="data:image/png;base64,{other_logos["depa"]}" style="height: 200px; width: auto; margin: 0 15px;" alt="DEPA Logo">'
    
    # Enhanced Header with logos (ย้ายโลโก้ไปด้านขวาสุด)
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
            <div style="display: flex; align-items: center;">
                {logos_html}
                <div>
                    <h1 style="font-size: 3rem; margin: 0;">AMULET-AI</h1>
                    <h2 style="font-size: 1.5rem; margin: 0.5rem 0 0 0; opacity: 0.9;">
                        ระบบวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์
                    </h2>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 2rem; margin-left: auto;">
                {partner_logos}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # API Status
    api_healthy = check_api_health()
    
    if api_healthy:
        st.success("🟢 API เชื่อมต่อสำเร็จ - ระบบพร้อมใช้งาน")
    else:
        st.error("🔴 API ไม่พร้อมใช้งาน - กำลังใช้โหมด Demo")
    
    st.markdown("---")
    
    # File Upload Section with Camera
    st.markdown("## อัปโหลดรูปพระเครื่อง")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### รูปด้านหน้า")
        
        # Tabs for upload methods
        tab1, tab2 = st.tabs(["📁 อัปโหลดไฟล์", "📸 ถ่ายรูป"])
        
        front_image = None
        
        with tab1:
            front_image = st.file_uploader(
                "เลือกรูปด้านหน้า",
                type=['png', 'jpg', 'jpeg'],
                key="front_file"
            )
        
        with tab2:
            # ปุ่มเปิดกล้อง
            if st.button("📸 เปิดกล้องถ่ายรูป", key="open_front_camera", use_container_width=True):
                # Show loading
                st.session_state.is_loading = True
                st.session_state.loading_message = "📷 กำลังเปิดกล้อง..."
                st.session_state.show_front_camera = True
                st.rerun()
            
            if st.session_state.show_front_camera:
                st.info("📷 กรุณาอนุญาตให้เข้าถึงกล้องแล้วกด 'Take Photo'")
                front_camera = st.camera_input(
                    "ถ่ายรูปด้านหน้า",
                    key="front_camera"
                )
                
                if front_camera is not None:
                    # Show loading for saving image
                    st.session_state.is_loading = True
                    st.session_state.loading_message = "💾 กำลังบันทึกรูป..."
                    st.session_state.front_camera_image = front_camera
                    st.session_state.show_front_camera = False
                    st.success("✅ ถ่ายรูปด้านหน้าสำเร็จ!")
                    st.rerun()
                
                # ปุ่มยกเลิก
                if st.button("❌ ยกเลิก", key="cancel_front_camera"):
                    st.session_state.show_front_camera = False
                    st.rerun()
            else:
                st.info("📸 กดปุ่มด้านบนเพื่อเปิดกล้อง")
        
        # แสดงพรีวิวรูป (รวมทั้งจากไฟล์อัปโหลดและกล้อง)
        display_front_image = front_image or st.session_state.front_camera_image
        if display_front_image:
            st.image(display_front_image, caption="รูปด้านหน้า", use_container_width=True)
            
            # ปุ่มลบรูป
            if st.session_state.front_camera_image and st.button("🗑️ ลบรูปด้านหน้า", key="clear_front_image"):
                st.session_state.is_loading = True
                st.session_state.loading_message = "🗑️ กำลังลบรูป..."
                st.session_state.front_camera_image = None
                st.rerun()
    
    with col2:
        st.markdown("### รูปด้านหลัง")
        
        # Tabs for upload methods
        tab1, tab2 = st.tabs(["📁 อัปโหลดไฟล์", "📸 ถ่ายรูป"])
        
        back_image = None
        
        with tab1:
            back_image = st.file_uploader(
                "เลือกรูปด้านหลัง",
                type=['png', 'jpg', 'jpeg'],
                key="back_file"
            )
        
        with tab2:
            # ปุ่มเปิดกล้อง
            if st.button("📸 เปิดกล้องถ่ายรูป", key="open_back_camera", use_container_width=True):
                # Show loading
                st.session_state.is_loading = True
                st.session_state.loading_message = "📷 กำลังเปิดกล้อง..."
                st.session_state.show_back_camera = True
                st.rerun()
            
            if st.session_state.show_back_camera:
                st.info("📷 กรุณาอนุญาตให้เข้าถึงกล้องแล้วกด 'Take Photo'")
                back_camera = st.camera_input(
                    "ถ่ายรูปด้านหลัง",
                    key="back_camera"
                )
                
                if back_camera is not None:
                    # Show loading for saving image
                    st.session_state.is_loading = True
                    st.session_state.loading_message = "💾 กำลังบันทึกรูป..."
                    st.session_state.back_camera_image = back_camera
                    st.session_state.show_back_camera = False
                    st.success("✅ ถ่ายรูปด้านหลังสำเร็จ!")
                    st.rerun()
                
                # ปุ่มยกเลิก
                if st.button("❌ ยกเลิก", key="cancel_back_camera"):
                    st.session_state.show_back_camera = False
                    st.rerun()
            else:
                st.info("📸 กดปุ่มด้านบนเพื่อเปิดกล้อง")
        
        # แสดงพรีวิวรูป (รวมทั้งจากไฟล์อัปโหลดและกล้อง)
        display_back_image = back_image or st.session_state.back_camera_image
        if display_back_image:
            st.image(display_back_image, caption="รูปด้านหลัง", use_container_width=True)
            
            # ปุ่มลบรูป
            if st.session_state.back_camera_image and st.button("🗑️ ลบรูปด้านหลัง", key="clear_back_image"):
                st.session_state.is_loading = True
                st.session_state.loading_message = "🗑️ กำลังลบรูป..."
                st.session_state.back_camera_image = None
                st.rerun()
    
    # Analysis Section
    # ตรวจสอบรูปทั้งจากไฟล์อัปโหลดและกล้อง
    final_front_image = front_image or st.session_state.front_camera_image
    final_back_image = back_image or st.session_state.back_camera_image
    
    if final_front_image and final_back_image:
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_button = st.button(
                "เริ่มการวิเคราะห์ด้วย AI", 
                type="primary", 
                use_container_width=True
            )
        
        if analyze_button:
            st.markdown("## ผลการวิเคราะห์")
            
            with st.spinner("AI กำลังวิเคราะห์รูปภาพ... กรุณารอสักครู่"):
                # Try real API first, then fallback to mock
                if api_healthy:
                    result = predict_image(final_front_image)
                else:
                    # Mock result for demo
                    time.sleep(2)
                    thai_names = ['พระสมเด็จวัดระฆัง', 'พระนางพญา', 'พระพิมพ์เล็ก', 'พระพิมพ์พุทธคุณ', 'พระไอย์ไข่']
                    result = {
                        'thai_name': np.random.choice(thai_names),
                        'confidence': np.random.uniform(0.75, 0.95),
                        'predicted_class': f'class_{np.random.randint(1, 6)}',
                        'analysis_type': 'dual_image',
                        'processing_time': np.random.uniform(1.2, 2.5)
                    }
                
                # Display results
                if 'error' not in result:
                    st.success("✅ การวิเคราะห์เสร็จสิ้น!")
                    
                    # Get data from new API format
                    if 'prediction' in result:
                        predicted_class = result['prediction']['class']
                        confidence = result['prediction']['confidence']
                        probabilities = result['prediction']['probabilities']
                        model_info = result.get('model_info', {})
                    else:
                        # Fallback for old format
                        predicted_class = result.get('thai_name', 'ไม่ระบุ')
                        confidence = result.get('confidence', 0)
                        probabilities = {}
                        model_info = {}
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "ประเภทพระเครื่อง",
                            predicted_class
                        )
                    
                    with col2:
                        confidence_percent = confidence * 100 if confidence <= 1 else confidence
                        st.metric(
                            "ความเชื่อมั่น",
                            f"{confidence_percent:.1f}%"
                        )
                    
                    with col3:
                        processing_time = result.get('processing_time', 0.1)
                        st.metric(
                            "เวลาประมวลผล",
                            f"{processing_time:.2f} วินาที"
                        )
                    
                    # Confidence interpretation
                    if confidence >= 0.9:
                        st.success("🎯 ความเชื่อมั่นสูงมาก - ผลลัพธ์น่าเชื่อถือ")
                    elif confidence >= 0.7:
                        st.warning("⚠️ ความเชื่อมั่นปานกลาง - ควรตรวจสอบเพิ่มเติม")
                    else:
                        st.error("⚡ ความเชื่อมั่นต่ำ - ควรถ่ายรูปใหม่หรือปรึกษาผู้เชี่ยวชาญ")
                    
                    # Show probabilities if available
                    if probabilities:
                        st.markdown("### 📊 ความน่าจะเป็นของแต่ละประเภท")
                        
                        # Sort probabilities
                        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                        
                        for class_name, prob in sorted_probs[:5]:  # Show top 5
                            prob_percent = prob * 100 if prob <= 1 else prob
                            st.progress(prob, text=f"{class_name}: {prob_percent:.1f}%")
                    
                    # Model info
                    if model_info:
                        st.markdown("### 🤖 ข้อมูลโมเดล")
                        with st.expander("รายละเอียดโมเดล AI"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**เวอร์ชัน**: {model_info.get('version', 'N/A')}")
                                st.write(f"**สถาปัตยกรรม**: {model_info.get('architecture', 'N/A')}")
                            with col2:
                                st.write(f"**ความแม่นยำ**: {model_info.get('accuracy', 'N/A')}")
                                st.write(f"**จำนวนคลาส**: {model_info.get('total_classes', 'N/A')}")
                        
                else:
                    st.error(f"❌ เกิดข้อผิดพลาด: {result['error']}")
    
    else:
        st.markdown("""
        <div class="upload-section">
            <h3 style="color: #800000; margin-bottom: 1rem;">พร้อมเริ่มการวิเคราะห์แล้วหรือยัง?</h3>
            <p style="color: #666; font-size: 1.1rem;">
                กรุณาอัปโหลดรูปภาพทั้งสองด้าน (หน้า และ หลัง) เพื่อเริ่มการวิเคราะห์ด้วย AI
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tips Section
    st.markdown("---")
    st.markdown("## เคล็ดลับการใช้งาน")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### การถ่ายรูปที่ดี
        - ใช้แสงสว่างเพียงพอ
        - รูปภาพชัดเจน ไม่เบลอ
        - พื้นหลังสีเรียบ
        - ขนาดไฟล์ไม่เกิน 10MB
        """)
    
    with col2:
        st.markdown("""
        ### การตีความผลลัพธ์
        - >90%: ความเชื่อมั่นสูงมาก
        - 70-90%: ความเชื่อมั่นปานกลาง
        - <70%: ความเชื่อมั่นต่ำ
        - เวลาประมวลผล: 2-5 วินาที
        """)
    
    # Footer และ All Text Content Section
    st.markdown("---")
    
    # Section แสดงข้อความทั้งหมดในระบบ
    with st.expander("📝 ข้อความทั้งหมดในระบบ", expanded=False):
        st.markdown("## ข้อความและข้อมูลทั้งหมดในระบบ")
        
        st.markdown("### 1. ข้อความในส่วน Header")
        st.text("AMULET-AI")
        st.text("ระบบวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์")
        
        st.markdown("### 2. ข้อความสถานะ API")
        st.text("🟢 API เชื่อมต่อสำเร็จ - ระบบพร้อมใช้งาน")
        st.text("🔴 API ไม่พร้อมใช้งาน - กำลังใช้โหมด Demo")
        
        st.markdown("### 3. ข้อความในส่วนอัปโหลด")
        st.text("อัปโหลดรูปพระเครื่อง")
        st.text("รูปด้านหน้า")
        st.text("รูปด้านหลัง")
        st.text("📁 อัปโหลดไฟล์")
        st.text("📸 ถ่ายรูป")
        st.text("เลือกรูปด้านหน้า")
        st.text("เลือกรูปด้านหลัง")
        st.text("📸 เปิดกล้องถ่ายรูป")
        st.text("📷 กรุณาอนุญาตให้เข้าถึงกล้องในเบราว์เซอร์ของคุณ")
        st.text("ถ่ายรูปด้านหน้า")
        st.text("ถ่ายรูปด้านหลัง")
        st.text("✅ ถ่ายรูปด้านหน้าสำเร็จ!")
        st.text("✅ ถ่ายรูปด้านหลังสำเร็จ!")
        st.text("📸 กดปุ่มด้านบนเพื่อเปิดกล้อง")
        
        st.markdown("### 4. ข้อความในส่วนการวิเคราะห์")
        st.text("เริ่มการวิเคราะห์ด้วย AI")
        st.text("ผลการวิเคราะห์")
        st.text("AI กำลังวิเคราะห์รูปภาพ... กรุณารอสักครู่")
        st.text("✅ การวิเคราะห์เสร็จสิ้น!")
        st.text("ประเภทพระเครื่อง")
        st.text("ความเชื่อมั่น")
        st.text("เวลาประมวลผล")
        st.text("พระสมเด็จวัดระฆัง")
        st.text("พระนางพญา")
        st.text("พระพิมพ์เล็ก")
        st.text("พระพิมพ์พุทธคุณ")
        st.text("พระไอย์ไข่")
        st.text("ไม่ระบุ")
        st.text("วินาที")
        st.text("🎯 ความเชื่อมั่นสูงมาก - ผลลัพธ์น่าเชื่อถือ")
        st.text("⚠️ ความเชื่อมั่นปานกลาง - ควรตรวจสอบเพิ่มเติม")
        st.text("⚡ ความเชื่อมั่นต่ำ - ควรถ่ายรูปใหม่หรือปรึกษาผู้เชี่ยวชาญ")
        st.text("❌ เกิดข้อผิดพลาด:")
        
        st.markdown("### 5. ข้อความในส่วนคำแนะนำ")
        st.text("พร้อมเริ่มการวิเคราะห์แล้วหรือยัง?")
        st.text("กรุณาอัปโหลดรูปภาพทั้งสองด้าน (หน้า และ หลัง) เพื่อเริ่มการวิเคราะห์ด้วย AI")
        st.text("เคล็ดลับการใช้งาน")
        st.text("การถ่ายรูปที่ดี")
        st.text("- ใช้แสงสว่างเพียงพอ")
        st.text("- รูปภาพชัดเจน ไม่เบลอ")
        st.text("- พื้นหลังสีเรียบ")
        st.text("- ขนาดไฟล์ไม่เกิน 10MB")
        st.text("การตีความผลลัพธ์")
        st.text("- >90%: ความเชื่อมั่นสูงมาก")
        st.text("- 70-90%: ความเชื่อมั่นปานกลาง")
        st.text("- <70%: ความเชื่อมั่นต่ำ")
        st.text("- เวลาประมวลผล: 2-5 วินาที")
        
        st.markdown("### 6. ข้อความในส่วน Footer")
        st.text("Amulet-AI Project")
        st.text("© 2025 Amulet-AI | Powered by Advanced AI Technology")
        
        st.markdown("### 7. ข้อความในส่วน Configuration")
        st.text("API_BASE_URL = http://localhost:8000")
        st.text("MAX_FILE_SIZE = 10MB")
        st.text("Amulet-AI - ระบบวิเคราะห์พระเครื่องด้วย AI")
        
        st.markdown("### 8. ข้อความข้อผิดพลาด")
        st.text("API Error:")
        st.text("Connection Error:")
        st.text("Import error:")
        
        st.markdown("### 9. ข้อความในการประมวลผล")
        st.text("enhance = true")
        st.text("analysis_type = dual")
        st.text("analysis_type = single")
        st.text("front.jpg")
        st.text("back.jpg")
        st.text("image/jpeg")
        st.text("class_1")
        st.text("class_2")
        st.text("class_3")
        st.text("class_4")
        st.text("class_5")
        st.text("class_6")
        st.text("dual_image")
        
        st.markdown("### 10. ข้อความ Metadata")
        st.text("Convert logo image to base64 for embedding")
        st.text("Get other partnership logos")
        st.text("ตรวจสอบสถานะ API")
        st.text("ส่งรูปภาพไป API เพื่อทำนาย")
        st.text("Amulet-AI_logo.png")
        st.text("Logo Thai-Austrain.gif")
        st.text("LogoDEPA-01.png")
        st.text("Thai-Austrian Logo")
        st.text("DEPA Logo")
        st.text("Amulet-AI Logo")
        
        st.markdown("### 11. CSS Classes และ Styling")
        st.text("stApp")
        st.text("main-header")
        st.text("stButton")
        st.text("upload-section")
        st.text("stTabs")
        st.text("stCameraInput")
        st.text("stFileUploader")
        st.text("font-family: Prompt")
        st.text("background: linear-gradient")
        st.text("border-radius: 10px")
        st.text("color: #800000")
        st.text("color: #B8860B")
        st.text("box-shadow")
        st.text("transition: all 0.3s ease")
        st.text("transform: translateY(-2px)")
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f5f5f5; border-radius: 10px;">
        <h3 style="margin: 0; color: #800000;">Amulet-AI Project</h3>
        <p style="margin: 0.5rem 0 0 0; color: #666;">
            © 2025 Amulet-AI | Powered by Advanced AI Technology
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()