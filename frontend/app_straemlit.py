import streamlit as st
import requests
from datetime import datetime
from PIL import Image
import io

# Import functions from utils file (inline import approach)
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from frontend.utils import validate_and_convert_image, send_predict_request, SUPPORTED_FORMATS, FORMAT_DISPLAY
except ImportError:
    # Fallback: define functions locally if import fails
    # รองรับ HEIC format
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "heic", "heif", "webp", "bmp", "tiff"]
        FORMAT_DISPLAY = "JPG, JPEG, PNG, HEIC, HEIF, WebP, BMP, TIFF"
    except ImportError:
        SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
        FORMAT_DISPLAY = "JPG, JPEG, PNG, WebP, BMP, TIFF"

    MAX_FILE_SIZE_MB = 10
    
    def validate_and_convert_image(uploaded_file):
        """Validate uploaded image, enforce size and extension limits, convert to RGB JPEG bytes."""
        try:
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

            if hasattr(uploaded_file, 'read'):
                file_bytes = uploaded_file.read()
            else:
                file_bytes = getattr(uploaded_file, 'getvalue', lambda: b'')()

            if not file_bytes:
                return False, None, None, 'Empty file or unreadable upload'

            if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
                return False, None, None, f'File too large (> {MAX_FILE_SIZE_MB} MB)'

            filename = getattr(uploaded_file, 'name', '') or ''
            if filename:
                ext = filename.rsplit('.', 1)[-1].lower()
                if ext not in SUPPORTED_FORMATS:
                    return False, None, None, f'Unsupported file extension: .{ext}'

            img = Image.open(io.BytesIO(file_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)

            return True, img, img_byte_arr, None
        except Exception as e:
            return False, None, None, str(e)
    
    def send_predict_request(files, api_url, timeout=60):
        """Send POST to /predict with given files dict."""
        url = api_url.rstrip('/') + '/predict'
        prepared = {}
        for k, v in files.items():
            fname, fileobj, mime = v
            try:
                fileobj.seek(0)
            except Exception:
                pass
            prepared[k] = (fname, fileobj, mime)
        
        resp = requests.post(url, files=prepared, timeout=timeout)
        return resp

# เปลี่ยนจาก st.secrets.get() เป็นการกำหนดค่าตรงๆ
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    /* Enhanced animated background with multiple layers */
    .main {
        padding-top: 2rem;
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 226, 0.3) 0%, transparent 50%),
            linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
        background-size: 800% 800%, 600% 600%, 700% 700%, 400% 400%;
        animation: gradientShift 20s ease infinite, floatingBubbles 25s ease-in-out infinite;
        min-height: 100vh;
        position: relative;
        overflow: hidden;
    }
    
    /* Enhanced gradient animation */
    @keyframes gradientShift {
        0% {
            background-position: 0% 50%, 100% 0%, 50% 100%, 0% 50%;
        }
        25% {
            background-position: 100% 50%, 0% 100%, 0% 0%, 100% 50%;
        }
        50% {
            background-position: 100% 0%, 0% 0%, 100% 50%, 50% 100%;
        }
        75% {
            background-position: 0% 100%, 100% 50%, 50% 0%, 0% 0%;
        }
        100% {
            background-position: 0% 50%, 100% 0%, 50% 100%, 0% 50%;
        }
    }
    
    @keyframes floatingBubbles {
        0%, 100% {
            transform: translateY(0) scale(1);
        }
        33% {
            transform: translateY(-10px) scale(1.05);
        }
        66% {
            transform: translateY(5px) scale(0.95);
        }
    }
    
    /* Expand content width for wide layout */
    .block-container {
        max-width: 95% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Full width sections */
    .main-content-area {
        max-width: none !important;
        width: 100% !important;
    }
    
    /* Sidebar and main area adjustments */
    section[data-testid="stSidebar"] {
        width: 350px !important;
        min-width: 350px !important;
    }
    
    .main .block-container {
        max-width: calc(100% - 350px) !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }
    
    /* Column spacing adjustments */
    div[data-testid="column"] {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Background animation */
    @keyframes gradientShift {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    /* Enhanced floating particles animation */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 15% 25%, rgba(255, 255, 255, 0.15) 0%, transparent 3%),
            radial-gradient(circle at 85% 75%, rgba(255, 255, 255, 0.12) 0%, transparent 2.5%),
            radial-gradient(circle at 25% 80%, rgba(255, 255, 255, 0.18) 0%, transparent 2%),
            radial-gradient(circle at 75% 15%, rgba(255, 255, 255, 0.14) 0%, transparent 3.5%),
            radial-gradient(circle at 45% 45%, rgba(255, 255, 255, 0.16) 0%, transparent 2%),
            radial-gradient(circle at 90% 35%, rgba(255, 255, 255, 0.13) 0%, transparent 2.8%),
            radial-gradient(circle at 10% 65%, rgba(255, 255, 255, 0.17) 0%, transparent 2.3%),
            radial-gradient(circle at 60% 90%, rgba(255, 255, 255, 0.11) 0%, transparent 3.2%);
        background-size: 180px 180px, 220px 220px, 160px 160px, 250px 250px, 190px 190px, 200px 200px, 170px 170px, 240px 240px;
        animation: floatingParticles 35s ease-in-out infinite, particleDrift 45s linear infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes floatingParticles {
        0%, 100% {
            transform: translateY(0px) rotate(0deg);
        }
        25% {
            transform: translateY(-25px) rotate(90deg);
        }
        50% {
            transform: translateY(-10px) rotate(180deg);
        }
        75% {
            transform: translateY(15px) rotate(270deg);
        }
    }
    
    @keyframes particleDrift {
        0% {
            background-position: 0% 0%, 100% 100%, 0% 100%, 100% 0%, 50% 50%, 25% 75%, 75% 25%, 50% 0%;
        }
        100% {
            background-position: 100% 100%, 0% 0%, 100% 0%, 0% 100%, 0% 0%, 75% 25%, 25% 75%, 0% 50%;
        }
    }
    
    /* Additional depth and atmosphere layers */
    .main::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            linear-gradient(45deg, rgba(102, 126, 234, 0.03) 0%, transparent 70%),
            linear-gradient(-45deg, rgba(118, 75, 162, 0.03) 0%, transparent 70%),
            linear-gradient(90deg, rgba(240, 147, 251, 0.02) 0%, transparent 50%);
        animation: atmosphericFlow 50s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes atmosphericFlow {
        0%, 100% {
            opacity: 0.3;
            transform: scale(1) rotate(0deg);
        }
        50% {
            opacity: 0.6;
            transform: scale(1.05) rotate(180deg);
        }
    }
    
    /* Custom header styling with enhanced animation */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        animation: headerGlow 3s ease-in-out infinite alternate;
        position: relative;
        overflow: hidden;
    }
    
    .custom-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 4s infinite;
        pointer-events: none;
    }
    
    @keyframes headerGlow {
        0% {
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }
        100% {
            box-shadow: 0 8px 30px rgba(102, 126, 234, 0.5);
        }
    }
    
    @keyframes shimmer {
        0% {
            transform: translateX(-100%) translateY(-100%) rotate(45deg);
        }
        100% {
            transform: translateX(100%) translateY(100%) rotate(45deg);
        }
    }
    
    .custom-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: titleBounce 2s ease-in-out infinite;
    }
    
    @keyframes titleBounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-10px);
        }
        60% {
            transform: translateY(-5px);
        }
    }
    
    .custom-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        animation: fadeInUp 1.5s ease-out 0.5s both;
    }
    
    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translateY(30px);
        }
        100% {
            opacity: 0.9;
            transform: translateY(0);
        }
    }
    
    /* Upload sections with hover animations */
    .upload-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px dashed #e1e8ed;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        backdrop-filter: blur(10px);
        animation: slideInFromLeft 0.8s ease-out;
        width: 100% !important;
        max-width: none !important;
    }
    
    @keyframes slideInFromLeft {
        0% {
            opacity: 0;
            transform: translateX(-50px);
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .upload-section:hover {
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        transform: translateY(-5px) scale(1.02);
        background: rgba(255, 255, 255, 1);
    }
    
    /* Result cards with staggered animation */
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        backdrop-filter: blur(10px);
        animation: slideInFromRight 0.8s ease-out;
        transition: all 0.3s ease;
        width: 100% !important;
        max-width: none !important;
    }
    
    @keyframes slideInFromRight {
        0% {
            opacity: 0;
            transform: translateX(50px);
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 30px rgba(0,0,0,0.15);
    }
    
    .confidence-high {
        border-left-color: #4CAF50;
    }
    
    .confidence-medium {
        border-left-color: #FF9800;
    }
    
    .confidence-low {
        border-left-color: #f44336;
    }
    
    /* Metrics styling with enhanced animations */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: popIn 0.6s ease-out;
        animation-delay: var(--delay, 0s);
    }
    
    @keyframes popIn {
        0% {
            opacity: 0;
            transform: scale(0.3) rotate(-10deg);
        }
        100% {
            opacity: 1;
            transform: scale(1) rotate(0deg);
        }
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.1);
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #fff 0%, #e8f4fd 100%);
    }
    
    /* Tips section with wave animation */
    .tips-container {
        background: rgba(248, 249, 250, 0.9);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        animation: waveFloat 6s ease-in-out infinite;
        position: relative;
        overflow: hidden;
        width: 100% !important;
        max-width: none !important;
    }
    
    @keyframes waveFloat {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    .tips-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: wave 3s infinite;
    }
    
    @keyframes wave {
        0% {
            left: -100%;
        }
        100% {
            left: 100%;
        }
    }
    
    .tip-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-top: 3px solid #667eea;
        backdrop-filter: blur(5px);
        animation: slideInScale 0.8s ease-out;
        animation-fill-mode: both;
        transition: all 0.3s ease;
        width: 100% !important;
        max-width: none !important;
    }
    
    @keyframes slideInScale {
        0% {
            opacity: 0;
            transform: translateY(30px) scale(0.9);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    .tip-card:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    /* Button styling with enhanced animations */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
        animation: buttonPulse 2s infinite;
    }
    
    @keyframes buttonPulse {
        0% {
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        50% {
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }
        100% {
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar styling with animated gradient */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(248, 249, 250, 0.95) 0%, rgba(233, 236, 239, 0.95) 100%);
        backdrop-filter: blur(10px);
        animation: sidebarGlow 8s ease-in-out infinite;
    }
    
    @keyframes sidebarGlow {
        0%, 100% {
            background: linear-gradient(180deg, rgba(248, 249, 250, 0.95) 0%, rgba(233, 236, 239, 0.95) 100%);
        }
        50% {
            background: linear-gradient(180deg, rgba(255, 248, 250, 0.95) 0%, rgba(240, 233, 255, 0.95) 100%);
        }
    }
    
    /* Loading and progress animations */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .analyzing {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes bounce {
        0%, 20%, 53%, 80%, 100% {
            animation-timing-function: cubic-bezier(0.215, 0.610, 0.355, 1.000);
            transform: translate3d(0,0,0);
        }
        40%, 43% {
            animation-timing-function: cubic-bezier(0.755, 0.050, 0.855, 0.060);
            transform: translate3d(0, -10px, 0);
        }
        70% {
            animation-timing-function: cubic-bezier(0.755, 0.050, 0.855, 0.060);
            transform: translate3d(0, -5px, 0);
        }
        90% {
            transform: translate3d(0,-2px,0);
        }
    }
    
    /* Responsive animations for mobile */
    @media (max-width: 768px) {
        .main {
            animation-duration: 20s;
        }
        
        .custom-header {
            animation-duration: 4s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px) scale(1.03);
        }
    }
    
    /* Smooth scroll behavior */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom scrollbar with animation */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(241, 241, 241, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #764ba2, #667eea);
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# ใช้ฟังก์ชันจาก utils แทน

st.set_page_config(
    page_title="Amulet-AI", 
    page_icon=None, 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Section with custom styling
st.markdown("""
<div class="custom-header">
    <h1>Amulet-AI</h1>
    <p>วิเคราะห์พระเครื่องลึกลับด้วยปัญญาประดิษฐ์</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0.5rem; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">📚 คู่มือใช้งาน</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            แนวทางการใช้งานระบบ AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🎯 วิธีการใช้งาน", expanded=True):
        st.markdown("""
        ### ขั้นตอนการใช้งาน
        
        **1. เตรียมภาพ** 📸
        - ภาพด้านหน้า (บังคับ)
        - ภาพด้านหลัง (บังคับ)
        
        **2. อัปโหลดภาพ** 📤
        - คลิกปุ่ม "อัปโหลด" หรือ
        - ใช้กล้องถ่ายภาพใหม่
        
        **3. ตรวจสอบภาพ** ✅
        - ดูตัวอย่างภาพ
        - ตรวจสอบความชัด
        
        **4. วิเคราะห์** 🔍
        - กดปุ่ม "วิเคราะห์ตอนนี้"
        - รอระบบประมวลผล (30-60 วินาที)
        
        **5. ผลลัพธ์** 📊
        - ดูการจำแนกประเภท
        - ประเมินราคา
        - ข้อแนะนำการขาย
        """)
    
    with st.expander("⚡ ข้อมูลระบบ"):
        st.markdown("""
        ### เทคโนโลยี
        
        **🧠 AI Model**
        - TensorFlow 2.x
        - Custom CNN Architecture
        - Transfer Learning
        
        **🔧 Backend**
        - FastAPI Framework
        - Python 3.9+
        - REST API
        
        **🎨 Frontend**
        - Streamlit
        - Responsive Design
        - Real-time Processing
        
        ### ฟีเจอร์หลัก
        
        ✅ **การจำแนกประเภท** - Top-3 Results  
        ✅ **ประเมินราคา** - ช่วงราคาตลาด  
        ✅ **คำแนะนำ** - ช่องทางขายที่เหมาะสม  
        ✅ **ข้อมูลประวัติ** - ความเป็นมา  
        
        ### สถิติระบบ
        
        📈 **ความแม่นยำ**: ~85%  
        ⚡ **เวลาประมวลผล**: 30-60 วินาที  
        🗄️ **ฐานข้อมูล**: 5,000+ รูปแบบ  
        """)
    
    with st.expander("📷 เคล็ดลับถ่ายรูป"):
        st.markdown("""
        ### 💡 แสงสว่าง
        
        **✅ ควรทำ:**
        - ใช้แสงธรรมชาติ (แสงแดดอ่อนๆ)
        - แสงขาวเย็น (LED)
        - แสงกระจายสม่ำเสมอ
        
        **❌ หลีกเลี่ยง:**
        - แสงแฟลช (สร้างเงาแข็ง)
        - แสงเหลืองอุ่น
        - แสงจากด้านเดียว
        
        ### 📐 มุมกล้อง
        
        **✅ มุมที่ดี:**
        - ถ่ายตั้งฉาก 90° กับพระ
        - ระยะ 20-30 ซม.
        - วางพระกึ่งกลางเฟรม
        
        **❌ หลีกเลี่ยง:**
        - มุมเอียง หรือ มุมขวาง
        - ถ่ายใกล้หรือไกลเกินไป
        - พระเอียงในภาพ
        
        ### 🎨 พื้นหลัง
        
        **✅ พื้นหลังดี:**
        - สีขาว หรือ ครีม
        - ผิวเรียบไม่มีลาย
        - ไม่มีเงาหรือสิ่งรบกวน
        
        **❌ หลีกเลี่ยง:**
        - พื้นหลังมีลาย/สี
        - วัสดุสะท้อนแสง
        - มีวัตถุอื่นในภาพ
        
        ### 🔍 คุณภาพภาพ
        
        **📱 การตั้งค่า:**
        - ความละเอียด: ≥ 1080p
        - ไม่เปิด Digital Zoom
        - ไม่ใช้ฟิลเตอร์
        
        **⚡ ขณะถ่าย:**
        - ถือมือให้นิ่ง
        - กดชัตเตอร์เบาๆ
        - ตรวจสอบภาพก่อนอัปโหลด
        """)
    
    with st.expander("⚠️ ข้อจำกัดและข้อควรรู้"):
        st.markdown("""
        ### 🚧 ข้อจำกัดปัจจุบัน
        
        **📊 ความแม่นยำ**
        - ระบบยังอยู่ในช่วงพัฒนา
        - ความแม่นยำประมาณ 80-85%
        - อาจมีผลลัพธ์ผิดพลาดได้
        
        **🗃️ ข้อมูลเทรนนิ่ง**
        - ใช้ข้อมูลจำลองบางส่วน
        - ครอบคลุมพระยอดนิยม
        - อัพเดตฐานข้อมูลต่อเนื่อง
        
        **💰 ราคาประเมิน**
        - เป็นการประมาณการเท่านั้น
        - อ้างอิงจากข้อมูลตลาด
        - ไม่ใช่ราคาแน่นอน
        
        ### 📋 ข้อแนะนำ
        
        **🎯 การใช้งาน**
        - ใช้เป็นข้อมูลเบื้องต้น
        - สอบถามผู้เชี่ยวชาญเพิ่มเติม
        - เปรียบเทียบกับแหล่งอื่น
        
        **🔒 ความปลอดภัย**
        - ภาพจะไม่ถูกเก็บในระบบ
        - ประมวลผลแล้วลบทิ้ง
        - ไม่มีการเก็บข้อมูลส่วนตัว
        """)
        
        # เพิ่ม warning box
        st.error("""
        **⚠️ ข้อควรระวัง**
        
        ระบบนี้เป็น Demo เพื่อการทดสอบเท่านั้น  
        ไม่ควรใช้เป็นเกณฑ์หลักในการซื้อขาย
        """)
    
    # เพิ่มส่วน Stats
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; 
                background: rgba(102, 126, 234, 0.1); 
                border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: #667eea; margin: 0;">📊 สถิติการใช้งาน</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔍 วิเคราะห์วันนี้", "127", "↗️ +23")
    with col2:
        st.metric("🎯 ความแม่นยำ", "84.5%", "↗️ +2.1%")
    
    # เพิ่มลิงก์และข้อมูลติดต่อ
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h4 style="color: #667eea; margin-bottom: 0.5rem;">📞 ช่วยเหลือ</h4>
        <p style="color: #6c757d; font-size: 0.85rem; margin: 0;">
            หากมีปัญหาการใช้งาน<br>
            สามารถติดต่อสอบถามได้
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main Content
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <h2 style="color: #667eea; margin-bottom: 0.5rem;">อัปโหลดรูปภาพพระเครื่อง</h2>
    <p style="color: #6c757d; margin: 0;">รองรับไฟล์: <code>{}</code></p>
</div>
""".format(FORMAT_DISPLAY), unsafe_allow_html=True)

# Image input options
col_upload, col_camera = st.columns(2)

with col_upload:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: #495057; margin: 0;">อัปโหลดจากไฟล์</h4>
        <p style="color: #6c757d; font-size: 0.9rem; margin: 0.5rem 0 0 0;">เลือกไฟล์จากเครื่องของคุณ</p>
    </div>
    """, unsafe_allow_html=True)

with col_camera:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: #495057; margin: 0;">ถ่ายรูปด้วยกล้อง</h4>
        <p style="color: #6c757d; font-size: 0.9rem; margin: 0.5rem 0 0 0;">ขอสิทธิ์เมื่อกดใช้งาน</p>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #e8f5e8; 
                border: 1px solid #c3e6c3; border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: #2d5016; margin: 0;">ภาพด้านหน้า</h4>
        <p style="color: #2d5016; font-size: 0.85rem; margin: 0.3rem 0 0 0;">
            (บังคับ - สำหรับการวิเคราะห์พระเครื่อง)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tab สำหรับเลือกวิธีการ input
    tab1, tab2 = st.tabs(["อัปโหลด", "ถ่ายรูป"])
    
    with tab1:
        st.markdown("""
        <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 2rem; 
                    text-align: center; margin: 1rem 0; background: #f9f9f9;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">ไฟล์</div>
            <div style="color: #666; font-size: 1rem; margin-bottom: 0.5rem;">
                เลือกไฟล์จากเครื่องของคุณ
            </div>
            <div style="color: #999; font-size: 0.9rem;">
                Limit 200MB per file • JPG, JPEG, PNG, HEIC, HEIF, WEBP, BMP, TIFF, TIF
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        front_file = st.file_uploader(
            "เลือกไฟล์ภาพด้านหน้า", 
            type=SUPPORTED_FORMATS,
            key="front_upload",
            label_visibility="collapsed"
        )
        if front_file:
            st.button("Browse files", key="front_browse", disabled=True)
        front = front_file
        front_source = "upload"
    
    with tab2:
        st.markdown("""
        <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 2rem; 
                    text-align: center; margin: 1rem 0; background: #f9f9f9;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">กล้อง</div>
            <div style="color: #666; font-size: 1rem; margin-bottom: 0.5rem;">
                ถ่ายรูปด้วยกล้อง
            </div>
            <div style="color: #999; font-size: 0.9rem;">
                ขอสิทธิ์เมื่อกดใช้งาน
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("เปิดกล้องถ่ายรูป", key="front_camera_btn", use_container_width=True):
            st.session_state.show_front_camera = True
        
        if st.session_state.get('show_front_camera', False):
            front_camera = st.camera_input(
                "ถ่ายรูปภาพด้านหน้า",
                key="front_camera"
            )
            if front_camera:
                front = front_camera
                front_source = "camera"
                # ซ่อนกล้องหลังถ่ายเสร็จ
                if st.button("ใช้รูปนี้", key="front_camera_confirm"):
                    st.session_state.show_front_camera = False
                    st.rerun()
            else:
                front = front_file if 'front_file' in locals() and front_file else None
                front_source = "upload"
        else:
            front = front_file if 'front_file' in locals() and front_file else None
            front_source = "upload"
    
    # แสดงภาพและตรวจสอบ
    if front:
        is_valid, processed_img, processed_bytes, error_msg = validate_and_convert_image(front)
        if is_valid:
            # Success message with enhanced styling
            st.markdown("""
            <div style="background: linear-gradient(135deg, #d4edda, #c3e6cb); 
                        border: 1px solid #c3e6cb; border-radius: 10px; 
                        padding: 0.8rem; margin: 1rem 0; text-align: center;">
                <div style="color: #155724; font-size: 1rem; font-weight: bold;">
                    ภาพถูกต้อง กำลังแสดงผล...
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced image display
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <h5 style="color: #495057; margin: 0;">ภาพด้านหน้า ({front_source})</h5>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(processed_img, use_container_width=True)
            # เก็บข้อมูลที่ประมวลผลแล้วไว้ใน session_state
            st.session_state.front_processed = processed_bytes
            st.session_state.front_filename = front.name if hasattr(front, 'name') else f"camera_front_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        else:
            # Enhanced error message
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8d7da, #f5c6cb); 
                        border: 1px solid #f5c6cb; border-radius: 10px; 
                        padding: 1rem; margin: 1rem 0; text-align: center;">
                <div style="color: #721c24; font-size: 1rem; font-weight: bold;">
                    ไฟล์ภาพไม่ถูกต้อง: {error_msg}
                </div>
                <div style="color: #856404; font-size: 0.9rem; margin-top: 0.5rem;">
                    ลองใช้รูปภาพอื่น หรือถ่ายรูปใหม่
                </div>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #e8f5e8; 
                border: 1px solid #c3e6c3; border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: #2d5016; margin: 0;">ภาพด้านหลัง</h4>
        <p style="color: #2d5016; font-size: 0.85rem; margin: 0.3rem 0 0 0;">
            (บังคับ - สำหรับการวิเคราะห์ที่ละเอียดยิ่งขึ้น)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["อัปโหลด", "ถ่ายรูป"])
    
    with tab1:
        st.markdown("""
        <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 2rem; 
                    text-align: center; margin: 1rem 0; background: #f9f9f9;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">ไฟล์</div>
            <div style="color: #666; font-size: 1rem; margin-bottom: 0.5rem;">
                เลือกไฟล์จากเครื่องของคุณ
            </div>
            <div style="color: #999; font-size: 0.9rem;">
                Limit 200MB per file • JPG, JPEG, PNG, HEIC, HEIF, WEBP, BMP, TIFF, TIF
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        back_file = st.file_uploader(
            "เลือกไฟล์ภาพด้านหลัง", 
            type=SUPPORTED_FORMATS,
            key="back_upload",
            label_visibility="collapsed"
        )
        if back_file:
            st.button("Browse files", key="back_browse", disabled=True)
        back = back_file
        back_source = "upload"
    
    with tab2:
        st.markdown("""
        <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 2rem; 
                    text-align: center; margin: 1rem 0; background: #f9f9f9;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">กล้อง</div>
            <div style="color: #666; font-size: 1rem; margin-bottom: 0.5rem;">
                ถ่ายรูปด้วยกล้อง
            </div>
            <div style="color: #999; font-size: 0.9rem;">
                ขอสิทธิ์เมื่อกดใช้งาน
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("เปิดกล้องถ่ายรูป", key="back_camera_btn", use_container_width=True):
            st.session_state.show_back_camera = True
        
        if st.session_state.get('show_back_camera', False):
            back_camera = st.camera_input(
                "ถ่ายรูปภาพด้านหลัง",
                key="back_camera"
            )
            if back_camera:
                back = back_camera
                back_source = "camera"
                # ซ่อนกล้องหลังถ่ายเสร็จ
                if st.button("ใช้รูปนี้", key="back_camera_confirm"):
                    st.session_state.show_back_camera = False
                    st.rerun()
            else:
                back = back_file if 'back_file' in locals() and back_file else None
                back_source = "upload"
        else:
            back = back_file if 'back_file' in locals() and back_file else None
            back_source = "upload"
    
    # แสดงภาพและตรวจสอบ
    if back:
        is_valid, processed_img, processed_bytes, error_msg = validate_and_convert_image(back)
        if is_valid:
            # Success message with enhanced styling
            st.markdown("""
            <div style="background: linear-gradient(135deg, #d4edda, #c3e6cb); 
                        border: 1px solid #c3e6cb; border-radius: 10px; 
                        padding: 0.8rem; margin: 1rem 0; text-align: center;">
                <div style="color: #155724; font-size: 1rem; font-weight: bold;">
                    ภาพถูกต้อง กำลังแสดงผล...
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced image display
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <h5 style="color: #495057; margin: 0;">ภาพด้านหลัง ({back_source})</h5>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(processed_img, use_container_width=True)
            # เก็บข้อมูลที่ประมวลผลแล้วไว้ใน session_state
            st.session_state.back_processed = processed_bytes
            st.session_state.back_filename = back.name if hasattr(back, 'name') else f"camera_back_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        else:
            # Enhanced error message
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8d7da, #f5c6cb); 
                        border: 1px solid #f5c6cb; border-radius: 10px; 
                        padding: 1rem; margin: 1rem 0; text-align: center;">
                <div style="color: #721c24; font-size: 1rem; font-weight: bold;">
                    ไฟล์ภาพไม่ถูกต้อง: {error_msg}
                </div>
                <div style="color: #856404; font-size: 0.9rem; margin-top: 0.5rem;">
                    ลองใช้รูปภาพอื่น หรือถ่ายรูปใหม่
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# Analysis Section
if (front and hasattr(st.session_state, 'front_processed') and 
    back and hasattr(st.session_state, 'back_processed')):
    # Enhanced analyze button section
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0 1rem 0;">
        <h3 style="color: #495057; margin: 0;">พร้อมวิเคราะห์แล้ว</h3>
        <p style="color: #6c757d; font-size: 0.9rem;">กดปุ่มด้านล่างเพื่อเริ่มการวิเคราะห์ด้วย AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("วิเคราะห์ตอนนี้", type="primary", use_container_width=True):
        # ใช้ไฟล์ที่ประมวลผลแล้ว - บังคับทั้งหน้าและหลัง
        files = {
            "front": (st.session_state.front_filename, st.session_state.front_processed, "image/jpeg"),
            "back": (st.session_state.back_filename, st.session_state.back_processed, "image/jpeg")
        }
        # Enhanced loading message
        with st.spinner("กำลังประมวลผลด้วย AI... โปรดรอสักครู่"):
            try:
                r = send_predict_request(files, API_URL, timeout=60)
                
                if r.ok:
                    data = r.json()
                    # Enhanced success message
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #d4edda, #c3e6cb); 
                                border: 1px solid #c3e6cb; border-radius: 15px; 
                                padding: 1.5rem; margin: 1.5rem 0; text-align: center;">
                        <div style="color: #155724; font-size: 1.2rem; font-weight: bold;">
                            วิเคราะห์เสร็จสิ้น!
                        </div>
                        <div style="color: #155724; font-size: 0.9rem; margin-top: 0.5rem;">
                            ระบบ AI ได้ประมวลผลข้อมูลเรียบร้อยแล้ว
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Main Result
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 1.5rem 0;">
                        <h2 style="color: #495057; margin: 0;">ผลการวิเคราะห์</h2>
                        <p style="color: #6c757d; font-size: 0.9rem;">ผลลัพธ์จากระบบปัญญาประดิษฐ์</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top-1 Result with enhanced styling
                    confidence_percent = data['top1']['confidence'] * 100
                    
                    # Determine confidence color
                    if confidence_percent >= 80:
                        conf_color = "#155724"
                        bg_color = "linear-gradient(135deg, #d4edda, #c3e6cb)"
                        border_color = "#c3e6cb"
                    elif confidence_percent >= 60:
                        conf_color = "#856404"
                        bg_color = "linear-gradient(135deg, #fff3cd, #ffeaa7)"
                        border_color = "#ffeaa7"
                    else:
                        conf_color = "#721c24"
                        bg_color = "linear-gradient(135deg, #f8d7da, #f5c6cb)"
                        border_color = "#f5c6cb"
                    
                    st.markdown(f"""
                    <div style="padding: 2rem; border-radius: 15px; 
                                background: {bg_color}; 
                                border: 2px solid {border_color};
                                margin: 1.5rem 0; text-align: center;
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                animation: resultBounceIn 1s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                                position: relative;
                                overflow: hidden;">
                        <div style="position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; 
                                    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
                                    animation: resultShimmer 3s infinite; pointer-events: none;"></div>
                        <div style="font-size: 2rem; margin-bottom: 1rem; animation: crownSpin 3s ease-in-out infinite;"></div>
                        <h2 style="color: {conf_color}; margin: 0; font-size: 1.5rem; position: relative; z-index: 1;">
                            {data['top1']['class_name']}
                        </h2>
                        <div style="margin: 1rem 0; font-size: 1.2rem; color: {conf_color}; position: relative; z-index: 1;">
                            <strong>ความน่าจะเป็น: {confidence_percent:.1f}%</strong>
                        </div>
                        <div style="font-size: 0.9rem; color: {conf_color}; opacity: 0.8; position: relative; z-index: 1;">
                            ผลการวิเคราะห์อันดับ 1 จากระบบ AI
                        </div>
                    </div>
                    <style>
                    @keyframes resultBounceIn {{
                        0% {{
                            opacity: 0;
                            transform: scale(0.3) rotate(-10deg);
                        }}
                        50% {{
                            opacity: 1;
                            transform: scale(1.05) rotate(5deg);
                        }}
                        70% {{
                            transform: scale(0.95) rotate(-2deg);
                        }}
                        100% {{
                            opacity: 1;
                            transform: scale(1) rotate(0deg);
                        }}
                    }}
                    
                    @keyframes resultShimmer {{
                        0% {{
                            transform: translateX(-100%) translateY(-100%) rotate(45deg);
                        }}
                        100% {{
                            transform: translateX(100%) translateY(100%) rotate(45deg);
                        }}
                    }}
                    
                    @keyframes crownSpin {{
                        0%, 100% {{
                            transform: rotateY(0deg) scale(1);
                        }}
                        50% {{
                            transform: rotateY(180deg) scale(1.1);
                        }}
                    }}
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Top-3 Results with enhanced styling
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 1.5rem 0;">
                        <h3 style="color: #495057; margin: 0;">ตัวเลือกอื่นๆ (Top-3)</h3>
                        <p style="color: #6c757d; font-size: 0.9rem;">ผลการจัดอันดับทั้งหมดจากระบบ AI</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create enhanced styled results
                    for i, item in enumerate(data['topk'], 1):
                        confidence_pct = item['confidence'] * 100
                        
                        # Medal and styling based on rank
                        if i == 1:
                            icon = ""
                            bg_gradient = "linear-gradient(135deg, #fff3e0, #ffe0b3)"
                            border_color = "#ffcc80"
                            text_color = "#e65100"
                        elif i == 2:
                            icon = ""
                            bg_gradient = "linear-gradient(135deg, #f3e5f5, #ce93d8)"
                            border_color = "#ba68c8"
                            text_color = "#4a148c"
                        else:
                            icon = ""
                            bg_gradient = "linear-gradient(135deg, #fff8e1, #ffecb3)"
                            border_color = "#ffcc02"
                            text_color = "#f57f17"
                        
                        st.markdown(f"""
                        <div style="padding: 1rem; margin: 0.8rem 0; border-radius: 10px;
                                    background: {bg_gradient}; 
                                    border: 1px solid {border_color};
                                    display: flex; align-items: center;
                                    animation: rankSlideIn 0.8s ease-out {i * 0.2}s both;
                                    transition: all 0.3s ease;
                                    position: relative;
                                    overflow: hidden;"
                             onmouseover="this.style.transform='translateX(10px) scale(1.02)'"
                             onmouseout="this.style.transform='translateX(0) scale(1)'">
                            <div style="position: absolute; top: 0; left: -100%; width: 100%; height: 100%; 
                                        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                                        transition: left 0.6s;" class="rank-shimmer"></div>
                            <div style="font-size: 1.5rem; margin-right: 1rem; position: relative; z-index: 1;"></div>
                            <div style="flex-grow: 1; position: relative; z-index: 1;">
                                <div style="font-weight: bold; color: {text_color}; font-size: 1.1rem;">
                                    {item['class_name']}
                                </div>
                                <div style="color: {text_color}; font-size: 0.9rem; opacity: 0.8;">
                                    ความน่าจะเป็น: {confidence_pct:.1f}%
                                </div>
                            </div>
                            <div style="text-align: right; color: {text_color}; position: relative; z-index: 1;">
                                <div style="font-size: 0.8rem; opacity: 0.7;">อันดับ {i}</div>
                            </div>
                        </div>
                        <style>
                        @keyframes rankSlideIn {{
                            0% {{
                                opacity: 0;
                                transform: translateX(-30px);
                            }}
                            100% {{
                                opacity: 1;
                                transform: translateX(0);
                            }}
                        }}
                        
                        .rank-shimmer:hover {{
                            left: 100% !important;
                        }}
                        </style>
                        """, unsafe_allow_html=True)
                        
                        col_rank, col_name, col_conf = st.columns([0.5, 3, 1])
                        with col_rank:
                            st.markdown(f"**{''}**")
                        with col_name:
                            st.markdown(f"**{item['class_name']}**")
                        with col_conf:
                            st.markdown(f"`{confidence_pct:.1f}%`")
                    
                    # Price Valuation
                    st.markdown("---")
                    st.subheader("ช่วงราคาประเมิน")
                    
                    price_col1, price_col2, price_col3 = st.columns(3)
                    with price_col1:
                        st.metric(
                            label="ราคาต่ำ (P05)",
                            value=f"{data['valuation']['p05']:,.0f} ฿",
                            help="ราคาต่ำสุดในตลาด"
                        )
                    with price_col2:
                        st.metric(
                            label="ราคากลาง (P50)",
                            value=f"{data['valuation']['p50']:,.0f} ฿",
                            help="ราคาเฉลี่ยในตลาด"
                        )
                    with price_col3:
                        st.metric(
                            label="ราคาสูง (P95)",
                            value=f"{data['valuation']['p95']:,.0f} ฿",
                            help="ราคาสูงสุดในตลาด"
                        )
                    
                    # Recommendations
                    st.markdown("---")
                    st.subheader("แนะนำช่องทางการขาย")
                    
                    for i, rec in enumerate(data["recommendations"], 1):
                        with st.expander(f"{rec['market']}", expanded=i==1):
                            st.write(f"**เหตุผล:** {rec['reason']}")
                            if rec['market'] == "Facebook Marketplace":
                                st.info("เหมาะสำหรับการขายให้คนทั่วไป")
                            elif rec['market'] == "Shopee":
                                st.info("มีระบบรีวิวและการันตี")
                
                else:
                    st.error(f"เกิดข้อผิดพลาด โปรดลองใหม่อีกครั้ง: {r.status_code} - {r.text}")
                    
            except requests.exceptions.Timeout:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fff3cd, #ffeaa7); 
                            border: 1px solid #ffeaa7; border-radius: 10px; 
                            padding: 1.5rem; margin: 1rem 0; text-align: center;">
                    <div style="color: #856404; font-size: 1.1rem; font-weight: bold;">
                        การประมวลผลใช้เวลานานเกินไป
                    </div>
                    <div style="color: #856404; font-size: 0.9rem; margin-top: 0.5rem;">
                        กรุณาลองใหม่อีกครั้ง หรือลองใช้ภาพที่มีขนาดเล็กกว่า
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except requests.exceptions.ConnectionError:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f8d7da, #f5c6cb); 
                            border: 1px solid #f5c6cb; border-radius: 10px; 
                            padding: 1.5rem; margin: 1rem 0; text-align: center;">
                    <div style="color: #721c24; font-size: 1.1rem; font-weight: bold;">
                        ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ได้
                    </div>
                    <div style="color: #721c24; font-size: 0.9rem; margin-top: 0.5rem;">
                        กรุณาตรวจสอบว่า Backend กำลังทำงานอยู่บนพอร์ต 8000
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8d7da, #f5c6cb); 
                            border: 1px solid #f5c6cb; border-radius: 10px; 
                            padding: 1.5rem; margin: 1rem 0; text-align: center;">
                    <div style="color: #721c24; font-size: 1.1rem; font-weight: bold;">
                        เกิดข้อผิดพลาดไม่คาดคิด
                    </div>
                    <div style="color: #721c24; font-size: 0.9rem; margin-top: 0.5rem;">
                        รายละเอียด: {str(e)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    # ตรวจสอบว่ามีภาพไหนขาดไป
    missing_images = []
    if not (front and hasattr(st.session_state, 'front_processed')):
        missing_images.append("ภาพด้านหน้า")
    if not (back and hasattr(st.session_state, 'back_processed')):
        missing_images.append("ภาพด้านหลัง")
    
    if (front and not hasattr(st.session_state, 'front_processed')) or (back and not hasattr(st.session_state, 'back_processed')):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fff3cd, #ffeaa7); 
                    border: 1px solid #ffeaa7; border-radius: 10px; 
                    padding: 1.5rem; margin: 1rem 0; text-align: center;">
            <div style="color: #856404; font-size: 1.1rem; font-weight: bold;">
                กำลังประมวลผลรูปภาพ...
            </div>
            <div style="color: #856404; font-size: 0.9rem; margin-top: 0.5rem;">
                กรุณารอสักครู่ ระบบกำลังเตรียมข้อมูล
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        missing_text = " และ ".join(missing_images)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #cce7ff, #b3daff); 
                    border: 1px solid #b3daff; border-radius: 10px; 
                    padding: 2rem; margin: 2rem 0; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 1rem;"></div>
            <div style="color: #0056b3; font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">
                เริ่มต้นการวิเคราะห์
            </div>
            <div style="color: #0056b3; font-size: 0.95rem;">
                กรุณาอัปโหลด{missing_text}ก่อนเริ่มการวิเคราะห์
            </div>
            <div style="color: #d32f2f; font-size: 0.9rem; margin-top: 0.8rem; font-weight: bold;">
                จำเป็นต้องมีทั้งภาพหน้าและหลังสำหรับการวิเคราะห์
            </div>
            <div style="color: #0056b3; font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">
                เคล็ดลับ: ภาพที่มีแสงสว่างเพียงพอจะให้ผลลัพธ์ที่ดีกว่า
            </div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Tips section
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin: 2rem 0 1.5rem 0;">
    <h2 style="color: #495057; margin: 0;">เคล็ดลับการถ่ายภาพที่ดี</h2>
    <p style="color: #6c757d; font-size: 0.9rem;">วิธีการถ่ายภาพเพื่อให้ได้ผลการวิเคราะห์ที่แม่นยำที่สุด</p>
</div>
""", unsafe_allow_html=True)

col_tip1, col_tip2, col_tip3 = st.columns(3)

with col_tip1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e3f2fd, #bbdefb); 
                border-radius: 15px; padding: 1.5rem; height: 180px;
                text-align: center; border: 1px solid #90caf9;
                animation: tipFloat1 4s ease-in-out infinite;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;"
         onmouseover="this.style.transform='translateY(-10px) scale(1.05)'"
         onmouseout="this.style.transform='translateY(0) scale(1)'">
        <div style="position: absolute; top: -2px; left: -2px; right: -2px; bottom: -2px;
                    background: linear-gradient(45deg, #90caf9, #e3f2fd, #90caf9);
                    border-radius: 15px; z-index: -1;
                    animation: borderGlow1 3s ease-in-out infinite;"></div>
        <div style="font-size: 2rem; margin-bottom: 0.8rem; animation: iconBounce1 2s ease-in-out infinite;"></div>
        <h4 style="color: #1565c0; margin: 0.5rem 0;">แสงสว่าง</h4>
        <p style="color: #1565c0; font-size: 0.85rem; margin: 0; line-height: 1.4;">
            ถ่ายในที่แสงสว่างเพียงพอ<br>
            หลีกเลี่ยงแสงแรง<br>
            ใช้แสงธรรมชาติ
        </p>
    </div>
    <style>
    @keyframes tipFloat1 {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    @keyframes borderGlow1 {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    @keyframes iconBounce1 {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    </style>
    """, unsafe_allow_html=True)

with col_tip2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e5f5, #e1bee7); 
                border-radius: 15px; padding: 1.5rem; height: 180px;
                text-align: center; border: 1px solid #ce93d8;
                animation: tipFloat2 4s ease-in-out infinite 0.5s;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;"
         onmouseover="this.style.transform='translateY(-10px) scale(1.05)'"
         onmouseout="this.style.transform='translateY(0) scale(1)'">
        <div style="position: absolute; top: -2px; left: -2px; right: -2px; bottom: -2px;
                    background: linear-gradient(45deg, #ce93d8, #f3e5f5, #ce93d8);
                    border-radius: 15px; z-index: -1;
                    animation: borderGlow2 3s ease-in-out infinite 0.5s;"></div>
        <div style="font-size: 2rem; margin-bottom: 0.8rem; animation: iconSpin 3s ease-in-out infinite;"></div>
        <h4 style="color: #6a1b9a; margin: 0.5rem 0;">มุมกล้อง</h4>
        <p style="color: #6a1b9a; font-size: 0.85rem; margin: 0; line-height: 1.4;">
            ถ่ายตรงกลางวัตถุ<br>
            หลีกเลี่ยงมุมเอียง<br>
            ระยะ 20-30 ซม.
        </p>
    </div>
    <style>
    @keyframes tipFloat2 {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    @keyframes borderGlow2 {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    @keyframes iconSpin {
        0% { transform: rotate(0deg) scale(1); }
        50% { transform: rotate(180deg) scale(1.1); }
        100% { transform: rotate(360deg) scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)

with col_tip3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fff3e0, #ffcc80); 
                border-radius: 15px; padding: 1.5rem; height: 180px;
                text-align: center; border: 1px solid #ffb74d;
                animation: tipFloat3 4s ease-in-out infinite 1s;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;"
         onmouseover="this.style.transform='translateY(-10px) scale(1.05)'"
         onmouseout="this.style.transform='translateY(0) scale(1)'">
        <div style="position: absolute; top: -2px; left: -2px; right: -2px; bottom: -2px;
                    background: linear-gradient(45deg, #ffb74d, #fff3e0, #ffb74d);
                    border-radius: 15px; z-index: -1;
                    animation: borderGlow3 3s ease-in-out infinite 1s;"></div>
        <div style="font-size: 2rem; margin-bottom: 0.8rem; animation: iconPulse 2s ease-in-out infinite;"></div>
        <h4 style="color: #e65100; margin: 0.5rem 0;">พื้นหลัง</h4>
        <p style="color: #e65100; font-size: 0.85rem; margin: 0; line-height: 1.4;">
            ใช้พื้นหลังเรียบ<br>
            สีขาวหรือสีอ่อน<br>
            ไม่มีสิ่งรบกวน
        </p>
    </div>
    <style>
    @keyframes tipFloat3 {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }
    @keyframes borderGlow3 {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; }
    }
    @keyframes iconPulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.8; }
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin: 2rem 0 1rem 0;">
    <h3 style="color: #495057; margin: 0;">เทคโนโลยี</h3>
    <p style="color: #6c757d; font-size: 0.9rem;">ระบบปัญญาประดิษฐ์ขั้นสูงสำหรับการวิเคราะห์พระเครื่อง</p>
</div>
""", unsafe_allow_html=True)

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;
                animation: infoCardSlide 1s ease-out;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;"
         onmouseover="this.style.transform='translateY(-8px) rotateY(5deg)'"
         onmouseout="this.style.transform='translateY(0) rotateY(0)'">
        <div style="position: absolute; top: 0; left: -100%; width: 100%; height: 100%; 
                    background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
                    animation: infoShimmer 3s infinite;"></div>
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem; animation: techIconFloat 2s ease-in-out infinite;"></div>
        <h4 style="color: #495057; margin: 0.5rem 0;">AI Technology</h4>
        <p style="color: #6c757d; font-size: 0.85rem; margin: 0;">TensorFlow + FastAPI</p>
    </div>
    <style>
    @keyframes infoCardSlide {
        0% { opacity: 0; transform: translateX(-30px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    @keyframes techIconFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-3px); }
    }
    @keyframes infoShimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    </style>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;
                animation: infoCardSlide 1s ease-out 0.2s both;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;"
         onmouseover="this.style.transform='translateY(-8px) rotateY(-5deg)'"
         onmouseout="this.style.transform='translateY(0) rotateY(0)'">
        <div style="position: absolute; top: 0; left: -100%; width: 100%; height: 100%; 
                    background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
                    animation: infoShimmer 3s infinite 0.5s;"></div>
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem; animation: formatIconSpin 4s linear infinite;"></div>
        <h4 style="color: #495057; margin: 0.5rem 0;">Multi-Format</h4>
        <p style="color: #6c757d; font-size: 0.85rem; margin: 0;">JPG, PNG, HEIC & More</p>
    </div>
    <style>
    @keyframes formatIconSpin {
        0% { transform: rotateY(0deg); }
        100% { transform: rotateY(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)

with col_info3:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;
                animation: infoCardSlide 1s ease-out 0.4s both;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;"
         onmouseover="this.style.transform='translateY(-8px) scale(1.05)'"
         onmouseout="this.style.transform='translateY(0) scale(1)'">
        <div style="position: absolute; top: 0; left: -100%; width: 100%; height: 100%; 
                    background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
                    animation: infoShimmer 3s infinite 1s;"></div>
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem; animation: cameraIconPulse 1.5s ease-in-out infinite;"></div>
        <h4 style="color: #495057; margin: 0.5rem 0;">Camera Ready</h4>
        <p style="color: #6c757d; font-size: 0.85rem; margin: 0;">ถ่ายรูปได้ทันที</p>
    </div>
    <style>
    @keyframes cameraIconPulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    </style>
    """, unsafe_allow_html=True)

# Development info
with st.expander("Developer Info"):
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.write(f"API URL: {API_URL}")
    st.write("Developed with Streamlit & FastAPI")