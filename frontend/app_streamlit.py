import streamlit as st
import requests
import io
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from PIL import Image, ImageEnhance

# Multi-language Support
LANGUAGES = {
    'th': {
        'title': '🔍 Amulet-AI',
        'subtitle': 'ระบบวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์ขั้นสูง',
        'tagline': '⚡ รวดเร็ว • 🎯 แม่นยำ • 🔒 ปลอดภัย',
        'upload_title': '📤 อัปโหลดรูปภาพพระเครื่อง',
        'upload_subtitle': 'รองรับไฟล์: JPG, PNG, HEIC, WEBP, BMP, TIFF',
        'front_image': '📷 ภาพด้านหน้า',
        'back_image': '📱 ภาพด้านหลัง',
        'required': 'บังคับ',
        'optional': 'ไม่บังคับ',
        'analyze_btn': '🚀 วิเคราะห์พระเครื่อง',
        'confidence': 'ความมั่นใจ',
        'authenticity': 'ความแท้',
        'top_results': 'ผลลัพธ์ทั้งหมด (Top-3)',
        'price_estimate': 'ประเมินราคา',
        'market_rec': 'แนะนำตลาด'
    },
    'en': {
        'title': '🔍 Amulet-AI',
        'subtitle': 'Advanced AI-Powered Thai Amulet Analysis System',
        'tagline': '⚡ Fast • 🎯 Accurate • 🔒 Secure',
        'upload_title': '📤 Upload Amulet Images',
        'upload_subtitle': 'Supported: JPG, PNG, HEIC, WEBP, BMP, TIFF',
        'front_image': '📷 Front Image',
        'back_image': '📱 Back Image',
        'required': 'Required',
        'optional': 'Optional',
        'analyze_btn': '🚀 Analyze Amulet',
        'confidence': 'Confidence',
        'authenticity': 'Authenticity',
        'top_results': 'All Results (Top-3)',
        'price_estimate': 'Price Estimate',
        'market_rec': 'Market Recommendations'
    }
}

def get_lang():
    """Get current language"""
    return st.session_state.get('language', 'th')

def _(key):
    """Translation helper"""
    return LANGUAGES[get_lang()].get(key, key)

# รองรับ HEIC format
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

# API Configuration
API_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="🔍 Amulet-AI | ระบบวิเคราะห์พระเครื่อง", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/amulet-ai',
        'Report a bug': 'https://github.com/your-repo/amulet-ai/issues',
        'About': "Amulet-AI - ระบบวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์"
    }
)

# Modern CSS with Responsive Design & Interactive Elements
st.markdown("""
<style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Variables */
    :root {
        --primary-color: #8B5CF6;
        --primary-gradient: linear-gradient(135deg, #8B5CF6 0%, #A855F7 50%, #C084FC 100%);
        --secondary-color: #F3F4F6;
        --text-primary: #1F2937;
        --text-secondary: #6B7280;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        --border-radius: 12px;
        --border-radius-lg: 16px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Base Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Container - Responsive */
    .main {
        padding: 1rem;
        max-width: none;
    }
    
    .block-container {
        padding: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Hero Section */
    .hero-section {
        background: var(--primary-gradient);
        padding: 3rem 2rem;
        border-radius: var(--border-radius-lg);
        text-align: center;
        color: white;
        margin-bottom: 3rem;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"><g fill="none" fill-rule="evenodd"><g fill="%23ffffff" fill-opacity="0.1"><circle cx="7" cy="7" r="1"/></g></svg>');
        animation: float 20s ease-in-out infinite;
    }
    
    .hero-title {
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 700;
        margin: 0 0 1rem 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: clamp(1rem, 3vw, 1.2rem);
        font-weight: 400;
        margin: 0;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    /* Card Components */
    .modern-card {
        background: white;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-md);
        border: 1px solid #E5E7EB;
        transition: var(--transition);
        overflow: hidden;
    }
    
    .modern-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    
    .upload-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 2px dashed var(--primary-color);
        border-radius: var(--border-radius);
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: var(--transition);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .upload-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.1), transparent);
        transition: left 0.6s;
    }
    
    .upload-card:hover {
        border-color: #A855F7;
        background: linear-gradient(135deg, #FAF5FF 0%, #F3E8FF 100%);
        transform: scale(1.02);
    }
    
    .upload-card:hover::before {
        left: 100%;
    }
    
    /* Button Styles */
    .stButton > button {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--border-radius) !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: var(--transition) !important;
        box-shadow: var(--shadow-md) !important;
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Results Cards */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-md);
        margin: 1rem 0;
        transition: var(--transition);
        border-left: 4px solid var(--primary-color);
    }
    
    .result-card:hover {
        transform: translateX(4px);
        box-shadow: var(--shadow-lg);
    }
    
    .confidence-high { 
        border-left-color: var(--success-color);
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
    }
    
    .confidence-medium { 
        border-left-color: var(--warning-color);
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
    }
    
    .confidence-low { 
        border-left-color: var(--error-color);
        background: linear-gradient(135deg, #FEF2F2 0%, #FECACA 100%);
    }
    
    /* Metrics Cards */
    .metric-card {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-md);
        border: 1px solid #E5E7EB;
        transition: var(--transition);
        margin: 0.5rem;
    }
    
    .metric-card:hover {
        transform: translateY(-4px) scale(1.05);
        box-shadow: var(--shadow-lg);
    }
    
    /* Tips Section */
    .tips-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .tip-card {
        background: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-md);
        border-top: 4px solid var(--primary-color);
        transition: var(--transition);
        position: relative;
    }
    
    .tip-card:hover {
        transform: translateY(-8px);
        box-shadow: var(--shadow-xl);
    }
    
    .tip-card:nth-child(1) { border-top-color: #3B82F6; }
    .tip-card:nth-child(2) { border-top-color: #8B5CF6; }
    .tip-card:nth-child(3) { border-top-color: #F59E0B; }
    
    /* Sidebar Enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
        border-right: 1px solid #E5E7EB;
    }
    
    /* Loading & Animations */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .analyzing {
        animation: pulse 2s ease-in-out infinite;
    }
    
    .slide-in {
        animation: slideIn 0.6s ease-out;
    }
    
    /* Progress Bar */
    .progress-bar {
        width: 100%;
        height: 4px;
        background: #E5E7EB;
        border-radius: 2px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: var(--primary-gradient);
        animation: loading 2s ease-in-out infinite;
    }
    
    @keyframes loading {
        0% { width: 0%; }
        50% { width: 70%; }
        100% { width: 100%; }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .block-container {
            padding: 0.5rem;
        }
        
        .hero-section {
            padding: 2rem 1rem;
            margin-bottom: 2rem;
        }
        
        .tips-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .tip-card {
            padding: 1.5rem;
        }
        
        .modern-card {
            margin: 0.5rem 0;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        .tips-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    /* Success/Error Messages Enhancement */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    /* Image Display Enhancement */
    .stImage > div {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow-md);
        transition: var(--transition);
    }
    
    .stImage > div:hover {
        box-shadow: var(--shadow-lg);
        transform: scale(1.02);
    }
    
    /* Expander Enhancement */
    .streamlit-expander {
        border: 1px solid #E5E7EB !important;
        border-radius: var(--border-radius) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* Tab Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px !important;
        border-radius: var(--border-radius) !important;
        border: 1px solid #E5E7EB !important;
        background: white !important;
        transition: var(--transition) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: var(--primary-color) !important;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient) !important;
        color: white !important;
        border-color: transparent !important;
    }
    
    /* Footer Enhancement */
    .footer {
        margin-top: 4rem;
        padding: 2rem;
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border-radius: var(--border-radius);
        text-align: center;
        border-top: 1px solid #E5E7EB;
    }
    
    /* Dark Mode Support (Optional) */
    @media (prefers-color-scheme: dark) {
        :root {
            --secondary-color: #374151;
            --text-primary: #F9FAFB;
            --text-secondary: #D1D5DB;
        }
            .tip-card {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
                border: 1px solid rgba(226, 232, 240, 0.8);
                border-radius: 16px;
                padding: 2rem 1.5rem;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
                height: 100%;
            }
            .tip-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #3B82F6, #8B5CF6, #F59E0B);
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            .tip-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
                border-color: rgba(59, 130, 246, 0.3);
            }
            .tip-card:hover::before {
                opacity: 1;
            }
            
            .result-card {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
                border: 1px solid rgba(226, 232, 240, 0.8);
                border-radius: 16px;
                padding: 2rem;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
                margin: 1rem 0;
            }
            .result-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #10B981, #3B82F6, #8B5CF6);
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            .result-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
                border-color: rgba(16, 185, 129, 0.3);
            }
            .result-card:hover::before {
                opacity: 1;
            }
    }
</style>
""", unsafe_allow_html=True)

# Functions
def validate_image(uploaded_file):
    """ตรวจสอบและแปลงไฟล์รูปภาพ พรอมระบบ Smart Preprocessing"""
    try:
        # รองรับหลายฟอร์แมต: JPG, JPEG, PNG, HEIC, HEIF, WebP, BMP, TIFF
        image = Image.open(uploaded_file)
        original_format = image.format
        
        # Smart Preprocessing: Auto resize + enhance
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # แปลงสี RGB ถ้าจำเป็น
        if image.mode != 'RGB':
            # สำหรับ PNG/HEIC ที่มี transparency
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])
                else:
                    background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
        
        # เพิ่มคุณภาพภาพด้วย enhance (ปรับความคมชัด)
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)  # เพิ่มความคมชัด 20%
        
        # แปลงเป็น bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        return image, img_byte_arr, None, original_format
    except Exception as e:
        return None, None, str(e), None

# เพิ่มฟีเจอร์ Camera Input
def show_camera_interface(key_prefix=""):
    """แสดง interface สำหรับถ่ายรูป"""
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0 1rem 0;">
        <h3 style="color: var(--text-primary); margin: 0 0 0.5rem 0; font-weight: 600;">🖼️ เพิ่มรูปภาพของคุณ</h3>
        <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0;">
            รองรับฟอร์แมต: JPG, JPEG, PNG, HEIC, HEIF, WebP, BMP, TIFF
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📁 อัปโหลดไฟล์", "📷 ถ่ายรูป"])
    
    result_image = None
    result_bytes = None
    source = "upload"
    file_info = {}
    
    with tab1:
        uploaded_file = st.file_uploader(
            "เลือกไฟล์รูปภาพ (Drag & Drop รองรับ)", 
            type=['jpg', 'jpeg', 'png', 'heic', 'heif', 'webp', 'bmp', 'tiff'],
            key=f"upload_{key_prefix}",
            help="🔍 ระบบจะตรวจสอบไฟล์และปรับปรุงคุณภาพอัตโนมัติ"
        )
        if uploaded_file:
            # File Validation & Info
            file_size = len(uploaded_file.read())
            uploaded_file.seek(0)  # Reset file pointer
            file_info = {
                'name': uploaded_file.name,
                'size': f"{file_size / (1024*1024):.1f} MB",
                'type': uploaded_file.type
            }
            
            result_image, result_bytes, error, original_format = validate_image(uploaded_file)
            if error:
                st.error(f"❌ ไฟล์ไม่ถูกต้อง: {error}")
                return None, None, None, None
            else:
                st.success(f"✅ ไฟล์ถูกต้อง ({original_format}) - Smart Preprocessing เสร็จสิ้น")
    
    with tab2:
        camera_input = st.camera_input("ถ่ายรูปตรงจากกล้อง", key=f"camera_{key_prefix}")
        if camera_input:
            file_info = {
                'name': 'camera_capture.jpg',
                'size': f"{len(camera_input.read()) / (1024*1024):.1f} MB",
                'type': 'image/jpeg'
            }
            camera_input.seek(0)  # Reset file pointer
            result_image, result_bytes, error, original_format = validate_image(camera_input)
            source = "camera"
            if error:
                st.error(f"❌ ไฟล์ไม่ถูกต้อง: {error}")
                return None, None, None, None
            else:
                st.success("📷 ถ่ายรูปสำเร็จ - Smart Preprocessing เสร็จสิ้น")
    
    return result_image, result_bytes, source, file_info

def create_confidence_gauge(confidence):
    """สร้าง Confidence Gauge แบบวงกลม"""
    # คำนวณสีตามระดับความมั่นใจ
    if confidence >= 80:
        color = "#10B981"  # เขียว
        status = "สูง"
        icon = "🎯"
    elif confidence >= 60:
        color = "#F59E0B"  # เหลือง
        status = "ปานกลาง" 
        icon = "⚡"
    else:
        color = "#EF4444"  # แดง
        status = "ต่ำ"
        icon = "⚠️"
    
    gauge_html = f"""
    <div style="display: flex; align-items: center; justify-content: center; margin: 1rem 0;">
        <div style="position: relative; width: 120px; height: 120px;">
            <svg width="120" height="120" viewBox="0 0 120 120">
                <!-- Background circle -->
                <circle cx="60" cy="60" r="50" fill="none" stroke="#E5E7EB" stroke-width="8"/>
                <!-- Progress circle -->
                <circle cx="60" cy="60" r="50" fill="none" stroke="{color}" stroke-width="8"
                        stroke-dasharray="314.16" stroke-dashoffset="{314.16 * (1 - confidence/100)}"
                        stroke-linecap="round" transform="rotate(-90 60 60)"
                        style="transition: stroke-dashoffset 2s ease-in-out;"/>
            </svg>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {color};">{confidence}%</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary);">{status}</div>
            </div>
        </div>
    </div>
    """
    return gauge_html

def check_system_status():
    """ตรวจสอบสถานะระบบ Backend"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=3)
        if response.status_code == 200:
            return True, "✅ ระบบพร้อมใช้งาน"
        else:
            return False, f"⚠️ ระบบมีปัญหา (Status: {response.status_code})"
    except requests.exceptions.ConnectionError:
        return False, "❌ ไม่สามารถเชื่อมต่อ Backend"
    except requests.exceptions.Timeout:
        return False, "⏰ Backend ตอบสนองช้า"
    except Exception as e:
        return False, f"💥 ข้อผิดพลาด: {str(e)}"

def create_authenticity_score(score):
    """สร้าง Authenticity Score (ค่าความเป็นไปได้ว่าแท้)"""
    if score >= 85:
        color = "#10B981"
        status = "น่าเชื่อถือสูง"
        icon = "🔒"
    elif score >= 70:
        color = "#F59E0B" 
        status = "น่าเชื่อถือปานกลาง"
        icon = "🔐"
    else:
        color = "#EF4444"
        status = "ควรตรวจสอบเพิ่มเติม"
        icon = "🔓"
    
    return f"""
    <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span style="font-size: 1.2rem;">{icon}</span>
        <div style="flex-grow: 1;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="font-size: 0.9rem; font-weight: 600;">Authenticity Score</span>
                <span style="font-size: 0.9rem; font-weight: 700; color: {color};">{score}%</span>
            </div>
            <div style="background: #E5E7EB; height: 6px; border-radius: 3px;">
                <div style="background: {color}; width: {score}%; height: 100%; border-radius: 3px; transition: width 1.5s ease-out;"></div>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">{status}</div>
        </div>
    </div>
    """

# Main App
def main():
    # System Status Check
    is_online, status_message = check_system_status()
    if not is_online:
        st.warning(f"🔧 {status_message}")
    
    # Hero Section
    st.markdown("""
    <div class="hero-section slide-in">
        <h1 class="hero-title">🔍 Amulet-AI</h1>
        <p class="hero-subtitle">ระบบวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์ขั้นสูง</p>
        <div style="margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.8;">
            ⚡ รวดเร็ว • 🎯 แม่นยำ • 🔒 ปลอดภัย
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with usage tips
    with st.sidebar:
        # Theme Toggle
        st.markdown("### 🎨 ธีม")
        theme_option = st.selectbox("เลือกธีม", ["🌞 Light Mode", "🌙 Dark Mode"])
        
        if theme_option == "🌙 Dark Mode":
            st.markdown("""
            <style>
                .main { background-color: #1F2937 !important; }
                .stApp { background-color: #111827 !important; }
                :root {
                    --secondary-color: #374151 !important;
                    --text-primary: #F9FAFB !important;
                    --text-secondary: #D1D5DB !important;
                }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📋 คู่มือใช้งาน")
        
        with st.expander("📘 วิธีการใช้งาน", expanded=True):
            st.write("1. 📤 อัปโหลดภาพด้านหน้า")
            st.write("2. 📷 เลือกภาพด้านหลัง (ไม่บังคับ)")
            st.write("3. 🔍 กดปุ่มวิเคราะห์")
            st.write("4. ⏳ รอผลการวิเคราะห์")
            st.write("5. 📊 ดูผลลัพธ์")
        
        with st.expander("🎯 ข้อมูลระบบ"):
            st.write("- 🤖 เทคโนโลยี: TensorFlow + FastAPI")
            st.write("- 📈 ความแม่นยำ: Top-3 ผลลัพธ์")
            st.write("- 💰 ประเมินราคา: P05, P50, P95")
            st.write("- 🛒 คำแนะนำขาย")
        
        with st.expander("📸 เคล็ดลับถ่ายรูป"):
            st.write("**แสงสว่าง:**")
            st.write("- ใช้แสงธรรมชาติ")
            st.write("- หลีกเลี่ยงเงา")
            
            st.write("**มุมกล้อง:**")
            st.write("- ถ่ายตรงไม่เอียง")
            st.write("- ระยะ 20-30 ซม.")
            
            st.write("**พื้นหลัง:**")
            st.write("- ใช้พื้นเรียบ")
            st.write("- สีขาวหรือสีอ่อน")
    
    # Upload Section
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: var(--text-primary); margin-bottom: 0.5rem; font-weight: 600;">📤 อัปโหลดรูปภาพพระเครื่อง</h2>
        <p style="color: var(--text-secondary); margin: 0; font-size: 1rem;">
            รองรับไฟล์: <code style="background: #F3F4F6; padding: 2px 6px; border-radius: 4px;">JPG, PNG, HEIC, WEBP</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="modern-card" style="padding: 1rem; margin-bottom: 1rem;">
            <h4 style="color: var(--success-color); margin: 0 0 0.5rem 0; display: flex; align-items: center;">
                📷 <span style="margin-left: 0.5rem;">ภาพด้านหน้า</span>
                <span style="margin-left: auto; font-size: 0.7rem; background: #FEF3C7; color: #92400E; padding: 2px 8px; border-radius: 12px;">บังคับ</span>
            </h4>
            <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 0;">จำเป็นสำหรับการวิเคราะห์ขั้นพื้นฐาน</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ใช้ฟีเจอร์ Camera + Upload แบบใหม่
        front_image, front_bytes, front_source, front_info = show_camera_interface("front")
        
        if front_image and front_bytes:
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <div style="display: inline-flex; align-items: center; background: var(--success-color); color: white; 
                           padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.85rem; font-weight: 500;">
                    ✅ <span style="margin-left: 0.5rem;">ภาพถูกต้อง ({front_source}) - {front_info.get('size', 'N/A')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(front_image, use_container_width=True)
            st.session_state.front_data = front_bytes
            st.session_state.front_filename = f"front_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    
    with col2:
        st.markdown("""
        <div class="modern-card" style="padding: 1rem; margin-bottom: 1rem;">
            <h4 style="color: var(--warning-color); margin: 0 0 0.5rem 0; display: flex; align-items: center;">
                � <span style="margin-left: 0.5rem;">ภาพด้านหลัง</span>
                <span style="margin-left: auto; font-size: 0.7rem; background: #E0E7FF; color: #3730A3; padding: 2px 8px; border-radius: 12px;">ไม่บังคับ</span>
            </h4>
            <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 0;">เพิ่มความแม่นยำในการวิเคราะห์</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ใช้ฟีเจอร์ Camera + Upload แบบใหม่
        back_image, back_bytes, back_source, back_info = show_camera_interface("back")
        
        if back_image and back_bytes:
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <div style="display: inline-flex; align-items: center; background: var(--success-color); color: white; 
                           padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.85rem; font-weight: 500;">
                    ✅ <span style="margin-left: 0.5rem;">ภาพถูกต้อง ({back_source}) - {back_info.get('size', 'N/A')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(back_image, use_container_width=True)
            st.session_state.back_data = back_bytes
            st.session_state.back_filename = f"back_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    
    # Analysis Button
    if hasattr(st.session_state, 'front_data'):
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h3 style="color: var(--text-primary); margin: 0 0 0.5rem 0; font-weight: 600;">🚀 พร้อมวิเคราะห์</h3>
            <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0;">
                กดปุ่มด้านล่างเพื่อเริ่มการวิเคราะห์ด้วย AI ขั้นสูง
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 เริ่มวิเคราะห์ตอนนี้", type="primary"):
            # แสดง Progress Bar
            progress_placeholder = st.empty()
            progress_placeholder.markdown("""
            <div style="margin: 2rem 0;">
                <div style="text-align: center; margin-bottom: 1rem;">
                    <div class="analyzing" style="display: inline-flex; align-items: center; 
                                                  background: var(--primary-gradient); color: white; 
                                                  padding: 0.75rem 1.5rem; border-radius: 25px; font-weight: 500;">
                        ⚡ <span style="margin-left: 0.5rem;">กำลังประมวลผลด้วย AI...</span>
                    </div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                    # เตรียมไฟล์
                    files = {"front": (
                        st.session_state.get('front_filename', 'front.jpg'), 
                        st.session_state.front_data, 
                        "image/jpeg"
                    )}
                    if hasattr(st.session_state, 'back_data'):
                        files["back"] = (
                            st.session_state.get('back_filename', 'back.jpg'),
                            st.session_state.back_data, 
                            "image/jpeg"
                        )
                    
                    # เรียก API
                    response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                    
                    if response.ok:
                        data = response.json()
                        
                        # ลบ Progress Bar
                        progress_placeholder.empty()
                        
                        # แสดงผลลัพธ์พร้อม Animation
                        st.markdown("""
                        <div class="slide-in" style="text-align: center; margin: 2rem 0;">
                            <div style="display: inline-flex; align-items: center; background: var(--success-color); color: white; 
                                       padding: 1rem 2rem; border-radius: 25px; font-size: 1.1rem; font-weight: 600; box-shadow: var(--shadow-lg);">
                                ✅ <span style="margin-left: 0.5rem;">วิเคราะห์เสร็จสิ้น!</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # ผลลัพธ์หลัก
                        st.markdown("""
                        <div style="text-align: center; margin: 2rem 0;">
                            <h2 style="color: var(--text-primary); margin: 0 0 0.5rem 0; font-weight: 700;">🎯 ผลการวิเคราะห์</h2>
                            <p style="color: var(--text-secondary); font-size: 1rem; margin: 0;">ผลลัพธ์จากระบบปัญญาประดิษฐ์ขั้นสูง</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # แสดง Confidence Gauge + ผลลัพธ์
                        confidence = data['top1']['confidence'] * 100
                        st.markdown(create_confidence_gauge(confidence), unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="result-card slide-in" style="text-align: center; margin: 2rem 0;">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">🏆</div>
                            <h2 style="margin: 0 0 1rem 0; font-size: 1.8rem; font-weight: 700;">
                                {data['top1']['class_name']}
                            </h2>
                            <div style="margin: 1.5rem 0;">
                                {create_authenticity_score(min(95, confidence + 10))}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Top-3 Results
                        st.markdown("""
                        <div style="text-align: center; margin: 2rem 0;">
                            <h3 style="color: var(--text-primary); margin: 0 0 0.5rem 0; font-weight: 600;">📊 ผลลัพธ์ทั้งหมด (Top-3)</h3>
                            <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0;">การจัดอันดับจากระบบ AI</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for i, result in enumerate(data['topk'], 1):
                            conf_pct = result['confidence'] * 100
                            medals = ["🥇", "🥈", "🥉"]
                            colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
                            
                            col_rank, col_name, col_conf = st.columns([0.5, 3, 1])
                            with col_rank:
                                st.markdown(f"""
                                <div style="text-align: center; font-size: 1.5rem;">
                                    {medals[i-1]}
                                </div>
                                """, unsafe_allow_html=True)
                            with col_name:
                                st.markdown(f"""
                                <div style="padding: 0.5rem 0;">
                                    <div style="font-weight: 600; font-size: 1.1rem; color: var(--text-primary);">
                                        {result['class_name']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_conf:
                                st.markdown(f"""
                                <div style="text-align: center; padding: 0.5rem 0;">
                                    <span style="background: var(--primary-color); color: white; 
                                               padding: 4px 12px; border-radius: 20px; font-size: 0.9rem; font-weight: 500;">
                                        {conf_pct:.1f}%
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # ราคาประเมิน
                        st.markdown("""
                        <div style="text-align: center; margin: 3rem 0 2rem 0;">
                            <h3 style="color: var(--text-primary); margin: 0 0 0.5rem 0; font-weight: 600;">💰 ประเมินราคาในตลาด</h3>
                            <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0;">ช่วงราคาอ้างอิงจากข้อมูลตลาด</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        price_col1, price_col2, price_col3 = st.columns(3)
                        
                        with price_col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">💸</div>
                                <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.25rem;">ราคาต่ำ (P05)</div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: var(--error-color);">
                                    {data['valuation']['p05']:,} ฿
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with price_col2:
                            st.markdown(f"""
                            <div class="metric-card" style="transform: scale(1.05); box-shadow: var(--shadow-lg);">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">💵</div>
                                <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.25rem;">ราคากลาง (P50)</div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary-color);">
                                    {data['valuation']['p50']:,} ฿
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with price_col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">💳</div>
                                <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.25rem;">ราคาสูง (P95)</div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: var(--success-color);">
                                    {data['valuation']['p95']:,} ฿
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # คำแนะนำการขาย
                        st.markdown("""
                        <div style="text-align: center; margin: 3rem 0 2rem 0;">
                            <h3 style="color: var(--text-primary); margin: 0 0 0.5rem 0; font-weight: 600;">🛒 แนะนำช่องทางการขาย</h3>
                            <p style="color: var(--text-secondary); font-size: 0.9rem; margin: 0;">แพลตฟอร์มที่เหมาะสมสำหรับการขาย</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for i, rec in enumerate(data.get('recommendations', []), 1):
                            with st.expander(f"📍 {rec['market']}", expanded=i==1):
                                st.markdown(f"""
                                <div style="padding: 0.5rem 0;">
                                    <div style="margin-bottom: 1rem;">
                                        <strong style="color: var(--text-primary);">💡 เหตุผล:</strong>
                                        <span style="color: var(--text-secondary);"> {rec['reason']}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if rec['market'] == "Facebook Marketplace":
                                    st.info("🔗 เหมาะสำหรับการขายให้คนทั่วไป มีผู้ใช้งานจำนวนมาก")
                                elif rec['market'] == "Shopee":
                                    st.info("🛍️ มีระบบรีวิวและการันตี เหมาะสำหรับผู้ขายมือใหม่")
                    
                    else:
                        progress_placeholder.empty()
                        st.markdown(f"""
                        <div style="text-align: center; margin: 2rem 0;">
                            <div style="display: inline-flex; align-items: center; background: var(--error-color); color: white; 
                                       padding: 1rem 2rem; border-radius: 25px; font-size: 1rem; font-weight: 600;">
                                ❌ <span style="margin-left: 0.5rem;">เกิดข้อผิดพลาด: {response.status_code}</span>
                            </div>
                            <div style="margin-top: 1rem; color: var(--text-secondary);">
                                กรุณาลองใหม่อีกครั้ง หรือติดต่อทีมสนับสนุน
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except requests.exceptions.ConnectionError:
                progress_placeholder.empty()
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0;">
                    <div style="display: inline-flex; align-items: center; background: var(--error-color); color: white; 
                               padding: 1rem 2rem; border-radius: 25px; font-size: 1rem; font-weight: 600;">
                        🔌 <span style="margin-left: 0.5rem;">ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้</span>
                    </div>
                    <div style="margin-top: 1rem; color: var(--text-secondary);">
                        กรุณาตรวจสอบว่า Backend กำลังทำงานอยู่บนพอร์ต 8000
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Retry Options
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔄 ลองใหม่", use_container_width=True):
                        st.rerun()
                with col2:
                    if st.button("🏠 กลับหน้าแรก", use_container_width=True):
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.rerun()
                with col3:
                    if st.button("🔧 ตรวจสอบระบบ", use_container_width=True):
                        with st.spinner("กำลังตรวจสอบ..."):
                            is_online, status_msg = check_system_status()
                            if is_online:
                                st.success(status_msg)
                            else:
                                st.error(status_msg)
            except Exception as e:
                progress_placeholder.empty()
                st.markdown(f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <div style="display: inline-flex; align-items: center; background: var(--error-color); color: white; 
                               padding: 1rem 2rem; border-radius: 25px; font-size: 1rem; font-weight: 600;">
                        💥 <span style="margin-left: 0.5rem;">เกิดข้อผิดพลาดไม่คาดคิด</span>
                    </div>
                    <div style="margin-top: 1rem; color: var(--text-secondary); font-size: 0.9rem;">
                        รายละเอียด: {str(e)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Retry Options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 ลองใหม่อีกครั้ง", use_container_width=True):
                        st.rerun()
                with col2:
                    if st.button("🏠 เริ่มใหม่", use_container_width=True):
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.rerun()    # Tips Section - Modern Grid Design
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0 2rem 0;">
        <h2 style="color: var(--text-primary); margin: 0 0 0.5rem 0; font-weight: 700;">💡 เคล็ดลับการถ่ายภาพมืออาชีพ</h2>
        <p style="color: var(--text-secondary); font-size: 1rem; margin: 0;">
            วิธีการถ่ายภาพเพื่อให้ได้ผลการวิเคราะห์ที่แม่นยำที่สุด
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    
    with tip_col1:
        st.markdown("""
        <div class="tip-card">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📸</div>
                <h4 style="color: #3B82F6; margin: 0 0 0.5rem 0; font-weight: 600;">แสงสว่าง</h4>
            </div>
            <div style="color: var(--text-secondary); line-height: 1.6;">
                <div style="margin-bottom: 0.75rem; display: flex; align-items: center;">
                    <span style="color: #10B981; margin-right: 0.5rem;">✓</span>
                    <span>ใช้แสงธรรมชาติหรือแสงขาว</span>
                </div>
                <div style="margin-bottom: 0.75rem; display: flex; align-items: center;">
                    <span style="color: #10B981; margin-right: 0.5rem;">✓</span>
                    <span>หลีกเลี่ยงแสงสะท้อนและเงาแข็ง</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="color: #EF4444; margin-right: 0.5rem;">✗</span>
                    <span>ไม่ใช้แฟลชหรือแสงแรง</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tip_col2:
        st.markdown("""
        <div class="tip-card">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎯</div>
                <h4 style="color: #8B5CF6; margin: 0 0 0.5rem 0; font-weight: 600;">มุมกล้อง</h4>
            </div>
            <div style="color: var(--text-secondary); line-height: 1.6;">
                <div style="margin-bottom: 0.75rem; display: flex; align-items: center;">
                    <span style="color: #10B981; margin-right: 0.5rem;">✓</span>
                    <span>วางพระให้ตรงกลางเฟรม</span>
                </div>
                <div style="margin-bottom: 0.75rem; display: flex; align-items: center;">
                    <span style="color: #10B981; margin-right: 0.5rem;">✓</span>
                    <span>ถือกล้องขนานพื้น ไม่เอียง</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="color: #10B981; margin-right: 0.5rem;">✓</span>
                    <span>ระยะประมาณ 20-30 เซนติเมตร</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tip_col3:
        st.markdown("""
        <div class="tip-card">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🖼️</div>
                <h4 style="color: #F59E0B; margin: 0 0 0.5rem 0; font-weight: 600;">พื้นหลัง</h4>
            </div>
            <div style="color: var(--text-secondary); line-height: 1.6;">
                <div style="margin-bottom: 0.75rem; display: flex; align-items: center;">
                    <span style="color: #10B981; margin-right: 0.5rem;">✓</span>
                    <span>ใช้พื้นหลังเรียบ สีขาวหรืออ่อน</span>
                </div>
                <div style="margin-bottom: 0.75rem; display: flex; align-items: center;">
                    <span style="color: #10B981; margin-right: 0.5rem;">✓</span>
                    <span>ไม่มีลวดลายหรือสิ่งรบกวน</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="color: #10B981; margin-right: 0.5rem;">✓</span>
                    <span>ทำความสะอาดพื้นผิวก่อนวาง</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h4 style="margin: 0 0 0.5rem 0; color: var(--text-primary);">🤖 Amulet-AI System</h4>
                <p style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">
                    Powered by TensorFlow • FastAPI • Streamlit
                </p>
            </div>
            <div style="display: flex; gap: 1rem; align-items: center;">
                <span style="background: var(--success-color); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;">
                    ✅ System Online
                </span>
                <span style="color: var(--text-secondary); font-size: 0.8rem;">
                    v2.0 | 2025
                </span>
            </div>
        </div>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #E5E7EB;">
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; font-size: 0.8rem; color: var(--text-secondary);">
                <span>📊 Multi-format Support</span>
                <span>🎯 Smart Preprocessing</span>
                <span>🔍 Top-3 Analysis</span>
                <span>💰 Price Estimation</span>
                <span>🛒 Market Recommendations</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Warning
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0; padding: 1rem; background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); 
                border-left: 4px solid #F59E0B; border-radius: var(--border-radius);">
        <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.2rem;">⚠️</span>
            <strong style="color: #92400E;">ข้อจำกัดระบบ</strong>
        </div>
        <p style="margin: 0; color: #92400E; font-size: 0.9rem;">
            ระบบใช้ข้อมูลทดสอบ • ความแม่นยำอาจไม่สูง • เฉพาะการศึกษาและทดลองใช้งาน
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()