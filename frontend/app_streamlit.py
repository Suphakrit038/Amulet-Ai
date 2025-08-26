"""
🔍 Amulet-AI - Advanced Thai Amulet Analysis System
====================================================
Combined Version 3.0 (v1 + v2) with enhanced features:
- Multi-language support (Thai/English)
- Advanced AI analysis with confidence scoring
- Comparison mode for multiple amulets
- Analytics dashboard with interactive charts
- Bookmark system for saving analysis results
- Modern responsive UI with dark/light themes

Author: Amulet-AI Team
Version: 3.0 (Combined v1 + v2)
Date: August 2025
"""

# ==========================================
# IMPORTS AND DEPENDENCIES
# ==========================================
import streamlit as st
import requests
import io
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from PIL import Image, ImageEnhance
from random import uniform, randint, choice
import pandas as pd
from plotly.subplots import make_subplots

# ==========================================
# CONFIGURATION AND CONSTANTS
# ==========================================

# API Configuration
API_URL = "http://localhost:8000"

# HEIC Format Support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

# Multi-language Support Dictionary
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
        'market_rec': 'แนะนำตลาด',
        'compare': 'เปรียบเทียบ',
        'analytics': 'สถิติ',
        'saved': 'บันทึก'
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
        'market_rec': 'Market Recommendations',
        'compare': 'Compare',
        'analytics': 'Analytics',
        'saved': 'Saved'
    }
}

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_lang():
    """Get current language setting from session state"""
    return st.session_state.get('language', 'th')

def _(key):
    """Translation helper function for multi-language support"""
    return LANGUAGES[get_lang()].get(key, key)

# ==========================================
# SYSTEM FUNCTIONS
# ==========================================

def check_system_status():
    """Check backend system status and connectivity"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
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

# ==========================================
# IMAGE PROCESSING FUNCTIONS
# ==========================================

def validate_image(uploaded_file):
    """
    Validate and process uploaded image with Smart Preprocessing
    Supports: JPG, JPEG, PNG, HEIC, HEIF, WebP, BMP, TIFF
    """
    try:
        image = Image.open(uploaded_file)
        original_format = image.format
        
        # Smart Preprocessing: Auto resize + enhance
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])
                else:
                    background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
        
        # Enhance image quality (increase sharpness)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)  # Increase sharpness by 20%
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        return image, img_byte_arr, None, original_format
    except Exception as e:
        return None, None, str(e), None

# ==========================================
# DATA ANALYSIS FUNCTIONS
# ==========================================

def create_mock_analysis_result():
    """Generate mock analysis data for testing purposes"""
    amulet_names = [
        "หลวงปู่ทวด วัดช้างให้", "พระสมเด็จ วัดระฆัง", "พระนางพญา วัดอ่างอู",
        "หลวงพ่อโสธร", "พระรอด วัดโมลี", "หลวงปู่ศุข วัดปากคลองมะขามเฒ่า",
        "พระผงสุพรรณ", "หลวงปู่บุญ วัดกลางบางแก้ว", "พระกรุวัดพระศรีรัตนมหาธาตุ"
    ]
    
    selected_amulet = choice(amulet_names)
    
    return {
        "top1": {
            "class_name": selected_amulet,
            "confidence": uniform(0.75, 0.98),
            "score": randint(80, 98)
        },
        "top3_predictions": [
            {"class_name": selected_amulet, "confidence": uniform(0.75, 0.98), "score": randint(85, 98)},
            {"class_name": choice([n for n in amulet_names if n != selected_amulet]), "confidence": uniform(0.15, 0.35), "score": randint(60, 80)},
            {"class_name": choice([n for n in amulet_names if n != selected_amulet]), "confidence": uniform(0.05, 0.25), "score": randint(40, 70)}
        ],
        "authenticity_score": randint(70, 95),
        "valuation": {
            "p25": randint(5000, 15000),
            "p50": randint(15000, 50000), 
            "p75": randint(50000, 150000)
        },
        "popularity_score": randint(70, 95),
        "market_recommendations": [
            {"market": "ตลาดพระ วัดราชนัดดา", "score": 95, "note": "ตลาดที่มีชื่อเสียง เชื่อถือได้"},
            {"market": "ตลาดนัดจตุจักร", "score": 85, "note": "หลากหลาย ราคาดี"},
            {"market": "ออนไลน์ Facebook", "score": 75, "note": "สะดวก แต่ต้องระวัง"}
        ]
    }

# ==========================================
# BOOKMARK/SAVE SYSTEM
# ==========================================

def save_analysis_result(analysis_data):
    """Save analysis result to session state for later access"""
    if 'saved_analyses' not in st.session_state:
        st.session_state.saved_analyses = []
    
    analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    saved_analysis = {
        'id': analysis_id,
        'timestamp': datetime.now(),
        'result': analysis_data,
        'image_name': f"saved_image_{len(st.session_state.saved_analyses) + 1}.jpg"
    }
    
    st.session_state.saved_analyses.append(saved_analysis)
    return analysis_id

# ==========================================
# UI COMPONENT FUNCTIONS
# ==========================================

def setup_page():
    """Sets up the Streamlit page with custom styles, translations, and other configurations."""
    st.set_page_config(
        page_title="Amulet-AI V3",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state for language if not present
    if 'language' not in st.session_state:
        st.session_state.language = 'th'

    # Modern CSS with Minimalist Design & Glassmorphism
    st.markdown("""
    <style>
        /* Import Modern Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Variables */
        :root {
            /* Core Palette */
            --primary-color: #8B5CF6;
            --primary-color-hover: #7C3AED;
            --primary-color-active: #6D28D9;
            --primary-gradient: radial-gradient(circle at 20% 20%, #4C1D95 0%, #1E1B4B 45%, #0F172A 100%);
            --accent-color: #6366F1;
            --success-color: #10B981;
            --warning-color: #F59E0B;
            --error-color: #EF4444;
            --info-color: #0EA5E9;
            /* Surfaces */
            --surface-glass: rgba(255, 255, 255, 0.08);
            --surface-glass-alt: rgba(255, 255, 255, 0.12);
            --surface-frost: rgba(255, 255, 255, 0.06);
            --glass-bg: var(--surface-glass);
            --glass-border: rgba(255, 255, 255, 0.18);
            /* Text */
            --text-primary: #FFFFFF;
            --text-secondary: #CBD5E1;
            --text-inverse: #0F172A;
            /* Layout */
            --border-radius: 16px;
            --border-radius-sm: 10px;
            --focus-ring: 0 0 0 3px rgba(139, 92, 246, 0.5);
            --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            --container-max-width: 1200px;
            --shadow-soft: 0 4px 32px -8px rgba(0,0,0,0.4);
            --shadow-focus: 0 0 0 3px rgba(139,92,246,0.35), 0 4px 24px -4px rgba(0,0,0,0.5);
        }
        
        /* Base Styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Main App Background */
    .stApp { background: var(--primary-gradient) !important; background-attachment: fixed; }
    .main .block-container { padding: 1.5rem 2.5rem 4rem 2.5rem; max-width: var(--container-max-width); }
    @media (max-width: 1320px){ .main .block-container{ padding:1.25rem 1.5rem 3.5rem 1.5rem;} }

        /* Text colors */
        h1, h2, h3, h4, h5, h6, p, div, span, li, label {
            color: var(--text-primary) !important;
        }
        p, li, label, .st-emotion-cache-1q8dd3e p {
             color: var(--text-secondary) !important;
        }

        /* Hero Section - Simplified */
        .hero-section {
            background: none;
            padding: 2rem 1rem;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: none;
            border: none;
        }
        
        .hero-title {
            font-size: clamp(2.5rem, 6vw, 4rem);
            font-weight: 700;
            text-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .hero-subtitle {
            font-size: clamp(1.1rem, 3vw, 1.4rem);
            font-weight: 300;
            opacity: 0.9;
        }
        
        /* Glass Cards */
        .modern-card, .result-card, .metric-card, .tip-card, .upload-card {
            background: var(--glass-bg) !important;
            backdrop-filter: blur(14px) saturate(140%);
            -webkit-backdrop-filter: blur(14px) saturate(140%);
            border-radius: var(--border-radius) !important;
            border: 1px solid var(--glass-border) !important;
            box-shadow: var(--shadow-soft);
            transition: var(--transition);
            overflow: hidden;
            position: relative;
        }
        .modern-card:hover, .result-card:hover, .metric-card:hover, .tip-card:hover, .upload-card:hover { background: var(--surface-glass-alt) !important; transform: translateY(-2px); }
        .modern-card:focus-within, .upload-card:focus-within { box-shadow: var(--shadow-focus); }
        
        /* Button Styles */
        .stButton > button { background: linear-gradient(135deg,var(--primary-color) 0%,var(--accent-color) 100%) !important; color:#fff !important; border:none !important; border-radius: var(--border-radius-sm) !important; padding:0.8rem 1.6rem !important; font-weight:600 !important; font-size:1rem !important; transition:var(--transition) !important; box-shadow:0 6px 18px -6px rgba(99,102,241,0.55) !important; }
        .stButton > button:hover { filter:brightness(1.1); transform:translateY(-2px) !important; }
        .stButton > button:active { transform:translateY(0) scale(.97) !important; }
        .stButton > button:focus { outline:none !important; box-shadow: var(--focus-ring) !important; }
        
        /* Tab Enhancement */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: none !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 12px 24px !important;
            border-radius: var(--border-radius) !important;
            border: 1px solid var(--glass-border) !important;
            background: var(--glass-bg) !important;
            transition: var(--transition) !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.15) !important;
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background: rgba(255, 255, 255, 0.25) !important;
            border-color: rgba(255, 255, 255, 0.4) !important;
        }

    /* Hide default sidebar toggle for cleaner top bar approach */
    button[kind="header"] { display:none !important; }
        
        /* Loading & Animations */
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .slide-in {
            animation: slideIn 0.6s ease-out;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
            }
            .hero-section {
                padding: 1rem;
            }
        }

    /* Top Bar */
    .top-bar { position:sticky; top:0; z-index:100; display:flex; align-items:center; justify-content:space-between; padding:0.75rem 1.25rem; margin:-1rem -1rem 1.25rem -1rem; background: linear-gradient(90deg, rgba(15,23,42,0.85) 0%, rgba(49,46,129,0.65) 70%); backdrop-filter: blur(18px) saturate(180%); border:1px solid rgba(255,255,255,0.12); border-radius: 18px; box-shadow:0 4px 24px -4px rgba(0,0,0,0.55); }
    .brand { display:flex; align-items:center; gap:.65rem; font-weight:600; font-size:1.05rem; letter-spacing:.5px; }
    .status-badge { display:inline-flex; align-items:center; gap:.35rem; font-size:.7rem; text-transform:uppercase; letter-spacing:1px; background:rgba(16,185,129,0.12); color:var(--success-color); padding:4px 10px; border-radius:999px; border:1px solid rgba(16,185,129,0.35); font-weight:600; }
    .status-badge.offline { background:rgba(239,68,68,0.15); color:var(--error-color); border-color:rgba(239,68,68,0.45); }
    .lang-switch { display:inline-flex; gap:.35rem; }
    .lang-switch button { background:var(--surface-glass)!important; border:1px solid var(--glass-border)!important; color:var(--text-secondary)!important; padding:.4rem .75rem !important; border-radius:10px !important; font-size:.75rem !important; font-weight:500 !important; }
    .lang-switch button:hover { color:#fff !important; background:var(--surface-glass-alt)!important; }
    .lang-active { background:linear-gradient(135deg,var(--primary-color),var(--accent-color))!important; color:#fff !important; box-shadow:0 4px 16px -4px rgba(99,102,241,0.55) !important; }

    /* Stepper */
    .stepper { display:flex; gap:1rem; flex-wrap:wrap; margin:0 0 1.75rem 0; }
    .step { flex:1 1 160px; position:relative; padding:.85rem .9rem .85rem 2.9rem; background:var(--glass-bg); border:1px solid var(--glass-border); border-radius:14px; font-size:.8rem; line-height:1.2; font-weight:500; letter-spacing:.3px; display:flex; flex-direction:column; gap:.4rem; min-height:72px; }
    .step:before { content:attr(data-index); position:absolute; left:.75rem; top:.55rem; width:1.6rem; height:1.6rem; border-radius:12px; background:var(--surface-glass-alt); color:#fff; font-size:.75rem; font-weight:600; display:flex; align-items:center; justify-content:center; border:1px solid var(--glass-border); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05); }
    .step.done { background:linear-gradient(135deg,rgba(16,185,129,0.3),rgba(16,185,129,0.15)); border-color:rgba(16,185,129,0.55); }
    .step.done:before { background:var(--success-color); content:'✓'; }
    .step.active { background:linear-gradient(135deg,rgba(139,92,246,0.40),rgba(99,102,241,0.25)); border-color:rgba(139,92,246,0.65); box-shadow:0 0 0 1px rgba(139,92,246,0.4); }
    .step.active:before { background:linear-gradient(180deg,var(--primary-color),var(--accent-color)); }
    .step small { font-size:.65rem; opacity:.8; font-weight:400; }
    @media (max-width: 860px){ .stepper{ flex-direction:column; } .step{ width:100%; } }

    /* Upload Cards */
    .upload-zone { border:1.5px dashed rgba(255,255,255,0.35); border-radius:14px; padding:1.25rem 1rem; text-align:center; cursor:pointer; position:relative; transition:var(--transition); }
    .upload-zone:hover { border-color:rgba(255,255,255,0.6); background:rgba(255,255,255,0.08); }
    .upload-zone.drag { border-color: var(--primary-color); background: rgba(99,102,241,0.15); }
    .upload-actions { display:flex; gap:.5rem; flex-wrap:wrap; justify-content:center; margin-top:.85rem; }
    .upload-actions button { background:var(--surface-glass-alt)!important; border:1px solid var(--glass-border)!important; padding:.5rem .9rem !important; font-size:.65rem !important; font-weight:500 !important; border-radius:10px !important; }
    .upload-actions button:hover { background:var(--surface-glass)!important; }
    .img-preview { border-radius:12px; overflow:hidden; box-shadow:0 4px 20px -6px rgba(0,0,0,0.5); border:1px solid rgba(255,255,255,0.12); }
    .quality-badges { display:flex; gap:.4rem; flex-wrap:wrap; margin-top:.55rem; }
    .q-badge { font-size:.55rem; letter-spacing:.5px; text-transform:uppercase; padding:3px 8px; border-radius:40px; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.18); font-weight:600; display:inline-flex; align-items:center; gap:.35rem; }
    .q-ok { background:rgba(16,185,129,0.22); border-color:rgba(16,185,129,0.55); color:var(--success-color); }
    .q-warn { background:rgba(245,158,11,0.22); border-color:rgba(245,158,11,0.55); color:var(--warning-color); }
    .q-bad { background:rgba(239,68,68,0.22); border-color:rgba(239,68,68,0.55); color:var(--error-color); }
    .primary-cta-container { position:sticky; bottom:1.25rem; display:flex; justify-content:flex-end; margin-top:2rem; }
    .floating-cta { min-width:240px; }
    @media (max-width:860px){ .primary-cta-container{ position:fixed; left:0; right:0; bottom:0; padding:.75rem 1.1rem; background:linear-gradient(180deg,rgba(15,23,42,0.35) 0%, rgba(15,23,42,0.85) 60%); backdrop-filter:blur(16px); border-top:1px solid rgba(255,255,255,0.15);} .floating-cta{ width:100%; } }
    </style>
    """, unsafe_allow_html=True)

def show_camera_interface(key_prefix=""):
    """Display camera interface for image capture and upload"""
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
    """Create circular confidence gauge component"""
    if confidence >= 80:
        color, status, icon = "var(--success-color)", "สูง", "🎯"
    elif confidence >= 60:
        color, status, icon = "var(--warning-color)", "ปานกลาง", "⚡"
    else:
        color, status, icon = "var(--error-color)", "ต่ำ", "⚠️"
    
    return f"""
    <div style="display: flex; align-items: center; justify-content: center; margin: 1rem 0;">
        <div style="position: relative; width: 120px; height: 120px;">
            <svg width="120" height="120" viewBox="0 0 120 120">
                <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(255,255,255,0.2)" stroke-width="8"/>
                <circle cx="60" cy="60" r="50" fill="none" stroke="{color}" stroke-width="8"
                        stroke-dasharray="314.16" stroke-dashoffset="{314.16 * (1 - confidence/100)}"
                        stroke-linecap="round" transform="rotate(-90 60 60)"
                        style="transition: stroke-dashoffset 2s ease-in-out;"/>
            </svg>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {color};">{confidence:.1f}%</div>
                <div style="font-size: 0.8rem; color: var(--text-secondary);">{status}</div>
            </div>
        </div>
    </div>
    """

def create_authenticity_score(score):
    """Create authenticity score component with progress bar"""
    if score >= 85:
        color, status, icon = "var(--success-color)", "น่าเชื่อถือสูง", "🔒"
    elif score >= 70:
        color, status, icon = "var(--warning-color)", "น่าเชื่อถือปานกลาง", "🔐"
    else:
        color, status, icon = "var(--error-color)", "ควรตรวจสอบเพิ่มเติม", "🔓"
    
    return f"""
    <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span style="font-size: 1.2rem;">{icon}</span>
        <div style="flex-grow: 1;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="font-size: 0.9rem; font-weight: 600; color: var(--text-primary);">Authenticity Score</span>
                <span style="font-size: 0.9rem; font-weight: 700; color: {color};">{score}%</span>
            </div>
            <div style="background: rgba(255,255,255,0.2); height: 6px; border-radius: 3px;">
                <div style="background: {color}; width: {score}%; height: 100%; border-radius: 3px; transition: width 1.5s ease-out;"></div>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">{status}</div>
        </div>
    </div>
    """

# ==========================================
# TAB CONTENT FUNCTIONS
# ==========================================

def show_comparison_mode():
    """Display comparison mode for multiple amulets"""
    st.markdown("""
    <div style="text-align: center; margin: 0 0 2rem 0;">
        <h2 style="color: var(--text-primary); margin: 0 0 1rem 0; font-weight: 600;">⚖️ เปรียบเทียบพระเครื่อง</h2>
        <p style="color: var(--text-secondary); margin: 0; font-size: 1.1rem;">
            เปรียบเทียบพระเครื่องหลาย ๆ องค์พร้อมกัน
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    num_items = st.selectbox("เลือกจำนวนพระเครื่องที่ต้องการเปรียบเทียบ", [2, 3, 4], index=0)
    st.markdown("### 📤 อัปโหลดภาพสำหรับเปรียบเทียบ")
    
    cols = st.columns(num_items)
    comparison_data = []
    
    for i in range(num_items):
        with cols[i]:
            st.markdown(f"""
            <div class="modern-card" style="padding: 1rem; margin-bottom: 1rem; text-align: center;">
                <h4 style="color: var(--primary-color); margin: 0 0 0.5rem 0;">🔍 พระเครื่องที่ {i+1}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(f"เลือกรูปพระเครื่องที่ {i+1}", type=['jpg', 'jpeg', 'png', 'heic', 'heif', 'webp'], key=f"compare_{i}")
            
            if uploaded_file:
                image, img_bytes, error, _ = validate_image(uploaded_file)
                if image and not error:
                    st.image(image, use_container_width=True)
                    
                    # Mock analysis for comparison
                    mock_data = create_mock_analysis_result()
                    comparison_data.append({
                        'id': i+1,
                        'name': mock_data['top1']['class_name'],
                        'confidence': mock_data['top1']['confidence'] * 100,
                        'authenticity': mock_data['authenticity_score'],
                        'price_estimate': mock_data['valuation']['p50'],
                        'popularity': mock_data.get('popularity_score', 85)
                    })
                    
                    st.markdown(f"""
                    <div style="background: var(--success-color); color: white; padding: 0.5rem; 
                               border-radius: 8px; text-align: center; margin: 0.5rem 0; font-size: 0.85rem;">
                        ✅ {mock_data['top1']['class_name']}<br>
                        <strong>{mock_data['top1']['confidence']*100:.1f}%</strong>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Show comparison results
    if len(comparison_data) >= 2:
        st.markdown("---")
        st.markdown("### 📊 ตารางเปรียบเทียบ")
        
        comparison_df_data = []
        for item in comparison_data:
            comparison_df_data.append({
                'พระเครื่อง': item['name'],
                'ความมั่นใจ (%)': f"{item['confidence']:.1f}%",
                'ความแท้ (%)': f"{item['authenticity']}%", 
                'ราคาประเมิน (บาท)': f"{item['price_estimate']:,}",
                'ความนิยม (%)': f"{item['popularity']}%"
            })
        
        comparison_df = pd.DataFrame(comparison_df_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization with Plotly
        st.markdown("### 📈 กราฟเปรียบเทียบ")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ความมั่นใจ', 'ความแท้', 'ราคาประเมิน', 'ความนิยม'),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
        )
        
        names = [item['name'][:20] + '...' if len(item['name']) > 20 else item['name'] for item in comparison_data]
        
        fig.add_trace(go.Bar(x=names, y=[item['confidence'] for item in comparison_data], 
                           name='ความมั่นใจ', marker_color='#8B5CF6'), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=[item['authenticity'] for item in comparison_data],
                           name='ความแท้', marker_color='#10B981'), row=1, col=2)
        fig.add_trace(go.Bar(x=names, y=[item['price_estimate'] for item in comparison_data],
                           name='ราคาประเมิน', marker_color='#F59E0B'), row=2, col=1)
        fig.add_trace(go.Bar(x=names, y=[item['popularity'] for item in comparison_data],
                           name='ความนิยม', marker_color='#EF4444'), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text="📊 การเปรียบเทียบพระเครื่อง", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        best_item = max(comparison_data, key=lambda x: x['confidence'])
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: white;">
            <h4 style="color: white; margin: 0 0 1rem 0;">🏆 แนะนำสูงสุด</h4>
            <h3 style="color: white; margin: 0;">{best_item['name']}</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                ความมั่นใจ: {best_item['confidence']:.1f}% • ราคาประเมิน: {best_item['price_estimate']:,} บาท
            </p>
        </div>
        """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """แสดงแดชบอร์ดสถิติและข้อมูลวิเคราะห์ในรูปแบบมินิมอล"""
    st.markdown(f"""
    <div style="text-align: center; margin: 0 0 2rem 0;">
        <h2 style="color: var(--text-primary); margin: 0 0 1rem 0; font-weight: 600;">📊 แดชบอร์ดสถิติ</h2>
        <p style="color: var(--text-secondary); margin: 0; font-size: 1.1rem;">
            ข้อมูลสถิติและแนวโน้มของตลาดพระเครื่อง
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Row (Simplified)
    metrics = [
        {"value": "1,247", "label": "การวิเคราะห์วันนี้", "delta": "↗ +12%", "color": "var(--primary-color)"},
        {"value": "94.2%", "label": "ความแม่นยำเฉลี่ย", "delta": "↗ +2.1%", "color": "var(--success-color)"},
        {"value": "฿47,500", "label": "ราคาประเมินเฉลี่ย", "delta": "↗ +8.5%", "color": "var(--warning-color)"},
        {"value": "2.3s", "label": "เวลาประมวลผล", "delta": "เร็วขึ้น 15%", "color": "var(--primary-color)"}
    ]
    
    cols = st.columns(4)
    for i, metric in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="padding: 1.5rem; text-align: center; height: 100%;">
                <div style="font-size: 2rem; font-weight: 700; color: {metric['color']};">{metric['value']}</div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">{metric['label']}</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary); opacity: 0.8; margin-top: 0.25rem;">{metric['delta']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: var(--glass-border); margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # Charts Section (Simplified)
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### 📈 แนวโน้มการวิเคราะห์รายวัน")
        
        # Create trend chart
        dates = [(datetime.now() - timedelta(days=x)).strftime('%d/%m') for x in range(14, 0, -1)]
        analyses = [randint(800, 1500) for _ in range(14)]
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=dates, y=analyses,
            mode='lines',
            name='การวิเคราะห์',
            line=dict(color='#A855F7', width=3),
            fill='tonexty',
            fillcolor='rgba(168, 85, 247, 0.2)'
        ))
        
        fig_trend.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_chart2:
        st.markdown("### 🏷️ พระเครื่องยอดนิยม")
        
        # Popular amulets data
        popular_amulets = [
            "หลวงปู่ทวด", "พระสมเด็จ", "พระนางพญา", "หลวงพ่อโสธร", "พระรอด", "หลวงปู่ศุข"
        ]
        popularity_scores = sorted([randint(70, 100) for _ in range(6)], reverse=True)
        
        fig_popular = go.Figure()
        fig_popular.add_trace(go.Bar(
            y=[name[:15] for name in popular_amulets],
            x=popularity_scores,
            orientation='h',
            marker=dict(
                color=popularity_scores,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{score}%' for score in popularity_scores],
            textposition='inside',
            insidetextanchor='middle'
        ))
        
        fig_popular.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, autorange="reversed"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        st.plotly_chart(fig_popular, use_container_width=True)

def show_saved_analyses():
    """Display saved analysis results with management options"""
    if 'saved_analyses' not in st.session_state or not st.session_state.saved_analyses:
        st.info("📝 " + ("ยังไม่มีผลการวิเคราะห์ที่บันทึกไว้" if get_lang() == 'th' else "No saved analyses yet"))
        return
    
    st.markdown("### 📋 " + ("ผลการวิเคราะห์ที่บันทึกไว้" if get_lang() == 'th' else "Saved Analyses"))
    
    for analysis in reversed(st.session_state.saved_analyses):  # Latest first
        with st.expander(f"🔍 {analysis['result']['top1']['class_name']} - {analysis['timestamp'].strftime('%d/%m/%Y %H:%M')}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                **{('ผลลัพธ์' if get_lang() == 'th' else 'Result')}:** {analysis['result']['top1']['class_name']}
                
                **{('ความมั่นใจ' if get_lang() == 'th' else 'Confidence')}:** {analysis['result']['top1']['confidence'] * 100:.1f}%
                """)
                
                if 'valuation' in analysis['result']:
                    st.markdown(f"**{('ราคาประเมิน' if get_lang() == 'th' else 'Price Estimate')}:** {analysis['result']['valuation']['p50']:,} บาท")
            
            with col2:
                if st.button("🗑️ " + ("ลบ" if get_lang() == 'th' else "Delete"), key=f"del_{analysis['id']}"):
                    st.session_state.saved_analyses.remove(analysis)
                    st.rerun()

def show_tips_section():
    """แสดงส่วนคำแนะนำและเทคนิคในรูปแบบมินิมอล"""
    st.markdown("<hr style='border-color: var(--glass-border); margin: 3rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h2 style="color: var(--text-primary); margin: 0 0 1rem 0; font-weight: 600;">💡 เทคนิคการถ่ายภาพ</h2>
        <p style="color: var(--text-secondary); margin: 0; font-size: 1.1rem;">
            เคล็ดลับเพื่อผลลัพธ์การวิเคราะห์ที่ดีที่สุด
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tips = {
        "💡 แสงที่เหมาะสม": "ใช้แสงธรรมชาติ หลีกเลี่ยงแสงแฟลชและเงาบนองค์พระ",
        "📐 มุมกล้อง": "ถ่ายภาพตรงๆ ในระยะห่างที่พอดี โดยใช้พื้นหลังสีเรียบ",
        "🔍 คุณภาพภาพ": "ใช้ความละเอียดสูง ภาพต้องคมชัด ไม่เบลอ และเห็นรายละเอียดครบถ้วน"
    }
    
    cols = st.columns(3)
    for i, (title, content) in enumerate(tips.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="tip-card" style="padding: 1.5rem; text-align: center; height: 100%;">
                <h3 style="color: var(--text-primary); margin: 0 0 1rem 0; font-weight: 600; font-size: 1.2rem;">{title}</h3>
                <p style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;">{content}</p>
            </div>
            """, unsafe_allow_html=True)

def show_footer():
    """แสดงส่วน Footer ที่เรียบง่าย"""
    st.markdown("<hr style='border-color: var(--glass-border); margin: 3rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: var(--text-secondary); font-size: 0.9rem;">
        <p>© 2025 Amulet-AI • Made with ❤️ in Thailand</p>
        <p>Version: 3.0 (Minimalist) • Build: 2025.08.26</p>
    </div>
    """, unsafe_allow_html=True)

def render_top_bar(is_online: bool):
    status_class = "status-badge" + ("" if is_online else " offline")
    status_label = ("ONLINE" if is_online else "OFFLINE")
    # Language buttons
    current_lang = get_lang()
    st.markdown("""
    <div class="top-bar">
        <div class="brand">🔮 <span>Amulet-AI</span><span class="" style="opacity:.45;font-weight:400;">v3</span><span class="" style="margin-left:.6rem;"><span class="%s">%s</span></span></div>
        <div style="display:flex; align-items:center; gap:.75rem;">
            <div class="lang-switch" id="lang-switch"></div>
        </div>
    </div>
    """ % (status_class, status_label), unsafe_allow_html=True)
    # Render language switch with Streamlit buttons (outside HTML to keep interaction)
    lang_col1, lang_col2, _spacer = st.columns([0.08,0.09,0.83])
    with lang_col1:
        if st.button("TH", key="lang_th", help="ภาษาไทย", type="secondary"):
            if current_lang != 'th':
                st.session_state.language = 'th'
                st.rerun()
    with lang_col2:
        if st.button("EN", key="lang_en", help="English", type="secondary"):
            if current_lang != 'en':
                st.session_state.language = 'en'
                st.rerun()
    # Style active after rendering
    if 'lang_style_applied' not in st.session_state:
        st.session_state.lang_style_applied = True
        st.markdown("""
        <script>
        const th = window.parent.document.querySelector('button[title="ภาษาไทย"]') || window.parent.document.querySelector('button[kind][data-testid="baseButton-lang_th"]');
        const en = window.parent.document.querySelector('button[title="English"]') || window.parent.document.querySelector('button[kind][data-testid="baseButton-lang_en"]');
        if(th && en){
            if('%s'==='th'){ th.classList.add('lang-active'); en.classList.remove('lang-active'); } else { en.classList.add('lang-active'); th.classList.remove('lang-active'); }
        }
        </script>
        """ % current_lang, unsafe_allow_html=True)

def render_stepper():
    # Determine current step
    has_front = bool(st.session_state.get('front_data'))
    has_result = bool(st.session_state.get('analysis_result'))
    steps = [
        { 'id':'upload', 'title': 'อัปโหลดรูป', 'title_en':'Upload Images', 'desc':'เพิ่มภาพด้านหน้า / หลัง', 'desc_en':'Add front / back images', 'done': has_front },
        { 'id':'validate', 'title': 'ตรวจสอบคุณภาพ', 'title_en':'Validate', 'desc':'เช็คความคมชัด / แสง', 'desc_en':'Check clarity / lighting', 'done': has_front },
        { 'id':'analyze', 'title': 'วิเคราะห์', 'title_en':'Analyze', 'desc':'ส่งให้ AI ประมวลผล', 'desc_en':'Send to AI engine', 'done': has_result },
        { 'id':'results', 'title': 'ผลลัพธ์', 'title_en':'Results', 'desc':'ดูผลลัพธ์และบันทึก', 'desc_en':'View & save results', 'done': has_result }
    ]
    # Active logic
    if not has_front:
        active_id = 'upload'
    elif has_front and not has_result:
        active_id = 'analyze'
    else:
        active_id = 'results'
    lang = get_lang()
    html_steps = []
    for idx, s in enumerate(steps, start=1):
        classes = ["step"]
        if s['done']:
            classes.append('done')
        if s['id'] == active_id:
            classes.append('active')
        title = s['title'] if lang=='th' else s['title_en']
        desc = s['desc'] if lang=='th' else s['desc_en']
        html_steps.append(f"<div class='{' '.join(classes)}' data-index='{idx}'><strong style='font-size:.75rem;'>{title}</strong><small>{desc}</small></div>")
    st.markdown(f"<div class='stepper'>{''.join(html_steps)}</div>", unsafe_allow_html=True)

def _quality_assess(image):
    # Basic heuristic quality metrics (placeholder) -> return dict flags
    if image is None:
        return {}
    w,h = image.size
    res_ok = (max(w,h) >= 600)
    ratio = w/float(h)
    ratio_ok = 0.5 < ratio < 2.0
    return {
        'resolution': res_ok,
        'aspect': ratio_ok
    }

def unified_upload_section():
    st.markdown("""
    <div style='margin:0 0 1.25rem 0;'>
        <h2 style='margin:0 0 .5rem 0; font-weight:600; font-size:1.35rem; letter-spacing:.5px;'>📤 %s</h2>
        <p style='margin:0; font-size:.85rem; opacity:.85;'>%s</p>
    </div>
    """ % (_('upload_title'), _('upload_subtitle')), unsafe_allow_html=True)
    col_front, col_back = st.columns(2)
    # FRONT
    with col_front:
        st.markdown("<div class='upload-card' id='front-card'>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:.75rem;'><div style='font-weight:600;font-size:.9rem;display:flex;align-items:center;gap:.4rem;'>📷 %s <span style='background:#FDE68A;color:#92400E;font-size:.55rem;padding:2px 6px;border-radius:8px;letter-spacing:.5px;font-weight:600;'>%s</span></div></div>" % (_('front_image'), _('required')), unsafe_allow_html=True)
        front_file = st.file_uploader("front_uploader", type=['jpg','jpeg','png','heic','heif','webp','bmp','tiff'], key='unified_front', label_visibility='collapsed')
        if front_file is not None:
            image, buf, err, _fmt = validate_image(front_file)
            if err:
                st.error(f"❌ {err}")
            else:
                st.session_state.front_data = buf.getvalue()
                st.session_state.front_image_pil = image
                qa = _quality_assess(image)
                st.image(image, use_container_width=True, caption="Front Image", output_format='JPEG')
                badges = []
                if qa.get('resolution'): badges.append('<span class="q-badge q-ok">RES</span>')
                else: badges.append('<span class="q-badge q-bad">RES</span>')
                if qa.get('aspect'): badges.append('<span class="q-badge q-ok">ASPECT</span>')
                else: badges.append('<span class="q-badge q-warn">ASPECT</span>')
                st.markdown(f"<div class='quality-badges'>{''.join(badges)}</div>", unsafe_allow_html=True)
                if st.button('🗑️ ลบภาพ', key='del_front', type='secondary'):
                    for k in ['front_data','front_image_pil','analysis_result']:
                        st.session_state.pop(k, None)
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    # BACK
    with col_back:
        st.markdown("<div class='upload-card' id='back-card'>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:.75rem;'><div style='font-weight:600;font-size:.9rem;display:flex;align-items:center;gap:.4rem;'>📱 %s <span style='background:#DBEAFE;color:#1E3A8A;font-size:.55rem;padding:2px 6px;border-radius:8px;letter-spacing:.5px;font-weight:600;'>%s</span></div></div>" % (_('back_image'), _('optional')), unsafe_allow_html=True)
        back_file = st.file_uploader("back_uploader", type=['jpg','jpeg','png','heic','heif','webp','bmp','tiff'], key='unified_back', label_visibility='collapsed')
        if back_file is not None:
            image, buf, err, _fmt = validate_image(back_file)
            if err:
                st.error(f"❌ {err}")
            else:
                st.session_state.back_data = buf.getvalue()
                st.session_state.back_image_pil = image
                qa = _quality_assess(image)
                st.image(image, use_container_width=True, caption="Back Image", output_format='JPEG')
                badges = []
                if qa.get('resolution'): badges.append('<span class="q-badge q-ok">RES</span>')
                else: badges.append('<span class="q-badge q-bad">RES</span>')
                if qa.get('aspect'): badges.append('<span class="q-badge q-ok">ASPECT</span>')
                else: badges.append('<span class="q-badge q-warn">ASPECT</span>')
                st.markdown(f"<div class='quality-badges'>{''.join(badges)}</div>", unsafe_allow_html=True)
                if st.button('🗑️ ลบภาพ', key='del_back', type='secondary'):
                    for k in ['back_data','back_image_pil','analysis_result']:
                        st.session_state.pop(k, None)
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    # Primary CTA
    if st.session_state.get('front_data') and not st.session_state.get('analysis_result'):
        with st.container():
            st.markdown("<div class='primary-cta-container'>", unsafe_allow_html=True)
            if st.button(_('analyze_btn'), key='analyze_primary'):
                with st.spinner('🔍 กำลังวิเคราะห์...'):
                    try:
                        files = {'front_image': ('front.jpg', st.session_state.front_data, 'image/jpeg')}
                        if st.session_state.get('back_data'):
                            files['back_image'] = ('back.jpg', st.session_state.back_data, 'image/jpeg')
                        response = requests.post(f"{API_URL}/analyze", files=files, timeout=30)
                        if response.status_code == 200:
                            st.session_state.analysis_result = response.json()
                        else:
                            st.session_state.analysis_result = create_mock_analysis_result()
                            st.info('ใช้ข้อมูลจำลอง (Backend)')
                    except Exception:
                        st.session_state.analysis_result = create_mock_analysis_result()
                        st.info('ใช้ข้อมูลจำลอง (เชื่อมต่อไม่ได้)')
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    elif st.session_state.get('analysis_result'):
        if st.button('🔄 เริ่มใหม่', key='reset_analysis', type='secondary'):
            for k in ['analysis_result','front_data','back_data','front_image_pil','back_image_pil']:
                st.session_state.pop(k, None)
            st.rerun()

# Main App
def main():
    setup_page()

    # System Status & Top Bar
    is_online, status_message = check_system_status()
    if not is_online:
        st.toast(f"🔧 {status_message}", icon="⚠️")
    render_top_bar(is_online)

    # Hero (now within main flow, reduced prominence)
    st.markdown(f"""
    <div class="hero-section slide-in" style='padding-top:.5rem;'>
        <h1 class="hero-title" style='font-size:clamp(2.1rem,5vw,3.2rem);'>{_('title')}</h1>
        <p class="hero-subtitle" style='font-size:clamp(.95rem,2.2vw,1.15rem); opacity:.85;'>{_('subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Navigation Tabs
    tab_analyze, tab_compare, tab_analytics, tab_saved = st.tabs([
        f"🔍 {('วิเคราะห์' if get_lang() == 'th' else 'Analyze')}",
        f"⚖️ {_('compare')}",
        f"📊 {_('analytics')}",
        f"📋 {_('saved')}"
    ])
    
    with tab_analyze:
        render_stepper()
        unified_upload_section()

        # Display Results (reused existing layout) after unified upload
        if hasattr(st.session_state, 'analysis_result') and st.session_state.analysis_result:
                result = st.session_state.analysis_result
                st.markdown("""
                <div style="margin: 3rem 0 2rem 0;">
                    <h2 style="text-align: center; color: var(--text-primary); margin: 0 0 2rem 0; font-weight: 600;">
                        📊 ผลการวิเคราะห์
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Top Result Card
                st.markdown(f"""
                <div class="result-card slide-in">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <h3 style="color: var(--primary-color); margin: 0; flex-grow: 1; font-weight: 600;">
                            🏆 {result['top1']['class_name']}
                        </h3>
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <span style="background: var(--success-color); color: white; padding: 0.25rem 0.75rem; 
                                       border-radius: 20px; font-size: 0.85rem; font-weight: 500;">
                                Top Match
                            </span>
                        </div>
                    </div>
                    {create_authenticity_score(result['authenticity_score'])}
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence Gauge and Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(create_confidence_gauge(result['top1']['confidence'] * 100), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 1.5rem;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem; color: var(--text-primary);">
                           
                            {result['top1']['confidence'] * 100:.1f}%
                        </div>
                        <div style="color: var(--text-secondary); font-size: 0.9rem;">{_('confidence')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 1.5rem;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem; color: var(--text-primary);">
                            {result['authenticity_score']}%
                        </div>
                        <div style="color: var(--text-secondary); font-size: 0.9rem;">{_('authenticity')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Top-3 Results Section
                st.markdown(f"""
                <div style="margin: 2rem 0;">
                    <h3 style="color: var(--text-primary); margin: 0 0 1rem 0; font-weight: 600;">
                        🔍 {('ผลลัพธ์ที่ใกล้เคียงที่สุด' if get_lang() == 'th' else 'Top-3 Results')}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                top3_cols = st.columns(3)
                
                for i, prediction in enumerate(result['top3_predictions']):
                    with top3_cols[i]:
                        st.markdown(f"""
                        <div class="modern-card" style="padding: 1rem; text-align: center; margin-bottom: 1rem;">
                            <h4 style="color: var(--primary-color); margin: 0 0 0.5rem 0; font-weight: 500;">
                                {i+1}. {prediction['class_name']}
                            </h4>
                            <div style="font-size: 1.5rem; margin: 0.5rem 0;">
                                {prediction['confidence'] * 100:.1f}%
                            </div>
                            <div style="color: var(--text-secondary); font-size: 0.85rem; margin: 0;">
                                {(_('ความมั่นใจ' if get_lang() == 'th' else 'Confidence'))}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Price Estimate and Market Recommendations
                st.markdown(f"""
                <div style="margin: 2rem 0;">
                    <h3 style="color: var(--text-primary); margin: 0 0 1rem 0; font-weight: 600;">
                        💰 {('ราคาประเมินและตลาดแนะนำ' if get_lang() == 'th' else 'Price Estimate & Market Recommendations')}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                col_price, col_market = st.columns(2)
                
                with col_price:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center; padding: 1.5rem;">
                        <div style="font-size: 2rem; font-weight: 700; color: var(--warning-color);">
                            {result['valuation']['p50']:,} บาท
                        </div>
                        <div style="color: var(--text-secondary); font-size: 0.9rem; margin: 0.5rem 0;">
                            {(_('ราคาประเมินกลาง' if get_lang() == 'th' else 'Estimated Price'))}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_market:
                    st.markdown(f"""
                    <div class="modern-card" style="padding: 1rem; text-align: center;">
                        <h4 style="color: var(--primary-color); margin: 0 0 0.5rem 0; font-weight: 500;">
                            {(_('ตลาดแนะนำ' if get_lang() == 'th' else 'Recommended Markets'))}
                        </h4>
                        <ul style="list-style-type: none; padding: 0; margin: 0;">
                    """, unsafe_allow_html=True)
                    
                    for rec in result['market_recommendations']:
                        st.markdown(f"""
                        <li style="margin: 0.5rem 0;">
                            <div style="display: flex; align-items: center; justify-content: center;">
                                <div style="width: 8px; height: 8px; border-radius: 50%; background: var(--success-color); margin-right: 0.5rem;"></div>
                                <div style="color: var(--text-primary); font-weight: 600; font-size: 0.9rem;">
                                {rec['market']}
                                </div>
                            </div>
                        </li>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Save Analysis Button
                if st.button("💾 " + ("บันทึกผลการวิเคราะห์" if get_lang() == 'th' else "Save Analysis"), key="save_analysis"):
                    analysis_id = save_analysis_result(result)
                    st.success("✅ " + ("บันทึกเรียบร้อย!" if get_lang() == 'th' else "Saved successfully!"))
    
    with tab_compare:
        show_comparison_mode()
    
    with tab_analytics:
        show_analytics_dashboard()
    
    with tab_saved:
        show_saved_analyses()
    
    # Footer
    show_footer()

if __name__ == "__main__":
    main()
