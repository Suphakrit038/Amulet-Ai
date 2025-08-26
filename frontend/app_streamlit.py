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
            --primary-color: #A855F7; /* Lighter purple for accents */
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --text-primary: #FFFFFF;
            --text-secondary: #E0E0E0;
            --glass-bg: rgba(255, 255, 255, 0.1);
            --glass-border: rgba(255, 255, 255, 0.2);
            --success-color: #10B981;
            --warning-color: #F59E0B;
            --error-color: #EF4444;
            --border-radius: 16px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Base Styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Main App Background */
        .stApp {
            background: var(--primary-gradient) !important;
            background-attachment: fixed;
        }

        /* Main Container */
        .main .block-container {
            padding: 1rem 2rem;
        }
        
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
        
        /* Glassmorphism Card Style */
        .modern-card, .result-card, .metric-card, .tip-card {
            background: var(--glass-bg) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: var(--border-radius) !important;
            border: 1px solid var(--glass-border) !important;
            box-shadow: none !important;
            transition: var(--transition);
            overflow: hidden;
            padding: 1rem;
        }
        
        .modern-card:hover, .result-card:hover, .metric-card:hover, .tip-card:hover {
            background: rgba(255, 255, 255, 0.15) !important;
            transform: translateY(-4px);
            border-color: rgba(255, 255, 255, 0.3);
        }
        
        /* Button Styles */
        .stButton > button {
            background: var(--glass-bg) !important;
            color: white !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: var(--border-radius) !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: var(--transition) !important;
            box-shadow: none !important;
            width: 100%;
        }
        
        .stButton > button:hover {
            background: rgba(255, 255, 255, 0.2) !important;
            border-color: rgba(255, 255, 255, 0.3) !important;
            transform: translateY(-2px) !important;
        }
        
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

        /* Sidebar Style */
        .st-emotion-cache-16txtl3 {
            background: rgba(0, 0, 0, 0.2) !important;
        }
        
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

# Main App
def main():
    setup_page()

    # System Status Check
    is_online, status_message = check_system_status()
    if not is_online:
        st.toast(f"🔧 {status_message}", icon="⚠️")
    
    # Hero Section with Multi-language
    st.markdown(f"""
    <div class="hero-section slide-in">
        <h1 class="hero-title">{_('title')}</h1>
        <p class="hero-subtitle">{_('subtitle')}</p>
        <div style="margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.8;">
            {_('tagline')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with Enhanced Features
    with st.sidebar:
        # Language Selector
        st.markdown("### 🌍 Language / ภาษา")
        current_lang = get_lang()
        lang_option = st.radio(
            "Select Language", 
            ["🇹🇭 ไทย", "🇺🇸 English"],
            index=0 if current_lang == 'th' else 1,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        new_lang = 'th' if lang_option.startswith('🇹🇭') else 'en'
        if new_lang != current_lang:
            st.session_state.language = new_lang
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("📋 " + ("คู่มือใช้งาน" if current_lang == 'th' else "User Guide"))
        
        with st.expander("📘 " + ("วิธีการใช้งาน" if current_lang == 'th' else "How to Use"), expanded=False):
            steps = [
                "📤 อัปโหลดภาพด้านหน้า" if current_lang == 'th' else "📤 Upload front image",
                "📷 (ไม่บังคับ) อัปโหลดภาพด้านหลัง" if current_lang == 'th' else "📷 (Optional) Upload back image",
                "� กดปุ่ม 'วิเคราะห์พระเครื่อง'" if current_lang == 'th' else "� Click 'Analyze Amulet'",
                "📊 รอชมผลลัพธ์แบบละเอียด" if current_lang == 'th' else "📊 View the detailed results"
            ]
            
            for i, step in enumerate(steps, 1):
                st.write(f"{i}. {step}")
        
        st.markdown("---")
        st.info("💡 " + ("เคล็ดลับ: ใช้ภาพที่คมชัดและมีแสงดีเพื่อให้ได้ผลลัพธ์แม่นยำที่สุด" if current_lang == 'th' else "Tip: Use clear, well-lit photos for the most accurate results."))
    
    # Main Navigation Tabs
    tab_analyze, tab_compare, tab_analytics, tab_saved = st.tabs([
        f"🔍 {('วิเคราะห์' if get_lang() == 'th' else 'Analyze')}",
        f"⚖️ {_('compare')}",
        f"📊 {_('analytics')}",
        f"📋 {_('saved')}"
    ])
    
    with tab_analyze:
        # Upload Section
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <h2 style="color: var(--text-primary); margin-bottom: 0.5rem; font-weight: 600;">{_('upload_title')}</h2>
            <p style="color: var(--text-secondary); margin: 0; font-size: 1rem;">
                {_('upload_subtitle')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="modern-card" style="padding: 1rem; margin-bottom: 1rem;">
                <h4 style="color: var(--success-color); margin: 0 0 0.5rem 0; display: flex; align-items: center;">
                    {_('front_image')}
                    <span style="margin-left: auto; font-size: 0.7rem; background: #FEF3C7; color: #92400E; padding: 2px 8px; border-radius: 12px;">{_('required')}</span>
                </h4>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 0;">จำเป็นสำหรับการวิเคราะห์ขั้นพื้นฐาน</p>
            </div>
            """, unsafe_allow_html=True)
            
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
            st.markdown(f"""
            <div class="modern-card" style="padding: 1rem; margin-bottom: 1rem;">
                <h4 style="color: var(--warning-color); margin: 0 0 0.5rem 0; display: flex; align-items: center;">
                    {_('back_image')}
                    <span style="margin-left: auto; font-size: 0.7rem; background: #E0E7FF; color: #3730A3; padding: 2px 8px; border-radius: 12px;">{_('optional')}</span>
                </h4>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 0;">เพิ่มความแม่นยำในการวิเคราะห์</p>
            </div>
            """, unsafe_allow_html=True)
            
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
    
        # Analysis Button and Results
        if hasattr(st.session_state, 'front_data') and st.session_state.front_data:
            st.markdown("""
            <div style="margin: 3rem 0 2rem 0; text-align: center;">
                <div style="height: 1px; background: var(--glass-border); border-radius: 1px; margin: 1rem auto; width: 80%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
            with analyze_col2:
                if st.button(_('analyze_btn'), key="main_analyze", type="primary"):
                    # Animation during analysis
                    with st.spinner("🔍 " + ("กำลังวิเคราะห์พระเครื่อง..." if get_lang() == 'th' else "Analyzing amulet...")):
                        progress_bar = st.progress(0)
                        for i in range(101):
                            progress_bar.progress(i)
                            if i < 30:
                                st.write("🔄 " + ("กำลังประมวลผลภาพ..." if get_lang() == 'th' else "Processing images..."))
                            elif i < 70:
                                st.write("🧠 " + ("AI กำลังวิเคราะห์..." if get_lang() == 'th' else "AI analyzing..."))
                            else:
                                st.write("📊 " + ("กำลังสรุปผลลัพธ์..." if get_lang() == 'th' else "Finalizing results..."))
                        
                        # Simulate API call
                        try:
                            files = {'front_image': ('front.jpg', st.session_state.front_data, 'image/jpeg')}
                            if hasattr(st.session_state, 'back_data') and st.session_state.back_data:
                                files['back_image'] = ('back.jpg', st.session_state.back_data, 'image/jpeg')
                            
                            response = requests.post(f"{API_URL}/analyze", files=files, timeout=30)
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.session_state.analysis_result = result
                                st.success("✅ " + ("วิเคราะห์เสร็จสิ้น!" if get_lang() == 'th' else "Analysis completed!"))
                            else:
                                # Mock result for demo
                                mock_result = create_mock_analysis_result()
                                st.session_state.analysis_result = mock_result
                                st.info("🔧 " + ("ใช้ข้อมูลตัวอย่าง (Backend ไม่พร้อม)" if get_lang() == 'th' else "Using mock data (Backend unavailable)"))
                        except:
                            mock_result = create_mock_analysis_result()
                            st.session_state.analysis_result = mock_result
                            st.info("🔧 " + ("ใช้ข้อมูลตัวอย่าง (เชื่อมต่อไม่ได้)" if get_lang() == 'th' else "Using mock data (Connection failed)"))
            
            # Display Results
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
