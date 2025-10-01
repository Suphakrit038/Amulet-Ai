#!/usr/bin/env python3
"""
Amulet-AI - Production Frontend
ระบบจำแนกพระเครื่องอัจฉริยะ
Thai Amu    .logo-img {
        height: 150px;
        width: auto;
        object-fit: contain;
    }
    
    .logo-img-small {
        height: 110px;
        width: auto;
        object-fit: contain;
    }ication System
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

# Import enhanced modules (with fallback)
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
    'primary': '#800000',
    'maroon': '#800000',
    'accent': '#B8860B',
    'dark_gold': '#B8860B',
    'gold': '#D4AF37',
    'success': '#10b981',
    'green': '#10b981',
    'warning': '#f59e0b',
    'yellow': '#f59e0b',
    'error': '#ef4444',
    'red': '#ef4444',
    'info': '#3b82f6',
    'blue': '#3b82f6',
    'gray': '#6c757d',
    'white': '#ffffff',
    'black': '#000000'
}

# Page Configuration
st.set_page_config(
    page_title="Amulet-AI - ระบบจำแนกพระเครื่อง",
    page_icon="พ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Modal Design CSS
st.markdown(f"""
<style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Modern App Background - Creamy White */
    .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #fdfbf7 0%, #f5f3ef 100%);
        background-attachment: fixed;
    }}
    
    /* Glassmorphism Container */
    .main .block-container {{
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        box-shadow: 0 8px 32px 0 rgba(128, 0, 0, 0.08);
        border: 1px solid rgba(212, 175, 55, 0.2);
        padding: 40px;
        margin: 20px auto;
        max-width: 1400px;
    }}
    
    /* Modal-Style Logo Header */
    .logo-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 35px 60px;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 40px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
    }}
    
    .logo-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['gold']}, {COLORS['primary']});
    }}
    
    .logo-left {{
        display: flex;
        align-items: center;
        gap: 25px;
        z-index: 1;
    }}
    
    .logo-right {{
        display: flex;
        align-items: center;
        gap: 30px;
        z-index: 1;
    }}
    
    .logo-img {{
        height: 150px;
        width: auto;
        object-fit: contain;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));
        transition: transform 0.3s ease;
    }}
    
    .logo-img:hover {{
        transform: scale(1.05);
    }}
    
    .logo-img-small {{
        height: 110px;
        width: auto;
        object-fit: contain;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));
        transition: transform 0.3s ease;
    }}
    
    .logo-img-small:hover {{
        transform: scale(1.05);
    }}
    
    /* Modal Card Style with Glassmorphism */
    .card {{
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        padding: 50px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        margin: 35px 0;
        border: 1px solid rgba(255, 255, 255, 0.5);
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.2);
    }}
    
    .card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['gold']});
    }}
    
    /* Modern Typography */
    h1 {{
        font-size: 3.8rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px !important;
        margin-bottom: 25px !important;
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['gold']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    h2 {{
        font-size: 3rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 22px !important;
        color: #2d3748 !important;
    }}
    
    h3 {{
        font-size: 2.4rem !important;
        font-weight: 600 !important;
        margin-bottom: 18px !important;
        color: #2d3748 !important;
    }}
    
    h4 {{
        font-size: 2rem !important;
        font-weight: 600 !important;
        margin-bottom: 16px !important;
        color: #4a5568 !important;
    }}
    
    /* Modern Button with Gradient and Animation */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 20px 50px;
        font-weight: 600;
        font-size: 1.3rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 6px 20px rgba(128, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(135deg, {COLORS['accent']} 0%, {COLORS['primary']} 100%);
        box-shadow: 0 10px 30px rgba(128, 0, 0, 0.5);
        transform: translateY(-3px) scale(1.02);
    }}
    
    .stButton > button:active {{
        transform: translateY(-1px) scale(0.98);
        box-shadow: 0 4px 15px rgba(128, 0, 0, 0.4);
    }}
    
    /* Modern Text Styling */
    p {{
        font-size: 1.4rem !important;
        line-height: 1.9 !important;
        color: #4a5568 !important;
        font-weight: 400 !important;
    }}
    
    /* Modern Input Fields with Glassmorphism */
    .stTextInput > div > div > input {{
        font-size: 1.3rem !important;
        padding: 18px !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(128, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 0 0 3px rgba(128, 0, 0, 0.1) !important;
    }}
    
    /* Modern File Uploader */
    [data-testid="stFileUploader"] {{
        font-size: 1.3rem !important;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 30px;
        border-radius: 16px;
        border: 2px dashed {COLORS['primary']};
        transition: all 0.3s ease;
    }}
    
    [data-testid="stFileUploader"]:hover {{
        background: rgba(255, 255, 255, 0.95);
        border-color: {COLORS['gold']};
        transform: scale(1.01);
    }}
    
    [data-testid="stFileUploader"] label {{
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: {COLORS['primary']} !important;
    }}
    
    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
        background: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        padding: 18px 40px !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.8) !important;
        border: 2px solid transparent !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(255, 255, 255, 1) !important;
        border-color: {COLORS['primary']} !important;
        transform: translateY(-2px);
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['accent']}) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(128, 0, 0, 0.3);
    }}
    
    /* Modern Alert Boxes with Glassmorphism */
    .stAlert {{
        border-radius: 16px !important;
        padding: 25px !important;
        font-size: 1.3rem !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }}
    
    /* Modal Success Box */
    .success-box {{
        background: linear-gradient(135deg, rgba(232, 245, 233, 0.95), rgba(200, 230, 201, 0.95));
        backdrop-filter: blur(10px);
        color: #1b5e20;
        padding: 30px;
        border-radius: 16px;
        border-left: 5px solid #4caf50;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.2);
        margin: 30px 0;
        font-size: 1.3rem;
        transition: transform 0.3s ease;
    }}
    
    .success-box:hover {{
        transform: translateX(5px);
    }}
    
    /* Modal Info Box */
    .info-box {{
        background: linear-gradient(135deg, rgba(227, 242, 253, 0.95), rgba(187, 222, 251, 0.95));
        backdrop-filter: blur(10px);
        color: #0d47a1;
        padding: 30px;
        border-radius: 16px;
        border-left: 5px solid #2196f3;
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.2);
        margin: 30px 0;
        font-size: 1.3rem;
        transition: transform 0.3s ease;
    }}
    
    .info-box:hover {{
        transform: translateX(5px);
    }}
    
    /* Modal Warning Box */
    .warning-box {{
        background: linear-gradient(135deg, rgba(255, 243, 224, 0.95), rgba(255, 224, 178, 0.95));
        backdrop-filter: blur(10px);
        color: #e65100;
        padding: 30px;
        border-radius: 16px;
        border-left: 5px solid #ff9800;
        box-shadow: 0 8px 25px rgba(255, 152, 0, 0.2);
        margin: 30px 0;
        font-size: 1.3rem;
        transition: transform 0.3s ease;
    }}
    
    .warning-box:hover {{
        transform: translateX(5px);
    }}
    
    /* Modal Error Box */
    .error-box {{
        background: linear-gradient(135deg, rgba(255, 235, 238, 0.95), rgba(255, 205, 210, 0.95));
        backdrop-filter: blur(10px);
        color: #b71c1c;
        padding: 30px;
        border-radius: 16px;
        border-left: 5px solid #f44336;
        box-shadow: 0 8px 25px rgba(244, 67, 54, 0.2);
        margin: 30px 0;
        font-size: 1.3rem;
        transition: transform 0.3s ease;
    }}
    
    .error-box:hover {{
        transform: translateX(5px);
    }}
    
    /* Modern Section Divider */
    .section-divider {{
        height: 2px;
        background: linear-gradient(90deg, transparent, {COLORS['gold']}, transparent);
        margin: 60px 0;
        border-radius: 2px;
    }}
    
    /* Modal Tips Card */
    .tips-card {{
        background: linear-gradient(135deg, rgba(255, 253, 231, 0.95), rgba(255, 249, 196, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 35px;
        margin: 35px 0;
        box-shadow: 0 8px 25px rgba(218, 165, 32, 0.15);
        border-left: 5px solid {COLORS['gold']};
        font-size: 1.3rem;
        transition: transform 0.3s ease;
    }}
    
    .tips-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(218, 165, 32, 0.25);
    }}
    
    /* Feature Card */
    .feature-card {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(250, 250, 250, 0.95));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 40px;
        margin: 25px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.12);
        border: 1px solid rgba(255, 255, 255, 0.5);
        transition: all 0.3s ease;
        height: 100%;
    }}
    
    .feature-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.2);
        border-color: {COLORS['gold']};
    }}
    
    .feature-card h3 {{
        color: {COLORS['primary']} !important;
        margin-bottom: 20px !important;
    }}
    
    .feature-card ul {{
        list-style: none;
        padding-left: 0;
    }}
    
    .feature-card ul li {{
        padding: 12px 0;
        padding-left: 30px;
        position: relative;
        font-size: 1.2rem;
        line-height: 1.8;
    }}
    
    .feature-card ul li:before {{
        content: '✓';
        position: absolute;
        left: 0;
        color: {COLORS['gold']};
        font-weight: bold;
        font-size: 1.4rem;
    }}
    
    /* Step Card */
    .step-card {{
        background: linear-gradient(135deg, rgba(227, 242, 253, 0.95), rgba(187, 222, 251, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 35px;
        margin: 25px 0;
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.15);
        border-left: 5px solid {COLORS['info']};
        transition: transform 0.3s ease;
    }}
    
    .step-card:hover {{
        transform: translateX(8px);
        box-shadow: 0 10px 30px rgba(33, 150, 243, 0.25);
    }}
    
    .step-card h4 {{
        color: {COLORS['info']} !important;
        margin-bottom: 15px !important;
    }}
    
    /* Hero Section */
    .hero-section {{
        text-align: center;
        padding: 60px 40px;
        background: linear-gradient(135deg, rgba(128, 0, 0, 0.05), rgba(212, 175, 55, 0.05));
        border-radius: 24px;
        margin: 40px 0;
    }}
    
    .hero-title {{
        font-size: 4.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 20px !important;
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['gold']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .hero-subtitle {{
        font-size: 1.8rem !important;
        color: {COLORS['gray']} !important;
        margin-bottom: 0 !important;
    }}
    
    /* Modal Result Card */
    .result-card {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(250, 250, 250, 0.98));
        backdrop-filter: blur(15px);
        padding: 50px;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        margin: 35px 0;
        border-top: 5px solid {COLORS['primary']};
        position: relative;
        overflow: hidden;
    }}
    
    .result-card::before {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 200px;
        height: 200px;
        background: linear-gradient(135deg, {COLORS['gold']}, transparent);
        opacity: 0.1;
        border-radius: 50%;
    }}
    
    /* Column Styling */
    [data-testid="column"] {{
        padding: 25px;
    }}
    
    /* Modern Spinner */
    .stSpinner > div {{
        border-color: {COLORS['primary']} {COLORS['gold']} {COLORS['primary']} {COLORS['gold']} !important;
    }}
    
    /* Modern Labels */
    label {{
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
        margin-bottom: 10px !important;
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['gold']}) !important;
    }}
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {{
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['gold']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    [data-testid="stSidebar"] {{display: none;}}
    [data-testid="collapsedControl"] {{display: none;}}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {{
        width: 12px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['gold']});
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, {COLORS['gold']}, {COLORS['primary']});
    }}
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
def classify_image(uploaded_file):
    """จำแนกรูปภาพ"""
    try:
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
        
        classifier = joblib.load(str(project_root / "trained_model/classifier.joblib"))
        scaler = joblib.load(str(project_root / "trained_model/scaler.joblib"))
        label_encoder = joblib.load(str(project_root / "trained_model/label_encoder.joblib"))
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        features = image_normalized.flatten()
        
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
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
            <h3>ผลการจำแนก</h3>
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
                st.success("ความเชื่อมั่นสูงมาก - ผลลัพธ์น่าเชื่อถือ")
            elif confidence >= 0.7:
                st.warning("ความเชื่อมั่นปานกลาง - ควรตรวจสอบเพิ่มเติม")
            else:
                st.error("ความเชื่อมั่นต่ำ - ควรถ่ายรูปใหม่หรือปรึกษาผู้เชี่ยวชาญ")
        
        if show_probabilities and 'probabilities' in result:
            with st.expander("ดูความน่าจะเป็นทั้งหมด"):
                probs = result['probabilities']
                for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**{class_name}**")
                    with col2:
                        st.write(f"{prob:.1%}")
                    st.progress(prob)
        
        st.caption(f"วิธีการทำนาย: {result.get('method', 'Unknown')}")
        
    else:
        error_msg = result.get('error', 'เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ')
        st.markdown(f"""
        <div class="error-box">
            <h3>เกิดข้อผิดพลาด</h3>
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
                <h2 style="margin: 0; color: {COLORS['primary']};">Amulet-AI</h2>
                <p style="margin: 0; color: {COLORS['gray']}; font-size: 0.9rem;">ระบบจำแนกพระเครื่องอัจฉริยะ</p>
            </div>
        </div>
        <div class="logo-right">
            {logo_right_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown(f"""
    <div class="hero-section">
        <h1 class="hero-title">🔮 Amulet-AI</h1>
        <p class="hero-subtitle">ระบบวิเคราะห์วัตถุมงคลด้วย AI</p>
        <p class="hero-subtitle">AI ช่วยวิเคราะห์และจำแนกพระเครื่องแบบง่าย ๆ ให้คุณเข้าใจได้ภายในไม่กี่วินาที</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Introduction Section - 3 Cards
    show_introduction_section()
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # How It Works Section
    show_how_it_works_section()
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Who Made This & Who Is This For
    show_about_section()
    
    # Default settings (no settings UI)
    analysis_mode = "สองด้าน (หน้า+หลัง)"
    show_confidence = True
    show_probabilities = True
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Main Content
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("อัปโหลดรูปพระเครื่อง")
    
    # Always use dual image mode
    dual_image_mode(show_confidence, show_probabilities)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tips Section
    show_tips_section()
    
    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; color: {COLORS['gray']};">
        <p>© 2025 Amulet-AI | พัฒนาโดยความร่วมมือกับ Thai-Austrian และ DEPA</p>
        <p style="font-size: 0.9rem;">ระบบนี้ใช้ AI ช่วยในการจำแนกพระเครื่อง ผลลัพธ์ควรใช้ประกอบการตัดสินใจเท่านั้น</p>
    </div>
    """, unsafe_allow_html=True)

def dual_image_mode(show_confidence, show_probabilities):
    """โหมดสองด้าน"""
    st.markdown("### อัปโหลดรูปทั้งสองด้าน")
    
    col1, col2 = st.columns(2)
    
    # Front image
    with col1:
        st.markdown("#### ด้านหน้า")
        
        front_upload, front_camera = st.tabs(["อัปโหลดไฟล์", "ถ่ายรูป"])
        
        front_image = None
        
        with front_upload:
            front_image = st.file_uploader("เลือกรูปด้านหน้า", type=['jpg', 'jpeg', 'png'], key="front_upload")
        
        with front_camera:
            # Camera will only activate when user enters this tab
            camera_front = st.camera_input("ถ่ายรูปด้านหน้า", key="front_camera")
            if camera_front:
                st.session_state.front_camera_image = camera_front
                st.success("ถ่ายรูปสำเร็จ!")
        
        display_front = front_image or st.session_state.front_camera_image
        if display_front:
            st.image(display_front, caption="รูปด้านหน้า", use_container_width=True)
    
    # Back image
    with col2:
        st.markdown("#### ด้านหลัง")
        
        back_upload, back_camera = st.tabs(["อัปโหลดไฟล์", "ถ่ายรูป"])
        
        back_image = None
        
        with back_upload:
            back_image = st.file_uploader("เลือกรูปด้านหลัง", type=['jpg', 'jpeg', 'png'], key="back_upload")
        
        with back_camera:
            # Camera will only activate when user enters this tab
            camera_back = st.camera_input("ถ่ายรูปด้านหลัง", key="back_camera")
            if camera_back:
                st.session_state.back_camera_image = camera_back
                st.success("ถ่ายรูปสำเร็จ!")
        
        display_back = back_image or st.session_state.back_camera_image
        if display_back:
            st.image(display_back, caption="รูปด้านหลัง", use_container_width=True)
    
    st.markdown("---")
    
    final_front = front_image or st.session_state.front_camera_image
    final_back = back_image or st.session_state.back_camera_image
    
    if final_front and final_back:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.success("มีรูปภาพทั้งสองด้านแล้ว!")
            
            if st.button("เริ่มการวิเคราะห์ทั้งสองด้าน", type="primary", use_container_width=True):
                with st.spinner("AI กำลังวิเคราะห์ทั้งสองด้าน..."):
                    start_time = time.time()
                    
                    front_result = classify_image(final_front)
                    back_result = classify_image(final_back)
                    
                    processing_time = time.time() - start_time
                    
                    st.success(f"เสร็จสิ้น! ({processing_time:.2f}s)")
                    
                    st.markdown("### ผลการวิเคราะห์")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown("#### ด้านหน้า")
                        display_classification_result(front_result, show_confidence, show_probabilities)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown("#### ด้านหลัง")
                        display_classification_result(back_result, show_confidence, show_probabilities)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Comparison
                    if (front_result.get("status") == "success" and back_result.get("status") == "success"):
                        front_class = front_result.get("predicted_class", "")
                        back_class = back_result.get("predicted_class", "")
                        front_conf = front_result.get("confidence", 0)
                        back_conf = back_result.get("confidence", 0)
                        
                        st.markdown("### การเปรียบเทียบ")
                        
                        if front_class == back_class:
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>ผลลัพธ์สอดคล้องกัน!</h4>
                                <p style="font-size: 1.1rem;"><strong>ทั้งสองด้านระบุเป็น:</strong> {front_class}</p>
                                <p><strong>ความเชื่อมั่นเฉลี่ย:</strong> {(front_conf + back_conf) / 2:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="warning-box">
                                <h4>ผลลัพธ์ไม่สอดคล้องกัน</h4>
                                <p><strong>ด้านหน้า:</strong> {front_class} ({front_conf:.1%})</p>
                                <p><strong>ด้านหลัง:</strong> {back_class} ({back_conf:.1%})</p>
                                <p>แนะนำให้ปรึกษาผู้เชี่ยวชาญเพิ่มเติม</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
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
            <h3>คำแนะนำ</h3>
            <p>กรุณาอัปโหลดรูปภาพทั้งสองด้าน (หน้า และ หลัง)</p>
            <p>การวิเคราะห์ทั้งสองด้านจะช่วยเพิ่มความแม่นยำ</p>
        </div>
        """, unsafe_allow_html=True)

def show_introduction_section():
    """แสดงส่วนแนะนำ - เว็บไซต์นี้ทำอะไร"""
    st.markdown("## 📋 เว็บไซต์นี้ทำอะไร")
    st.markdown("<p style='text-align: center; font-size: 1.3rem; color: #6c757d;'>ระบบ Amulet-AI ให้บริการหลากหลายเพื่อช่วยคุณวิเคราะห์พระเครื่อง</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 ฟีเจอร์หลัก</h3>
            <ul>
                <li>จำแนกประเภทพระเครื่องจากรูปภาพ</li>
                <li>รองรับภาพด้านหน้าและด้านหลัง</li>
                <li>บอกความเชื่อมั่นของการทำนาย</li>
                <li>แสดงจุดที่ AI ให้ความสำคัญ</li>
                <li>ดาวน์โหลดรายงานผลลัพธ์</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ ใช้งานง่าย</h3>
            <ul>
                <li>อัปโหลดรูปหรือถ่ายรูปได้ทันที</li>
                <li>ผลลัพธ์ออกภายในไม่กี่วินาที</li>
                <li>แสดงผลแบบกราฟและภาพประกอบ</li>
                <li>ไม่ต้องติดตั้งโปรแกรม</li>
                <li>ใช้งานผ่านเว็บเบราว์เซอร์</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🔒 ปลอดภัย</h3>
            <ul>
                <li>ประมวลผลตามนโยบายความเป็นส่วนตัว</li>
                <li>ข้อมูลเข้ารหัสอย่างปลอดภัย</li>
                <li>สามารถขอลบข้อมูลได้</li>
                <li>ไม่แชร์ข้อมูลโดยไม่ได้รับอนุญาต</li>
                <li>ใช้เทคโนโลยี AI ที่ทันสมัย</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_how_it_works_section():
    """แสดงวิธีการทำงาน 3 ขั้นตอน"""
    st.markdown("## 🔄 ระบบทำงานอย่างไร")
    st.markdown("<p style='text-align: center; font-size: 1.3rem; color: #6c757d;'>เข้าใจง่ายใน 3 ขั้นตอน</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="step-card">
            <h3>📁 ขั้นตอนที่ 1</h3>
            <h4>อัปโหลดรูปภาพ</h4>
            <p>ถ่ายรูปหรือเลือกไฟล์ภาพด้านหน้า/หลังของพระเครื่อง ระบบรองรับไฟล์ JPG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="step-card">
            <h3>🤖 ขั้นตอนที่ 2</h3>
            <h4>AI วิเคราะห์</h4>
            <p>ระบบตรวจสอบภาพ ดึงลักษณะเด่น และทำนายประเภทพร้อมคำนวณความเชื่อมั่น</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="step-card">
            <h3>📊 ขั้นตอนที่ 3</h3>
            <h4>แสดงผลพร้อมคำอธิบาย</h4>
            <p>ผลลัพธ์, กราฟความน่าจะเป็น และคำแนะนำขั้นตอนถัดไป</p>
        </div>
        """, unsafe_allow_html=True)

def show_about_section():
    """แสดงส่วนเกี่ยวกับผู้พัฒนาและกลุ่มเป้าหมาย"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>👥 ใครสร้างระบบนี้</h3>
            <p><strong>พัฒนาโดยทีมผู้เชี่ยวชาญ</strong> ด้าน Machine Learning, Computer Vision และผู้รู้เกี่ยวกับพระเครื่อง</p>
            <p><strong>ทำงานร่วมกับเครือข่าย</strong> ผู้เชี่ยวชาญและชุมชนสะสมพระเพื่อปรับปรุงความแม่นยำ</p>
            <p><strong>จุดมุ่งหมาย:</strong> ทำให้ความรู้ด้านพระเครื่องเข้าถึงได้ง่ายขึ้น และช่วยให้การประเมินเบื้องต้นทำได้รวดเร็ว</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>🎯 เหมาะกับใคร</h3>
            <p><strong>• ผู้เริ่มต้น</strong> - อยากรู้ว่าพระเครื่องที่มีเป็นรุ่นไหนเบื้องต้น</p>
            <p><strong>• นักสะสม/พ่อค้า</strong> - ตรวจสอบและจัดหมวดหมู่เบื้องต้นก่อนซื้อ-ขาย</p>
            <p><strong>• ผู้พัฒนา/นักวิจัย</strong> - สนใจข้อมูลเชิงเทคนิคหรือ dataset (มี API ให้เชื่อมต่อ)</p>
            <p><strong>• ผู้ที่สนใจเทคโนโลยี AI</strong> - เรียนรู้การประยุกต์ใช้ AI ในงานจำแนก</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Expectations & Limitations
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ ควรคาดหวังอะไร (Expectations & Limitations)</h3>
        <p><strong>• ผลลัพธ์เป็นการประเมินเบื้องต้น</strong> — ไม่ใช่การยืนยันความแท้ 100%</p>
        <p><strong>• คุณภาพของรูปมีผลต่อผลลัพธ์</strong> — รูปชัด แสงดี มุมถูกต้อง = ผลดีขึ้น</p>
        <p><strong>• หากความเชื่อมั่นต่ำ</strong> — ระบบจะแนะนำให้ส่งให้ผู้เชี่ยวชาญตรวจสอบ</p>
        <p><strong>• ใช้เป็นข้อมูลประกอบการตัดสินใจเท่านั้น</strong> — ไม่ควรใช้เป็นเกณฑ์เดียวในการซื้อ-ขาย</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Privacy Notice
    st.markdown("""
    <div class="info-box">
        <h3>🔒 ความเป็นส่วนตัว (Privacy)</h3>
        <p><strong>• ภาพจะถูกประมวลผล</strong> เพื่อการวิเคราะห์ตามนโยบายความเป็นส่วนตัว</p>
        <p><strong>• ถ้าคุณยินยอมให้เก็บภาพ</strong> ระบบจะใช้ภาพเพื่อปรับปรุงโมเดล แต่สามารถขอลบข้อมูลได้</p>
        <p><strong>• ข้อมูลทุกชิ้นเข้ารหัส</strong> และจัดเก็บอย่างปลอดภัยตามมาตรฐาน</p>
    </div>
    """, unsafe_allow_html=True)

def show_tips_section():
    """แสดงคู่มือการใช้งานแบบละเอียด"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 📖 คู่มือการใช้งานอย่างละเอียด")
    st.markdown("<p style='text-align: center; font-size: 1.3rem; color: #6c757d;'>ทำตามขั้นตอนเหล่านี้เพื่อผลลัพธ์ที่ดีที่สุด</p>", unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("""
    <div class="tips-card">
        <h3>🚀 วิธีใช้งานอย่างง่าย (Quick Start)</h3>
        <ol style="font-size: 1.2rem; line-height: 2;">
            <li><strong>กด 📁 อัปโหลดรูป หรือ 📷 ถ่ายรูป</strong> เพื่อแนบภาพด้านหน้าและด้านหลัง</li>
            <li><strong>ตรวจสอบภาพตัวอย่าง</strong> (Preview) ว่าไม่เบลอและเห็นรายละเอียด</li>
            <li><strong>กด 🔍 เริ่มการวิเคราะห์</strong> (ปุ่มใช้งานได้เมื่อมีทั้งสองภาพ)</li>
            <li><strong>รอผล</strong> — ระบบจะแจ้งสถานะและแสดงวงหมุน</li>
            <li><strong>อ่านผล</strong> — ดู Top-1/Top-3, ค่าความเชื่อมั่น และกราฟ</li>
            <li><strong>ถ้าผลไม่ชัด</strong> → ถ่ายใหม่หรือปรึกษาผู้เชี่ยวชาญ</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="tips-card">
            <h3>📸 คำแนะนำการถ่ายรูปให้ได้ผลดี</h3>
            <ul>
                <li><strong>ใช้แสงเพียงพอ</strong> (ไม่ย้อนแสง)</li>
                <li><strong>พื้นหลังเรียบ</strong> (เช่น ผ้าขาว / กระดาษสีเรียบ)</li>
                <li><strong>ภาพชัด ไม่เบลอ</strong> และถ่ายให้เห็นลักษณะเด่น</li>
                <li><strong>ถ่ายให้เห็นทั้งองค์</strong> ไม่ตัดขอบสำคัญออก</li>
                <li><strong>หลีกเลี่ยงเงา</strong> บนพระเครื่อง</li>
                <li><strong>ถ่ายในระยะใกล้พอสมควร</strong> ให้เห็นรายละเอียด</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tips-card">
            <h3>📊 การตีความผลลัพธ์</h3>
            <ul>
                <li><strong>มากกว่า 90%</strong><br/>🎯 <strong>ความเชื่อมั่นสูง</strong> — ผลลัพธ์น่าเชื่อถือ</li>
                <li><strong>70-90%</strong><br/>✅ <strong>เชื่อถือได้ดี</strong> — แต่ควรพิจารณาเพิ่มเติม</li>
                <li><strong>50-70%</strong><br/>⚠️ <strong>ควรตรวจสอบเพิ่ม</strong> — อาจต้องถ่ายใหม่</li>
                <li><strong>น้อยกว่า 50%</strong><br/>❌ <strong>ควรถ่ายใหม่</strong> — หรือส่งตรวจผู้เชี่ยวชาญ</li>
                <li><strong>ใช้ข้อมูลประกอบเท่านั้น</strong> — ไม่ควรเป็นเกณฑ์เดียว</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="tips-card">
            <h3>🏷️ ประเภทที่รองรับ</h3>
            <ul>
                <li>พระศิวลี</li>
                <li>พระสมเด็จ</li>
                <li>ปรกโพธิ์ 9 ใบ</li>
                <li>แหวกม่าน</li>
                <li>หลังรูปเหมือน</li>
                <li>วัดหนองอีดุก</li>
            </ul>
            <p style="margin-top: 20px; font-size: 1.1rem;"><strong>หมายเหตุ:</strong> ระบบมีการอัพเดทและเพิ่มประเภทใหม่อยู่เสมอ</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Warning Card
    st.markdown("""
    <div class="error-box">
        <h3>⚠️ คำเตือนสำคัญ</h3>
        <p style="font-size: 1.3rem; line-height: 1.9;">
            <strong>• ผลลัพธ์เป็นเพียงการประเมินเบื้องต้น</strong> — ควรใช้ร่วมกับผู้เชี่ยวชาญก่อนตัดสินใจซื้อ/ขาย<br/>
            <strong>• หากต้องการผลยืนยัน</strong> — ให้ปรึกษาผู้เชี่ยวชาญด้านพระเครื่องโดยตรง<br/>
            <strong>• ระบบไม่รับประกันความแท้</strong> — เป็นเพียงเครื่องมือช่วยตัดสินใจเบื้องต้น<br/>
            <strong>• ถ้าคุณต้องการความช่วยเหลือ</strong> — อ่านคู่มือ / FAQ ในเมนู Documentation
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
