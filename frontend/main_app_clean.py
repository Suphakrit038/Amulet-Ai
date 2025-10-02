#!/usr/bin/env python3
"""
🔮 Amulet-AI - Production Frontend
ระบบจำแนกพระเครื่องอัจฉริยะ
Thai Amulet Classification System

Modern Clean Implementation
"""

import streamlit as st
import requests
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

# Optional OpenCV import - ไม่จำเป็นสำหรับการทำงานหลัก
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    # cv2 is optional, we can work without it

# PyTorch imports with fallback
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    # Create dummy torch objects
    class DummyTorch:
        device = lambda x: 'cpu'
        no_grad = lambda: DummyContext()
        load = lambda x, **kwargs: {}
        
    class DummyF:
        softmax = lambda x, dim=1: x
        
    class DummyTransforms:
        Compose = lambda x: lambda img: img
        Resize = lambda x: lambda img: img
        ToTensor = lambda: lambda img: img
        Normalize = lambda **kwargs: lambda img: img
        
    class DummyContext:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        
    torch = DummyTorch()
    F = DummyF()
    transforms = DummyTransforms()
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using fallback mode")

# Lazy import joblib to avoid threading issues
def load_joblib_file(file_path):
    """Lazy load joblib file to avoid threading issues in Python 3.13"""
    try:
        import joblib
        return joblib.load(file_path)
    except ImportError:
        print(f"Warning: joblib not available, cannot load {file_path}")
        return None
    except Exception as e:
        print(f"Warning: Failed to load joblib file {file_path}: {e}")
        return None

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Debug: Print sys.path and project_root
print(f"Project root: {project_root}")
print(f"Python path configured correctly")

# Import AI Models (with comprehensive fallback)
AI_MODELS_AVAILABLE = False
try:
    # Import our actual AI models
    from ai_models.enhanced_production_system import EnhancedProductionClassifier
    from ai_models.updated_classifier import UpdatedAmuletClassifier, get_updated_classifier
    from ai_models.compatibility_loader import ProductionOODDetector, try_load_model
    AI_MODELS_AVAILABLE = True
    print("✅ AI Models loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: AI models not available: {e}")
    print("   Using fallback mode - basic functionality only")
    # Create comprehensive dummy classes
    class DummyClassifier:
        def __init__(self, *args, **kwargs):
            self.loaded = False
            
        def load_model(self, *args, **kwargs):
            return False
            
        def predict(self, *args, **kwargs):
            return {
                "status": "error", 
                "error": "AI models not available in this environment",
                "predicted_class": "Unknown",
                "confidence": 0.0,
                "probabilities": {},
                "method": "Fallback"
            }
    
    EnhancedProductionClassifier = DummyClassifier
    UpdatedAmuletClassifier = DummyClassifier
    get_updated_classifier = lambda: DummyClassifier()
    ProductionOODDetector = DummyClassifier
    AI_MODELS_AVAILABLE = False

# Import core modules (with comprehensive fallback)
CORE_AVAILABLE = False
try:
    from core.error_handling_enhanced import error_handler, validate_image_file
    from core.performance_monitoring import performance_monitor
    CORE_AVAILABLE = True
    print("✅ Core modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Core modules not available: {e}")
    print("   Using fallback implementations")
    # Comprehensive fallback implementations
    def error_handler(error_type="general"):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in {func.__name__}: {str(e)}")
                    return {
                        "status": "error", 
                        "error": f"Function {func.__name__} failed: {str(e)}",
                        "method": "Fallback"
                    }
            return wrapper
        return decorator
    
    def validate_image_file(file):
        """Basic file validation fallback"""
        if file is None:
            return False
        # Basic checks
        if hasattr(file, 'name') and hasattr(file, 'size'):
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
            return any(file.name.lower().endswith(ext) for ext in valid_extensions)
        return True  # If we can't check, assume valid
    
    class performance_monitor:
        @staticmethod
        def collect_metrics():
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "fallback_mode",
                "memory_usage": "unknown"
            }
    
    CORE_AVAILABLE = False

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
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Display system status at the top (for debugging)
if not AI_MODELS_AVAILABLE or not CORE_AVAILABLE:
    status_info = []
    if not AI_MODELS_AVAILABLE:
        status_info.append("AI Models: ❌ Fallback Mode")
    else:
        status_info.append("AI Models: ✅ Available")
        
    if not CORE_AVAILABLE:
        status_info.append("Core Modules: ❌ Fallback Mode")
    else:
        status_info.append("Core Modules: ✅ Available")
        
    if not TORCH_AVAILABLE:
        status_info.append("PyTorch: ❌ Not Available")
    else:
        status_info.append("PyTorch: ✅ Available")
        
    st.info(f"🔧 System Status: {' | '.join(status_info)}")

# Modern Modal Design CSS
st.markdown(f"""
<style>
    /* Import Modern Fonts - Thai + English */
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@200;300;400;500;600;700;800&family=Prompt:wght@300;400;500;600;700;800&display=swap');
    
    /* Modern App Background - Creamy White */
    .stApp {{
        font-family: 'Sarabun', 'Prompt', -apple-system, BlinkMacSystemFont, sans-serif;
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
        max-width: 1000px;
    }}
    
    /* Modal-Style Logo Header */
    .logo-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 60px 80px;
        background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(255,255,255,0.92) 100%);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(128, 0, 0, 0.08);
        margin-bottom: 30px;
        border: 1px solid rgba(212, 175, 55, 0.15);
        position: relative;
        overflow: hidden;
    }}
    
    .logo-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['gold']}, {COLORS['primary']});
    }}
    
    .logo-left {{
        display: flex;
        align-items: center;
        gap: 20px;
        z-index: 1;
    }}
    
    .logo-text {{
        display: flex;
        flex-direction: column;
        gap: 2px;
    }}
    
    .logo-title {{
        font-family: 'Prompt', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: {COLORS['primary']};
        margin: 0;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }}
    
    .logo-subtitle {{
        font-family: 'Sarabun', sans-serif;
        font-size: 1.5rem;
        font-weight: 500;
        color: {COLORS['gray']};
        margin: 0;
        letter-spacing: 0.3px;
    }}
    
    .logo-right {{
        display: flex;
        align-items: center;
        gap: 25px;
        z-index: 1;
    }}
    
    .logo-img {{
        height: 160px;
        width: auto;
        object-fit: contain;
        filter: drop-shadow(0 3px 6px rgba(0,0,0,0.12));
        transition: transform 0.3s ease;
    }}
    
    .logo-img:hover {{
        transform: scale(1.03);
    }}
    
    .logo-img-small {{
        height: 180px;
        width: auto;
        object-fit: contain;
        filter: drop-shadow(0 3px 6px rgba(0,0,0,0.12));
        transition: transform 0.3s ease;
    }}
    
    .logo-img-small:hover {{
        transform: scale(1.03);
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
        padding: 10px 0;
        padding-left: 28px;
        position: relative;
        font-size: 1.0rem;
        line-height: 1.7;
    }}
    
    .feature-card ul li:before {{
        content: '✓';
        position: absolute;
        left: 0;
        color: {COLORS['gold']};
        font-weight: bold;
        font-size: 1.2rem;
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
        max-width: 1200px;
        width: 98vw;
        min-width: 350px;
        margin: 30px auto;
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
    
    /* Modern Section Divider */
    .section-divider {{
        height: 1.5px;
        background: linear-gradient(90deg, transparent, {COLORS['gold']}, transparent);
        margin: 35px 0;
        border-radius: 2px;
        opacity: 0.6;
    }}
    
    /* Modern Button with Gradient and Animation */
    .stButton > button {{
        font-family: 'Sarabun', sans-serif;
        background: {COLORS['accent']};
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 40px;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 15px rgba(184, 134, 11, 0.3);
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
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:hover {{
        background: {COLORS['gold']};
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.4);
        transform: translateY(-2px) scale(1.01);
    }}
    
    .stButton > button:active {{
        transform: translateY(0) scale(0.98);
        box-shadow: 0 2px 10px rgba(184, 134, 11, 0.3);
    }}
    
    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background: transparent;
        padding: 0;
        margin-bottom: 25px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Sarabun', sans-serif;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 14px 32px !important;
        border-radius: 10px !important;
        background: #f5ebe0 !important;
        border: none !important;
        transition: all 0.25s ease !important;
        color: #6c757d !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: #ede0d4 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(128, 0, 0, 0.1);
        color: {COLORS['primary']} !important;
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: {COLORS['accent']} !important;
        color: white !important;
        box-shadow: 0 3px 12px rgba(184, 134, 11, 0.35);
    }}
    
    /* Modern Typography */
    h1 {{
        font-family: 'Prompt', sans-serif;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 20px !important;
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['gold']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.3 !important;
    }}
    
    h2 {{
        font-family: 'Prompt', sans-serif;
        font-size: 2.2rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px !important;
        margin-bottom: 18px !important;
        color: #2d3748 !important;
        line-height: 1.4 !important;
    }}
    
    h3 {{
        font-family: 'Sarabun', sans-serif;
        font-size: 1.65rem !important;
        font-weight: 600 !important;
        margin-bottom: 14px !important;
        color: #2d3748 !important;
        line-height: 1.4 !important;
    }}
    
    h4 {{
        font-family: 'Sarabun', sans-serif;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-bottom: 12px !important;
        color: #4a5568 !important;
        line-height: 1.5 !important;
    }}
    
    /* Modern Text Styling */
    p {{
        font-family: 'Sarabun', sans-serif;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
        color: #4a5568 !important;
        font-weight: 400 !important;
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['gold']}) !important;
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
        border-color: {COLORS['accent']};
        transform: scale(1.01);
    }}
    
    [data-testid="stFileUploader"] label {{
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: {COLORS['primary']} !important;
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

# AI Model Loading with Caching
@st.cache_resource
def load_ai_model():
    """Load AI model with caching and comprehensive error handling"""
    try:
        if not AI_MODELS_AVAILABLE or not CORE_AVAILABLE:
            st.warning("🔧 AI models not available. Using demo mode.")
            return None
            
        model_loader = AIModelLoader()
        model_data = model_loader.load_production_model()
        
        if model_data is None:
            st.error("❌ Failed to load AI model")
            return None
            
        st.success("✅ AI model loaded successfully")
        return model_data
        
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        return None

# Core Classification Function
def classify_image(image, model_data=None, use_api=False):
    """Main image classification function with multiple fallback modes"""
    try:
        if use_api:
            return classify_via_api(image)
        elif model_data and AI_MODELS_AVAILABLE:
            return ai_local_prediction(image, model_data)
        else:
            return demo_classification()
            
    except Exception as e:
        st.error(f"❌ Classification error: {str(e)}")
        return demo_classification()

# API Classification
def classify_via_api(image):
    """Classify image using API endpoint"""
    try:
        # Convert PIL image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        files = {'file': ('image.jpg', img_buffer, 'image/jpeg')}
        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return {
                'prediction': result.get('prediction', 'Unknown'),
                'confidence': result.get('confidence', 0.0),
                'details': result.get('details', {}),
                'method': 'API'
            }
        else:
            st.warning(f"API returned status {response.status_code}")
            return demo_classification()
            
    except requests.exceptions.RequestException as e:
        st.warning(f"API connection failed: {str(e)}")
        return demo_classification()
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return demo_classification()

# Local AI Prediction
def ai_local_prediction(image, model_data):
    """Perform local AI prediction with comprehensive error handling"""
    try:
        # Get model components
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        label_encoder = model_data.get('label_encoder')
        
        if not all([model, scaler, label_encoder]):
            st.warning("⚠️ Model components incomplete. Using demo mode.")
            return demo_classification()
        
        # Process image with error handling
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Resize image
            if img_array.shape[:2] != (224, 224):
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(img_array)
                pil_img = pil_img.resize((224, 224))
                img_array = np.array(pil_img)
            
            # Normalize pixel values
            img_array = img_array.astype(np.float32) / 255.0
            
            # Flatten for traditional ML models
            img_flattened = img_array.flatten().reshape(1, -1)
            
            # Scale features
            img_scaled = scaler.transform(img_flattened)
            
            # Make prediction
            prediction = model.predict(img_scaled)[0]
            confidence = max(model.predict_proba(img_scaled)[0])
            
            # Decode label
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            
            return {
                'prediction': predicted_class,
                'confidence': float(confidence),
                'details': {
                    'image_shape': img_array.shape,
                    'feature_count': img_scaled.shape[1],
                    'model_type': type(model).__name__
                },
                'method': 'Local AI'
            }
            
        except Exception as e:
            st.warning(f"Image processing error: {str(e)}")
            return demo_classification()
            
    except Exception as e:
        st.error(f"Local prediction error: {str(e)}")
        return demo_classification()

# Demo Classification (Fallback)
def demo_classification():
    """Demo classification for testing purposes"""
    demo_results = [
        {'prediction': 'พระสมเด็จ', 'confidence': 0.89},
        {'prediction': 'พระนางพญา', 'confidence': 0.84},
        {'prediction': 'พระพุทธ', 'confidence': 0.91},
        {'prediction': 'พระเครื่อง', 'confidence': 0.76},
        {'prediction': 'พระยันต์', 'confidence': 0.82}
    ]
    
    import random
    result = random.choice(demo_results)
    
    return {
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'details': {'mode': 'demo', 'note': 'This is a demonstration result'},
        'method': 'Demo Mode'
    }

# Result Display Function
def display_classification_result(result, image):
    """Display classification results with beautiful HTML formatting"""
    if not result:
        st.error("❌ No classification result available")
        return
    
    prediction = result.get('prediction', 'Unknown')
    confidence = result.get('confidence', 0.0)
    method = result.get('method', 'Unknown')
    details = result.get('details', {})
    
    # Confidence color coding
    if confidence >= 0.8:
        conf_color = COLORS['success']
        conf_icon = "🟢"
        conf_text = "สูง (High)"
    elif confidence >= 0.6:
        conf_color = COLORS['warning']
        conf_icon = "🟡"
        conf_text = "ปานกลาง (Medium)"
    else:
        conf_color = COLORS['error']
        conf_icon = "🔴"
        conf_text = "ต่ำ (Low)"
    
    # Display result card
    st.markdown(f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 25px;">
            <div style="flex: 1;">
                <h2 style="color: {COLORS['primary']}; margin-bottom: 15px; font-size: 2.2rem;">
                    🎯 ผลการวิเคราะห์
                </h2>
                <div style="background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['gold']}); 
                           color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                    <h3 style="color: white !important; margin-bottom: 10px; font-size: 1.8rem;">
                        📿 ประเภทพระเครื่อง
                    </h3>
                    <p style="font-size: 2.0rem; font-weight: 700; margin: 0; color: white !important;">
                        {prediction}
                    </p>
                </div>
                
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <span style="font-size: 1.4rem; margin-right: 10px;">{conf_icon}</span>
                    <span style="font-size: 1.5rem; font-weight: 600; color: {conf_color};">
                        ความมั่นใจ: {confidence:.1%} ({conf_text})
                    </span>
                </div>
                
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <p style="margin: 0; color: {COLORS['gray']} !important; font-size: 1.1rem;">
                        <strong>วิธีการวิเคราะห์:</strong> {method}
                    </p>
                </div>
                
                {"<div style='background: #e3f2fd; padding: 15px; border-radius: 10px; border-left: 4px solid #2196f3;'>" if details else ""}
                {"<h4 style='color: #1976d2 !important; margin-bottom: 10px;'>รายละเอียดเพิ่มเติม:</h4>" if details else ""}
                {"".join([f"<p style='margin: 5px 0; color: #424242 !important;'><strong>{k}:</strong> {v}</p>" for k, v in details.items()]) if details else ""}
                {"</div>" if details else ""}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def resize_image(image, max_size=800):
    """Resize image while maintaining aspect ratio"""
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def validate_image(uploaded_file):
    """Validate uploaded image file"""
    if uploaded_file is None:
        return False, "ไม่พบไฟล์ที่อัปโหลด"
    
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"ไฟล์ใหญ่เกินไป (สูงสุด {MAX_FILE_SIZE//1024//1024} MB)"
    
    try:
        image = Image.open(uploaded_file)
        if image.format.lower() not in ['jpeg', 'jpg', 'png', 'bmp']:
            return False, "รูปแบบไฟล์ไม่ถูกต้อง (รองรับ JPEG, PNG, BMP เท่านั้น)"
        return True, "ไฟล์ถูกต้อง"
    except Exception as e:
        return False, f"ไฟล์เสียหาย: {str(e)}"

def predict_via_api(image_data):
    """Make prediction via API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"image": image_data},
            timeout=30
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "ไม่สามารถเชื่อมต่อ API ได้ (ตรวจสอบว่า API Server ทำงานอยู่หรือไม่)"
    except requests.exceptions.Timeout:
        return False, "การเชื่อมต่อหมดเวลา"
    except Exception as e:
        return False, f"เกิดข้อผิดพลาด: {str(e)}"

# Mock prediction function for when API is not available
def mock_prediction(image):
    """Mock prediction for demo purposes"""
    time.sleep(1)  # Simulate processing time
    
    # Mock classes (Thai amulet types)
    classes = [
        "พระสมเด็จ", "พระนางพญา", "พระปิดตา", "พระกรุง", 
        "พระผงสุปดีป์", "พระพุทธชินราช"
    ]
    
    # Generate mock predictions
    predictions = []
    confidence_scores = np.random.dirichlet(np.ones(len(classes))) * 100
    
    for class_name, confidence in zip(classes, confidence_scores):
        predictions.append({
            "class": class_name,
            "confidence": float(confidence),
            "probability": float(confidence / 100)
        })
    
    # Sort by confidence
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {
        "status": "success",
        "predictions": predictions,
        "top_prediction": predictions[0],
        "processing_time": 1.23,
        "model_version": "v2.1.0"
    }

# Main Application Function
def main():
    """Main application with tab structure"""
    
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # App Header with Logo
    display_app_header()
    
    # System Status (conditional)
    if st.checkbox("🔧 แสดงสถานะระบบ", value=False):
        display_system_status()
    
    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 หน้าหลัก", 
        "📖 เกี่ยวกับระบบ", 
        "📚 คู่มือการใช้งาน", 
        "❓ FAQ"
    ])
    
    with tab1:
        dual_image_mode()
    
    with tab2:
        subtab1, subtab2 = st.tabs(["📋 ข้อมูลทั่วไป", "🔧 วิธีการทำงาน"])
        
        with subtab1:
            show_introduction_section()
        
        with subtab2:
            show_how_it_works_section()
            show_about_section()
    
    with tab3:
        show_tips_section()
    
    with tab4:
        show_faq_section()
    
    # Footer
    display_footer()

def display_app_header():
    """Display application header with logo"""
    try:
        # Try to load logo images
        logo_path_1 = os.path.join(PROJECT_ROOT, "frontend", "imgae", "Amulet-AI-logo-trans.png")
        logo_path_2 = os.path.join(PROJECT_ROOT, "frontend", "imgae", "thai-amulet-logo.png")
        
        # Check if logo files exist
        logo1_exists = os.path.exists(logo_path_1)
        logo2_exists = os.path.exists(logo_path_2)
        
        if logo1_exists or logo2_exists:
            # Header with logos
            st.markdown(f"""
            <div class="logo-header">
                <div class="logo-left">
                    <div class="logo-text">
                        <h1 class="logo-title">🔮 Amulet-AI</h1>
                        <p class="logo-subtitle">ระบบวิเคราะห์พระเครื่องอัจฉริยะ</p>
                    </div>
                </div>
                <div class="logo-right">
                    {"<img src='data:image/png;base64,{}' class='logo-img' alt='Amulet-AI Logo'>".format(encode_image_to_base64(Image.open(logo_path_1))) if logo1_exists else ""}
                    {"<img src='data:image/png;base64,{}' class='logo-img-small' alt='Thai Amulet Logo'>".format(encode_image_to_base64(Image.open(logo_path_2))) if logo2_exists else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Header without logos
            st.markdown("""
            <div class="logo-header">
                <div class="logo-left">
                    <div class="logo-text">
                        <h1 class="logo-title">🔮 Amulet-AI</h1>
                        <p class="logo-subtitle">ระบบวิเคราะห์พระเครื่องอัจฉริยะ - Thai Amulet Classification System</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        # Fallback header
        st.markdown("""
        <div class="card">
            <h1>🔮 Amulet-AI</h1>
            <p style="font-size: 1.3rem; color: #666;">ระบบวิเคราะห์พระเครื่องอัจฉริยะ</p>
        </div>
        """, unsafe_allow_html=True)

def display_footer():
    """Display application footer"""
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="text-align: center; padding: 40px 20px; background: {COLORS['light']}; border-radius: 15px; margin-top: 40px;">
        <h3 style="color: {COLORS['primary']}; margin-bottom: 15px;">🔮 Amulet-AI</h3>
        <p style="color: {COLORS['gray']}; font-size: 1.1rem; margin-bottom: 20px;">
            ระบบวิเคราะห์พระเครื่องอัจฉริยะ | Thai Amulet Classification System
        </p>
        <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin-bottom: 20px;">
            <div style="color: {COLORS['gray']};">
                <strong>🎯 ความแม่นยำ:</strong> มากกว่า 90%
            </div>
            <div style="color: {COLORS['gray']};">
                <strong>⚡ เวลาประมวลผล:</strong> < 3 วินาที
            </div>
            <div style="color: {COLORS['gray']};">
                <strong>🔒 ความปลอดภัย:</strong> ไม่เก็บข้อมูล
            </div>
        </div>
        <p style="color: {COLORS['muted']}; font-size: 0.95rem; margin: 0;">
            <em>พัฒนาด้วยเทคโนโลยี AI ขั้นสูงเพื่อการอนุรักษ์มรดกทางวัฒนธรรมไทย</em><br>
            © 2024 Amulet-AI Project | All Rights Reserved
        </p>
    </div>
    """, unsafe_allow_html=True)

# Helper utility functions
def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return ""

def resize_image(image, max_size=800):
    """Resize image while maintaining aspect ratio"""
    try:
        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        st.error(f"Error resizing image: {str(e)}")
        return image

def validate_image(uploaded_file):
    """Validate uploaded image file"""
    if uploaded_file is None:
        return False, "ไม่พบไฟล์ที่อัปโหลด"
    
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"ไฟล์ใหญ่เกินไป (สูงสุด {MAX_FILE_SIZE//1024//1024} MB)"
    
    try:
        image = Image.open(uploaded_file)
        if image.format.lower() not in ['jpeg', 'jpg', 'png', 'bmp']:
            return False, "รูปแบบไฟล์ไม่ถูกต้อง (รองรับ JPEG, PNG, BMP เท่านั้น)"
        return True, "ไฟล์ถูกต้อง"
    except Exception as e:
        return False, f"ไฟล์เสียหาย: {str(e)}"

if __name__ == "__main__":
    main()