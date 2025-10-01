#!/usr/bin/env python3
"""
Production-Ready Frontend for Amulet-AI
ระบบ Frontend ที่ใช้งานได้จริงสำหรับการจำแนกพระเครื่อง
เวอร์ชันที่ปรับปรุงและใช้ component แบบโมดูลาร์
"""

import streamlit as st
import os
import sys
import requests
import time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from pathlib import Path
from datetime import datetime
import psutil
import threading
from collections import deque

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components and utils
try:
    from frontend.components.image_display import ImageDisplayComponent
    from frontend.components.analysis_results import AnalysisResultsComponent
    from frontend.components.file_uploader import FileUploaderComponent
    from frontend.components.mode_selector import ModeSelectorComponent
    from frontend.utils.image_processor import ImagePreprocessor
    from frontend.utils.file_validator import FileValidator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ตั้งค่าหน้าเว็บ
# Load logo for page icon
try:
    logo_path = os.path.join(os.path.dirname(__file__), 'imgae', 'Amulet-AI_logo.png')
    if os.path.exists(logo_path):
        from PIL import Image
        logo_img = Image.open(logo_path)
        page_icon = logo_img
    else:
        page_icon = "พ"
except:
    page_icon = "พ"

st.set_page_config(
    page_title="Amulet-AI - ระบบวิเคราะห์พระเครื่องด้วย AI",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuration
API_BASE_URL = "http://localhost:8000"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Theme Colors สำหรับไทยธีม
THEME_COLORS = {
    'primary': '#800000',   # แดงเลือดหมู
    'accent': '#B8860B',    # ทอง (dark goldenrod)
    'gold': '#D4AF37',      # ทองสว่าง
    'success': '#10b981',
    'warning': '#f59e0b',
    'error': '#ef4444',
    'info': '#3b82f6'
}

def load_css():
    """Load CSS from external file with enhanced Thai styling"""
    css_file = os.path.join(os.path.dirname(__file__), 'style.css')
    
    # Enhanced CSS with Thai theme - Sharp edges, wave animation, no emojis
    enhanced_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700;800;900&display=swap');
    
    .stApp {{
        font-family: 'Prompt', 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #f8f5f0 0%, #ffffff 50%, #faf7f2 100%);
        border: 4px solid #800000;
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
        padding: 0;
        margin-bottom: 2rem;
        color: white;
        text-align: left;
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
    
    .feature-card {{
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 2px solid rgba(128, 0, 0, 0.1);
    }}
    
    .feature-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(128, 0, 0, 0.2);
    }}
    
    .glass-card {{
        background: rgba(255, 255, 255, 0.85);
        border: 2px solid rgba(128, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(128, 0, 0, 0.2);
        margin-bottom: 2rem;
    }}
    
    .status-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
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
    
    .upload-section {{
        background: linear-gradient(135deg, rgba(128, 0, 0, 0.05) 0%, rgba(184, 134, 11, 0.05) 100%);
        padding: 3rem 2rem;
        border: 3px solid {THEME_COLORS['primary']};
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
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }}
    
    .logo-img {{
        height: 80px;
        max-width: 200px;
        object-fit: contain;
        padding: 10px;
        background: white;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }}
    
    .logo-img:hover {{
        transform: scale(1.05);
    }}
    
    .metric-card {{
        background: white;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-left: 4px solid {THEME_COLORS['primary']};
        margin: 1rem 0;
    }}
    
    .section-divider {{
        height: 3px;
        background: linear-gradient(90deg, {THEME_COLORS['primary']} 0%, {THEME_COLORS['accent']} 50%, {THEME_COLORS['primary']} 100%);
        margin: 3rem 0;
    }}
    
    /* Sharp edges for uploaded images */
    .stImage img {{
        max-width: 100% !important;
        width: auto !important;
        height: auto !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }}
    
    [data-testid="stImage"] {{
        display: flex;
        justify-content: center;
    }}
    
    [data-testid="stImageContainer"] {{
        width: 100% !important;
    }}
    
    [data-testid="stImageCaption"] {{
        text-align: center;
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
        padding: 0.5rem;
        background: #f5f5f5;
        border: 1px solid #ddd;
        margin-top: 1rem;
    }}
    
    /* Sharp message boxes */
    .stAlert, .stSuccess, .stWarning, .stError, .stInfo {{
        border-radius: 0 !important;
        border-left: 4px solid !important;
    }}
    
    /* File uploader sharp edges */
    .stFileUploader {{
        border: 2px solid #ddd !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }}
    
    .stFileUploader:hover {{
        border-color: {THEME_COLORS['primary']} !important;
        box-shadow: 0 6px 20px rgba(128, 0, 0, 0.2) !important;
    }}
    
    /* Camera input styling */
    [data-testid="stCameraInput"] {{
        border: 2px solid #ddd !important;
        padding: 1rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }}
    
    [data-testid="stCameraInput"]:hover {{
        border-color: {THEME_COLORS['primary']} !important;
        box-shadow: 0 6px 20px rgba(128, 0, 0, 0.2) !important;
    }}
    
    /* Tabs styling for upload methods */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: #f5f5f5;
        border: 2px solid #ddd;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 3rem;
        background: #f5f5f5;
        border: none;
        color: #666;
        font-weight: 600;
        padding: 0 2rem;
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: #e0e0e0;
        color: {THEME_COLORS['primary']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {THEME_COLORS['primary']} !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(128, 0, 0, 0.3);
    }}
    
    .stTabs [data-baseweb="tab-panel"] {{
        border: 2px solid #ddd;
        border-top: none;
        padding: 1.5rem;
        background: white;
    }}
    </style>
    """
    
    # Try to load external CSS file
    if os.path.exists(css_file):
        try:
            with open(css_file, 'r', encoding='utf-8') as f:
                external_css = f.read()
            st.markdown(f'<style>{external_css}</style>', unsafe_allow_html=True)
        except:
            pass
    
    # Apply enhanced CSS
    st.markdown(enhanced_css, unsafe_allow_html=True)

import base64
from io import BytesIO

def get_logo_base64():
    """Convert logo image to base64 for embedding"""
    try:
        logo_path = os.path.join(os.path.dirname(__file__), 'imgae', 'Amulet-AI_logo.png')
        if os.path.exists(logo_path):
            with open(logo_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode()
        return ""
    except:
        return ""

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self):
        if 'perf_data' not in st.session_state:
            st.session_state.perf_data = {
                'cpu_usage': deque(maxlen=50),
                'memory_usage': deque(maxlen=50),
                'api_response_times': deque(maxlen=20),
                'processing_times': deque(maxlen=20),
                'timestamps': deque(maxlen=50)
            }
    
    def update_system_metrics(self):
        """อัปเดตข้อมูลระบบ"""
        try:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            timestamp = datetime.now()
            
            st.session_state.perf_data['cpu_usage'].append(cpu)
            st.session_state.perf_data['memory_usage'].append(memory)
            st.session_state.perf_data['timestamps'].append(timestamp)
        except:
            pass
    
    def add_api_response_time(self, response_time):
        """เพิ่มข้อมูลเวลาตอบสนอง API"""
        st.session_state.perf_data['api_response_times'].append(response_time)
    
    def add_processing_time(self, processing_time):
        """เพิ่มข้อมูลเวลาประมวลผล"""
        st.session_state.perf_data['processing_times'].append(processing_time)
    
    def get_performance_chart(self):
        """สร้าง performance chart"""
        if not st.session_state.perf_data['cpu_usage']:
            return None
        
        fig = go.Figure()
        
        # CPU Usage
        fig.add_trace(go.Scatter(
            y=list(st.session_state.perf_data['cpu_usage']),
            mode='lines+markers',
            name='CPU (%)',
            line=dict(color='#800000', width=2),
            marker=dict(size=4)
        ))
        
        # Memory Usage
        fig.add_trace(go.Scatter(
            y=list(st.session_state.perf_data['memory_usage']),
            mode='lines+markers',
            name='Memory (%)',
            line=dict(color='#B8860B', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title="Performance Monitor",
            xaxis_title="Time",
            yaxis_title="Usage (%)",
            height=200,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Prompt, sans-serif", size=10),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def get_api_stats(self):
        """ได้ข้อมูลสถิติ API"""
        response_times = list(st.session_state.perf_data['api_response_times'])
        processing_times = list(st.session_state.perf_data['processing_times'])
        
        stats = {
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'total_requests': len(st.session_state.analysis_history) if 'analysis_history' in st.session_state else 0
        }
        
        return stats

class AmuletFrontend:
    """Enhanced Frontend class สำหรับ API communication และ cache management"""
    
    def __init__(self):
        self.base_url = API_BASE_URL
        if 'api_cache' not in st.session_state:
            st.session_state.api_cache = {}
        self.perf_monitor = PerformanceMonitor()
    
    def check_api_health(self):
        """ตรวจสอบสถานะ API พร้อม cache"""
        try:
            # Check cache first
            cache_key = 'api_health_check'
            if cache_key in st.session_state.api_cache:
                cached_result, timestamp = st.session_state.api_cache[cache_key]
                # Cache for 30 seconds
                if (datetime.now() - timestamp).seconds < 30:
                    return cached_result
            
            response = requests.get(f"{self.base_url}/health", timeout=5)
            result = response.status_code == 200
            
            # Store in cache
            st.session_state.api_cache[cache_key] = (result, datetime.now())
            return result
        except:
            return False
    
    def get_classes_info(self):
        """ดึงข้อมูลคลาสจาก API"""
        try:
            response = requests.get(f"{self.base_url}/classes", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def predict_image(self, front_image, back_image=None, enhance=True):
        """ส่งรูปภาพไป API เพื่อทำนาย"""
        try:
            files = {}
            
            # Prepare front image
            if front_image:
                files['front_image'] = ('front.jpg', front_image.getvalue(), 'image/jpeg')
            
            # Prepare back image if provided
            if back_image:
                files['back_image'] = ('back.jpg', back_image.getvalue(), 'image/jpeg')
            
            # Additional parameters
            data = {
                'enhance': str(enhance).lower(),
                'analysis_type': 'dual' if back_image else 'single'
            }
            
            response = requests.post(
                f"{self.base_url}/predict",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}

def main():
    # Load CSS และ styling
    load_css()
    
    # Initialize components
    image_display = ImageDisplayComponent()
    analysis_results = AnalysisResultsComponent()
    file_uploader = FileUploaderComponent(MAX_FILE_SIZE)
    mode_selector = ModeSelectorComponent()
    frontend = AmuletFrontend()
    
    # Initialize session state
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = 'dual'
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    
    # Auto-refresh toggle in sidebar
    with st.sidebar:
        st.markdown("### ⚙️ การตั้งค่า")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox(
            "Auto-refresh Performance", 
            value=st.session_state.auto_refresh,
            help="อัปเดตข้อมูล Performance ทุก 5 วินาที"
        )
        st.session_state.auto_refresh = auto_refresh
        
        # Theme selector
        st.markdown("### 🎨 ธีมสี")
        theme_options = {
            "Thai Classic": {"primary": "#800000", "accent": "#B8860B"},
            "Royal Blue": {"primary": "#1e3a8a", "accent": "#3b82f6"},
            "Emerald": {"primary": "#065f46", "accent": "#10b981"},
            "Purple": {"primary": "#581c87", "accent": "#8b5cf6"}
        }
        
        selected_theme = st.selectbox(
            "เลือกธีมสี",
            options=list(theme_options.keys()),
            index=0
        )
        
        # Update theme colors if changed
        if 'current_theme' not in st.session_state or st.session_state.current_theme != selected_theme:
            st.session_state.current_theme = selected_theme
            THEME_COLORS.update(theme_options[selected_theme])
            st.rerun()
        
        # Performance settings
        st.markdown("### 📊 Performance")
        show_detailed_stats = st.checkbox("แสดงสถิติแบบละเอียด", value=True)
        st.session_state.show_detailed_stats = show_detailed_stats
        
        # Image processing settings
        st.markdown("### 🖼️ การประมวลผลรูปภาพ")
        
        image_quality = st.slider(
            "คุณภาพการบีบอัด (%)",
            min_value=70,
            max_value=100,
            value=90,
            step=5,
            help="คุณภาพสูงขึ้น = ไฟล์ใหญ่ขึ้น"
        )
        st.session_state.image_quality = image_quality
        
        auto_enhance_default = st.checkbox(
            "เปิดใช้ Auto Enhancement โดยอัตโนมัติ",
            value=True,
            help="ปรับปรุงคุณภาพรูปภาพอัตโนมัติเมื่ออัปโหลด"
        )
        st.session_state.auto_enhance_default = auto_enhance_default
        
        # Divider
        st.markdown("---")
        
        # System info
        st.markdown("### ℹ️ ข้อมูลระบบ")
        try:
            cpu_count = psutil.cpu_count()
            memory_total = psutil.virtual_memory().total / (1024**3)  # GB
            st.info(f"**CPU Cores**: {cpu_count}\\n**Memory**: {memory_total:.1f} GB")
        except:
            st.info("ไม่สามารถอ่านข้อมูลระบบได้")
        
        if auto_refresh:
            # Auto refresh every 5 seconds
            time.sleep(0.1)  # Small delay
            st.rerun()
    
    # Check API health
    api_healthy = frontend.check_api_health()
    
    # ==========================================================
    # Header Section with Thai Design
    # ==========================================================
    
    # Main header with pure raw HTML and logos on the right
    
    # Get all logo base64
    amulet_logo = get_logo_base64()
    
    # Load other logos
    def get_other_logos():
        try:
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
    
    other_logos = get_other_logos()
    
    # Build logos HTML section - จัดเรียงแนวนอน
    all_logos_html = ""
    
    # เพิ่มโลโก้ Thai-Austrian
    if 'thai_austrian' in other_logos:
        all_logos_html += f'<img src="data:image/gif;base64,{other_logos["thai_austrian"]}" style="height: 80px; width: auto; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); background: rgba(255,255,255,0.1); padding: 8px; margin: 0 8px;" alt="Thai-Austrian Logo">'
    
    # เพิ่มโลโก้ DEPA
    if 'depa' in other_logos:
        all_logos_html += f'<img src="data:image/png;base64,{other_logos["depa"]}" style="height: 80px; width: auto; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); background: rgba(255,255,255,0.1); padding: 8px; margin: 0 8px;" alt="DEPA Logo">'
    
    # Pure raw HTML header - ใหม่แบบเรียบง่าย
    header_html = f"""
    <div class="main-header">
        <div style="position: relative; z-index: 2; display: flex; align-items: center; justify-content: space-between; gap: 2rem; padding: 2rem;">
            <!-- Content Section -->
            <div style="flex: 1;">
                <h1 style="font-size: 6rem; font-weight: 900; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.4); 
                           color: white; letter-spacing: 3px; text-transform: uppercase;">
                    AMULET-AI
                </h1>
                <h2 style="font-size: 2.2rem; margin: 2rem 0 0 0; opacity: 0.95; font-weight: 700; text-shadow: 1px 1px 3px rgba(0,0,0,0.3); color: white;">
                    ระบบวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์
                </h2>
            </div>
            
            <!-- Logos Section -->
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                {all_logos_html}
            </div>
        </div>
    </div>
    """
    
    st.components.v1.html(header_html, height=280)
    
    # API Status Section พร้อม interactive elements
    status_class = "status-online" if api_healthy else "status-offline"
    status_text = "API เชื่อมต่อสำเร็จ" if api_healthy else "API ไม่พร้อมใช้งาน"
    status_indicator = "●" if api_healthy else "●"
    
    # Enhanced status with more info
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if api_healthy:
            st.success(f"🟢 {status_text} - ระบบพร้อมใช้งาน")
        else:
            st.error(f"🔴 {status_text} - กำลังใช้โหมด Demo")
            
        # Additional system info
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.caption(f"⏰ อัปเดตล่าสุด: {current_time}")
        except:
            pass
    
    # Section Divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # ==========================================================
    # File Upload Section
    # ==========================================================
    
    st.markdown("## อัปโหลดรูปพระเครื่อง")
    
    # Display camera tips
    file_uploader.display_camera_tips()
    
    # Display upload tips
    file_uploader.display_upload_tips(st.session_state.analysis_mode)
    
    # File upload based on mode
    front_image = None
    back_image = None
    
    # Use dual image uploader (since we removed single mode)
    front_image, back_image = file_uploader.dual_image_uploader()
    
    # Display uploaded images with enhanced info - แนวนอนขนาดใหญ่
    if front_image or back_image:
        st.markdown("### รูปภาพที่อัปโหลด")
        
        # สร้าง columns สำหรับแสดงรูปแนวนอน - ขนาดใหญ่ครึ่งหน้าจอ
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            if front_image:
                st.markdown('<div style="text-align: center; margin-bottom: 1rem;"><h3 style="color: #800000; margin: 0; padding: 1rem; background: #f5f5f5; border: 2px solid #800000;">รูปด้านหน้า</h3></div>', unsafe_allow_html=True)
                image_display.display_uploaded_image(front_image, "หน้า", MAX_FILE_SIZE)
            else:
                st.markdown('<div style="text-align: center; padding: 4rem 2rem; border: 3px solid #ccc; background: #f9f9f9; min-height: 400px; display: flex; align-items: center; justify-content: center;"><p style="color: #999; font-size: 1.5rem; margin: 0;">รอรูปด้านหน้า...</p></div>', unsafe_allow_html=True)
        
        with col2:
            if back_image:
                st.markdown('<div style="text-align: center; margin-bottom: 1rem;"><h3 style="color: #800000; margin: 0; padding: 1rem; background: #f5f5f5; border: 2px solid #800000;">รูปด้านหลัง</h3></div>', unsafe_allow_html=True)
                image_display.display_uploaded_image(back_image, "หลัง", MAX_FILE_SIZE)
            else:
                st.markdown('<div style="text-align: center; padding: 4rem 2rem; border: 3px solid #ccc; background: #f9f9f9; min-height: 400px; display: flex; align-items: center; justify-content: center;"><p style="color: #999; font-size: 1.5rem; margin: 0;">รอรูปด้านหลัง...</p></div>', unsafe_allow_html=True)
    
    # ==========================================================
    # Analysis Section
    # ==========================================================
    
    # Check if we can analyze
    can_analyze = front_image is not None and back_image is not None
    
    if can_analyze:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Analysis options
        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_button = st.button(
                "เริ่มการวิเคราะห์ด้วย AI", 
                type="primary", 
                use_container_width=True,
                help="คลิกเพื่อเริ่มการวิเคราะห์ด้วย AI"
            )
        with col2:
            auto_enhance = st.checkbox("ปรับปรุงคุณภาพอัตโนมัติ", value=True, help="ปรับปรุงความชัดเจนและสีของรูปภาพ")
        
        # Perform analysis
        if analyze_button:
            st.markdown("## ผลการวิเคราะห์")
            
            with st.spinner("AI กำลังวิเคราะห์รูปภาพ... กรุณารอสักครู่"):
                # Try real API first, then fallback to mock
                if api_healthy:
                    result = frontend.predict_image(front_image, back_image, auto_enhance)
                else:
                    # Mock result for demo
                    time.sleep(2)
                    thai_names = ['พระสมเด็จวัดระฆัง', 'พระนางพญา', 'พระพิมพ์เล็ก', 'พระพิมพ์พุทธคุณ', 'พระไอย์ไข่']
                    result = {
                        'thai_name': np.random.choice(thai_names),
                        'confidence': np.random.uniform(0.75, 0.95),
                        'predicted_class': f'class_{np.random.randint(1, 6)}',
                        'analysis_type': 'dual_image',
                        'processing_time': np.random.uniform(1.2, 2.5),
                        'enhanced_features': {
                            'image_quality': {
                                'overall_score': np.random.uniform(0.7, 0.95),
                                'quality_level': np.random.choice(['good', 'excellent']),
                                'was_enhanced': auto_enhance
                            }
                        }
                    }
                
                # Display results using component
                analysis_results.display_results(result, st.session_state.analysis_mode, show_details=True)
                
                # Show performance metrics if available
                if 'performance' in result:
                    perf = result['performance']
                    st.markdown("### ⚡ Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "API Response Time",
                            f"{perf['api_response_time']:.2f}s",
                            delta=None
                        )
                    with col2:
                        st.metric(
                            "Total Processing",
                            f"{perf['total_processing_time']:.2f}s",
                            delta=None
                        )
                    with col3:
                        # Calculate processing efficiency
                        efficiency = (perf['api_response_time'] / perf['total_processing_time']) * 100
                        st.metric(
                            "Efficiency",
                            f"{efficiency:.1f}%",
                            delta=None
                        )
                
                # Add to history
                if 'error' not in result:
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now(),
                        'result': result,
                        'mode': st.session_state.analysis_mode
                    })
                    
                    # Advanced Analytics - Show trend if multiple analyses
                    if len(st.session_state.analysis_history) > 1:
                        st.markdown("### 📊 Analysis Trends")
                        
                        # Get confidence scores over time
                        confidence_scores = [
                            record['result'].get('confidence', 0) * 100 
                            for record in st.session_state.analysis_history[-10:]  # Last 10
                        ]
                        
                        # Create trend chart
                        if confidence_scores:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=confidence_scores,
                                mode='lines+markers',
                                name='Confidence Trend',
                                line=dict(color='#800000', width=3),
                                marker=dict(size=6, color='#B8860B')
                            ))
                            
                            fig.update_layout(
                                title="Confidence Score Trend",
                                xaxis_title="Analysis #",
                                yaxis_title="Confidence (%)",
                                height=250,
                                showlegend=False,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(family="Prompt, sans-serif"),
                                margin=dict(l=40, r=40, t=40, b=40)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Average Confidence", f"{np.mean(confidence_scores):.1f}%")
                            with col2:
                                st.metric("Best Score", f"{max(confidence_scores):.1f}%")
                            with col3:
                                st.metric("Latest Score", f"{confidence_scores[-1]:.1f}%")
        else:
            st.markdown("""
            <div class="upload-section">
                <h3 style="color: #800000; margin-bottom: 1rem;">พร้อมเริ่มการวิเคราะห์แล้วหรือยัง?</h3>
                <p style="color: #666; font-size: 1.1rem;">
                    กรุณาอัปโหลดรูปภาพทั้งสองด้าน (หน้า และ หลัง) เพื่อเริ่มการวิเคราะห์ด้วย AI
                </p>
                <p style="color: #888; font-size: 0.9rem; margin-top: 1rem;">
                    เคล็ดลับ: รูปภาพที่ชัดเจนและมีแสงสว่างดีจะให้ผลลัพธ์ที่แม่นยำมากขึ้น
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # ==========================================================
    # Analysis History Section
    # ==========================================================
    
    if st.session_state.analysis_history:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## ประวัติการวิเคราะห์")
        
        with st.expander(f"ดูประวัติการวิเคราะห์ ({len(st.session_state.analysis_history)} ครั้ง)", expanded=False):
            for i, record in enumerate(reversed(st.session_state.analysis_history[-5:])):  # Show last 5
                st.markdown(f"""
                <div class="metric-card">
                    <strong>ครั้งที่ {len(st.session_state.analysis_history) - i}</strong> - 
                    {record['timestamp'].strftime('%H:%M:%S')} | 
                    {record['result'].get('thai_name', 'ไม่ระบุ')} 
                    ({record['result'].get('confidence', 0):.1%})
                </div>
                """, unsafe_allow_html=True)
    
    # ==========================================================
    # Tips and Information Section
    # ==========================================================
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## คำแนะนำการใช้งาน")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("เทคนิคการถ่ายรูปที่ดี", expanded=False):
            st.markdown("""
            ### การเตรียมรูปภาพ
            - **แสงสว่าง**: ถ่ายในที่มีแสงสว่างเพียงพอ หลีกเลี่ยงแสงแรงจ้า
            - **ความชัด**: รูปภาพต้องชัดเจน ไม่เบลอ โฟกัสที่พระเครื่อง
            - **มุมมอง**: ถ่ายให้เห็นพระเครื่องทั้งองค์ ไม่ถูกบดบัง
            - **พื้นหลัง**: ใช้พื้นหลังสีเรียบ ไม่มีลวดลายรบกวน
            - **ขนาด**: ขนาดไฟล์ไม่เกิน 10MB, ความละเอียดอย่างน้อย 300x300 พิกเซล
            
            ### ข้อแนะนำเพิ่มเติม
            - ถ่ายรูปในแนวตั้งเพื่อให้เห็นรายละเอียดชัดเจน
            - หลีกเลี่ยงการใช้แฟลช เพราะอาจทำให้เกิดแสงสะท้อน
            - ควรถ่ายรูปใกล้ๆ แต่ให้เห็นขอบพระเครื่องทั้งหมด
            """)
    
    with col2:
        with st.expander("เกี่ยวกับการวิเคราะห์", expanded=False):
            st.markdown("""
            ### เทคโนโลยีที่ใช้
            - **Dual Image Analysis**: ใช้ข้อมูลจากรูปทั้งสองด้านเพื่อความแม่นยำสูงสุด
            - **Deep Learning**: ระบบใช้ CNN ที่ฝึกจากฐานข้อมูลพระเครื่องไทย
            - **Computer Vision**: เทคนิคการมองเห็นของคอมพิวเตอร์ขั้นสูง
            
            ### การตีความผลลัพธ์
            - **Confidence Score**: คะแนนความเชื่อมั่น 0-100%
            - **>90%**: ความเชื่อมั่นสูงมาก แนะนำให้ใช้ผลลัพธ์
            - **70-90%**: ความเชื่อมั่นปานกลาง ควรตรวจสอบเพิ่มเติม
            - **<70%**: ความเชื่อมั่นต่ำ ควรถ่ายรูปใหม่หรือขอคำปรึกษาผู้เชี่ยวชาญ
            
            ### ระยะเวลาการประมวลผล
            - เวลาโดยเฉลี่ย: 2-5 วินาที
            - ขึ้นอยู่กับขนาดไฟล์และความเร็วอินเทอร์เน็ต
            """)
    
    # ==========================================================
    # Performance Dashboard Section
    # ==========================================================
    
    # Update system metrics
    frontend.perf_monitor.update_system_metrics()
    
    # Performance Dashboard
    with st.expander("📊 Performance Dashboard & Analytics", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Performance Chart
            perf_chart = frontend.perf_monitor.get_performance_chart()
            if perf_chart:
                st.plotly_chart(perf_chart, use_container_width=True)
            else:
                st.info("รอข้อมูล Performance...")
        
        with col2:
            # API Statistics
            api_stats = frontend.perf_monitor.get_api_stats()
            
            st.markdown("### 📈 ระบบสถิติ")
            
            # Create metrics
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric(
                    "การวิเคราะห์ทั้งหมด",
                    api_stats['total_requests'],
                    delta=None
                )
                
                if api_stats['avg_response_time'] > 0:
                    st.metric(
                        "เวลาตอบสนอง (วินาที)",
                        f"{api_stats['avg_response_time']:.2f}",
                        delta=f"{api_stats['max_response_time']:.2f} max"
                    )
            
            with col2_2:
                try:
                    cpu_usage = psutil.cpu_percent()
                    memory_usage = psutil.virtual_memory().percent
                    
                    st.metric(
                        "CPU (%)",
                        f"{cpu_usage:.1f}",
                        delta=None
                    )
                    
                    st.metric(
                        "Memory (%)",
                        f"{memory_usage:.1f}",
                        delta=None
                    )
                except:
                    st.info("ไม่สามารถอ่านข้อมูลระบบได้")
            
            # Real-time status indicators
            st.markdown("### 🔴 สถานะ Real-time")
            
            # System status
            status_color = "🟢" if api_healthy else "🔴"
            st.markdown(f"{status_color} **API Status**: {'Online' if api_healthy else 'Offline'}")
            
            # Performance status
            try:
                current_cpu = psutil.cpu_percent()
                if current_cpu < 50:
                    perf_status = "🟢 Normal"
                elif current_cpu < 80:
                    perf_status = "🟡 Medium"
                else:
                    perf_status = "🔴 High"
                st.markdown(f"**System Load**: {perf_status}")
            except:
                st.markdown("**System Load**: ❓ Unknown")
            
            # Memory status
            try:
                current_memory = psutil.virtual_memory().percent
                if current_memory < 60:
                    mem_status = "🟢 Normal"
                elif current_memory < 80:
                    mem_status = "🟡 Medium" 
                else:
                    mem_status = "🔴 High"
                st.markdown(f"**Memory Usage**: {mem_status}")
            except:
                st.markdown("**Memory Usage**: ❓ Unknown")
    
    # ==========================================================
    # Footer Section
    # ==========================================================
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {THEME_COLORS['primary']} 0%, {THEME_COLORS['accent']} 100%);
                color: white; padding: 2rem; border-radius: 15px; text-align: center; margin-top: 3rem;">
        <h3 style="margin: 0 0 1rem 0;">Amulet-AI Project</h3>
        <p style="margin: 0; opacity: 0.9; font-size: 1rem;">
            © 2025 Amulet-AI | Powered by Advanced AI Technology
        </p>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.8; font-size: 0.9rem;">
            สร้างโดยทีมวิจัย AI เพื่อการอนุรักษ์และศึกษาพระเครื่องไทย
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()