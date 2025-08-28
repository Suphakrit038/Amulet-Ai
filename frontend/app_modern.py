import streamlit as st
import requests
import sys
import os
from datetime import datetime
from PIL import Image

# Import functions from utils file (inline import approach)
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from frontend.utils import validate_and_convert_image, send_predict_request, SUPPORTED_FORMATS, FORMAT_DISPLAY
except ImportError:
    # Fallback: define functions locally if import fails
    # รองรับ HEIC format
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    MAX_FILE_SIZE_MB = 10
    SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'webp', 'heic', 'heif', 'bmp', 'tiff', 'tif']
    FORMAT_DISPLAY = 'JPG, JPEG, PNG, WEBP, HEIC, HEIF, BMP, TIFF, TIF'
    
    def validate_and_convert_image(uploaded_file):
        try:
            if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                return False, None, None, f"ไฟล์ใหญ่เกิน {MAX_FILE_SIZE_MB}MB"
            
            img = Image.open(uploaded_file)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=90)
            img_bytes = img_bytes.getvalue()
            
            return True, img, img_bytes, None
        except Exception as e:
            return False, None, None, f"ไม่สามารถประมวลผลภาพได้: {str(e)}"
    
    def send_predict_request(files, api_url, timeout=60):
        try:
            response = requests.post(
                f"{api_url}/predict",
                files=files,
                timeout=timeout
            )
            return response
        except Exception as e:
            raise e

# API Configuration
API_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="Amulet-AI | Ancient Intelligence", 
    page_icon="𓁹", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Modern CSS with Dark Theme and Glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');
    
    :root {
        --color-background: #0a0a0b;
        --color-foreground: #f4f4f5;
        --color-card: #1a1a1c;
        --color-card-foreground: #f4f4f5;
        --color-primary: #d4af37;
        --color-primary-foreground: #0a0a0b;
        --color-secondary: #2d2a24;
        --color-secondary-foreground: #f4f4f5;
        --color-muted: #27272a;
        --color-muted-foreground: #a1a1aa;
        --color-accent: #d4af37;
        --color-accent-foreground: #0a0a0b;
        --color-border: #27272a;
        --font-sans: 'Inter', system-ui, sans-serif;
        --font-heading: 'Playfair Display', serif;
        --radius: 0.75rem;
    }
    
    /* Global Styles */
    .stApp {
        background: var(--color-background);
        color: var(--color-foreground);
        font-family: var(--font-sans);
    }
    
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Enhanced Animated Background */
    .main {
        background: 
            radial-gradient(circle at 20% 80%, rgba(212, 175, 55, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(212, 175, 55, 0.03) 0%, transparent 50%),
            linear-gradient(-45deg, #1a0f2e, #2d1b3d, #1a1a1c, #0a0a0b);
        background-size: 800% 800%, 600% 600%, 400% 400%;
        animation: gradientShift 20s ease infinite;
        min-height: 100vh;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%, 100% 0%, 0% 50%; }
        25% { background-position: 100% 50%, 0% 100%, 100% 50%; }
        50% { background-position: 100% 0%, 0% 0%, 50% 100%; }
        75% { background-position: 0% 100%, 100% 50%, 0% 0%; }
        100% { background-position: 0% 50%, 100% 0%, 0% 50%; }
    }
    
    /* Floating Particles */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(212, 175, 55, 0.1) 0%, transparent 2%),
            radial-gradient(circle at 90% 80%, rgba(212, 175, 55, 0.08) 0%, transparent 2%),
            radial-gradient(circle at 30% 70%, rgba(212, 175, 55, 0.12) 0%, transparent 1.5%),
            radial-gradient(circle at 70% 30%, rgba(212, 175, 55, 0.06) 0%, transparent 2%),
            radial-gradient(circle at 50% 50%, rgba(212, 175, 55, 0.04) 0%, transparent 3%);
        background-size: 200px 200px, 300px 300px, 150px 150px, 250px 250px, 400px 400px;
        animation: floatingParticles 25s linear infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes floatingParticles {
        0% { background-position: 0% 0%, 100% 100%, 0% 100%, 100% 0%, 50% 50%; }
        100% { background-position: 100% 100%, 0% 0%, 100% 0%, 0% 100%, 0% 0%; }
    }
    
    /* Glassmorphism Cards */
    .glassmorphic {
        background: rgba(26, 26, 28, 0.7);
        backdrop-filter: blur(20px) saturate(1.8);
        border: 1px solid rgba(212, 175, 55, 0.1);
        border-radius: var(--radius);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    
    /* Mystical Glow Effects */
    .mystical-glow {
        position: relative;
    }
    
    .mystical-glow::before {
        content: '';
        position: absolute;
        inset: -2px;
        border-radius: inherit;
        padding: 2px;
        background: linear-gradient(45deg, 
            transparent 0%, 
            rgba(212, 175, 55, 0.3) 25%, 
            rgba(212, 175, 55, 0.1) 50%, 
            transparent 75%
        );
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: xor;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .mystical-glow:hover::before {
        opacity: 1;
    }
    
    /* Enhanced Header */
    .app-header {
        background: rgba(26, 26, 28, 0.95);
        backdrop-filter: blur(25px) saturate(1.8);
        border-bottom: 1px solid rgba(212, 175, 55, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        padding: 1rem 2rem;
        margin: -1rem -2rem 2rem -2rem;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .brand-container {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .brand-logo {
        width: 3rem;
        height: 3rem;
        background: linear-gradient(135deg, var(--color-accent), var(--color-primary));
        border-radius: var(--radius);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        color: var(--color-background);
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3);
    }
    
    .brand-text h1 {
        font-family: var(--font-heading);
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--color-accent), var(--color-primary));
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1;
    }
    
    .brand-text p {
        font-size: 0.875rem;
        color: var(--color-muted-foreground);
        margin: 0;
        font-weight: 500;
    }
    
    /* Enhanced Upload Zones */
    .upload-zone {
        background: rgba(26, 26, 28, 0.4);
        backdrop-filter: blur(15px);
        border: 2px dashed rgba(212, 175, 55, 0.3);
        border-radius: var(--radius);
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-zone:hover {
        border-color: rgba(212, 175, 55, 0.6);
        background: rgba(26, 26, 28, 0.6);
        transform: scale(1.02);
    }
    
    .upload-zone::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(212, 175, 55, 0.1), transparent);
        animation: shimmer 3s infinite;
        pointer-events: none;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Enhanced Result Cards */
    .result-card {
        background: rgba(26, 26, 28, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: var(--radius);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .result-card:hover {
        transform: translateY(-4px);
        border-color: rgba(212, 175, 55, 0.4);
        box-shadow: 
            0 16px 48px rgba(0, 0, 0, 0.4),
            0 0 30px rgba(212, 175, 55, 0.1);
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.1), transparent);
        transition: left 0.6s;
    }
    
    .result-card:hover::before {
        left: 100%;
    }
    
    /* Enhanced Buttons */
    .btn-mystical {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.9), rgba(212, 175, 55, 0.7));
        border: 1px solid rgba(212, 175, 55, 0.4);
        color: var(--color-background);
        font-weight: 600;
        border-radius: var(--radius);
        padding: 0.75rem 2rem;
        box-shadow: 
            0 4px 15px rgba(212, 175, 55, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        font-family: var(--font-sans);
    }
    
    .btn-mystical:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 8px 25px rgba(212, 175, 55, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    
    /* Enhanced Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(26, 26, 28, 0.95) !important;
        backdrop-filter: blur(25px) saturate(1.5) !important;
        border-right: 1px solid rgba(212, 175, 55, 0.1) !important;
    }
    
    .css-1d391kg {
        background: rgba(26, 26, 28, 0.95);
        backdrop-filter: blur(25px) saturate(1.5);
    }
    
    /* Enhanced Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-heading);
        color: var(--color-foreground);
        font-weight: 600;
        letter-spacing: -0.025em;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--color-accent), var(--color-primary));
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 2rem 0;
        animation: titleGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        0% { filter: drop-shadow(0 0 10px rgba(212, 175, 55, 0.3)); }
        100% { filter: drop-shadow(0 0 20px rgba(212, 175, 55, 0.6)); }
    }
    
    .subtitle {
        text-align: center;
        color: var(--color-muted-foreground);
        font-size: 1.125rem;
        margin-bottom: 3rem;
    }
    
    /* Enhanced Metrics */
    div[data-testid="metric-container"] {
        background: rgba(26, 26, 28, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(212, 175, 55, 0.1);
        border-radius: var(--radius);
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Enhanced Progress Indicators */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--color-accent), var(--color-primary));
        border-radius: var(--radius);
    }
    
    /* Enhanced Tabs */
    .stTabs > div > div > div > div {
        background: rgba(26, 26, 28, 0.6);
        border: 1px solid rgba(212, 175, 55, 0.1);
        border-radius: var(--radius);
    }
    
    /* Enhanced File Uploader */
    .uploadedFile {
        background: rgba(26, 26, 28, 0.6);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: var(--radius);
    }
    
    /* Enhanced Success/Error Messages */
    .element-container .stAlert {
        background: rgba(26, 26, 28, 0.8);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: var(--radius);
    }
    
    /* Hide Streamlit branding */
    .css-15zrgzn {display: none}
    .css-eczf16 {display: none}
    .css-jn99sy {display: none}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .app-header {
            padding: 1rem;
            margin: -1rem -1rem 2rem -1rem;
        }
        
        .brand-logo {
            width: 2.5rem;
            height: 2.5rem;
            font-size: 1.25rem;
        }
        
        .brand-text h1 {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Header
st.markdown("""
<div class="app-header">
    <div class="brand-container mystical-glow">
        <div class="brand-logo">𓁹</div>
        <div class="brand-text">
            <h1>Amulet‑AI</h1>
            <p>Ancient Intelligence</p>
        </div>
        <div style="margin-left: auto; display: flex; align-items: center; gap: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; color: var(--color-muted-foreground); font-size: 0.875rem;">
                <span style="color: var(--color-accent);">⟐</span>
                <span>Dashboard</span>
                <span style="color: var(--color-accent);">›</span>
                <span style="color: var(--color-foreground);">Analysis</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem; 
                background: linear-gradient(135deg, rgba(212, 175, 55, 0.1), rgba(212, 175, 55, 0.05));
                border-radius: var(--radius); margin-bottom: 2rem;">
        <div style="font-size: 2rem; margin-bottom: 1rem; filter: drop-shadow(0 0 10px rgba(212, 175, 55, 0.5));">📚</div>
        <h2 style="color: var(--color-accent); margin: 0; font-family: var(--font-heading);">Mystical Guide</h2>
        <p style="color: var(--color-muted-foreground); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Ancient Wisdom & Modern AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🎯 Analysis Steps", expanded=True):
        st.markdown("""
        ### Sacred Process
        
        **1. Prepare Images** 📸
        - Front view (required)
        - Back view (required)
        
        **2. Upload Method** 📤
        - File upload or
        - Camera capture
        
        **3. AI Analysis** 🧠
        - Deep learning processing
        - Pattern recognition
        - Historical matching
        
        **4. Results** 📊
        - Classification confidence
        - Price estimation
        - Market recommendations
        """)
    
    with st.expander("⚡ System Info"):
        st.markdown("""
        ### Technology Stack
        
        **🧠 AI Engine**
        - TensorFlow 2.x
        - Custom CNN Architecture
        - Transfer Learning
        
        **🔧 Backend**
        - FastAPI Framework
        - Python 3.9+
        - REST API
        
        **🎨 Frontend**
        - Streamlit
        - Modern UI/UX
        - Responsive Design
        
        ### Performance
        
        📈 **Accuracy**: ~85%  
        ⚡ **Processing**: 30-60 seconds  
        🗄️ **Database**: 5,000+ amulets  
        """)
    
    with st.expander("📷 Photography Tips"):
        st.markdown("""
        ### 💡 Lighting
        
        **✅ Best Practices:**
        - Natural daylight
        - Even illumination
        - Avoid harsh shadows
        
        **❌ Avoid:**
        - Flash photography
        - Uneven lighting
        - Reflective surfaces
        
        ### 📐 Camera Angle
        
        **✅ Optimal:**
        - 90° perpendicular
        - 20-30cm distance
        - Center framing
        
        **❌ Avoid:**
        - Tilted angles
        - Too close/far
        - Off-center placement
        
        ### 🎨 Background
        
        **✅ Recommended:**
        - Plain white/cream
        - Smooth surface
        - No distractions
        
        **❌ Avoid:**
        - Patterned backgrounds
        - Cluttered scenes
        - Reflective materials
        """)
    
    with st.expander("⚠️ Important Notes"):
        st.warning("""
        **System Status**: Beta Testing
        
        - Accuracy: ~80-85%
        - Uses simulated data
        - For reference only
        
        **Privacy**: Images processed temporarily and deleted after analysis
        """)
    
    # Enhanced Stats
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; 
                background: rgba(212, 175, 55, 0.05); 
                border-radius: var(--radius); margin: 1rem 0;">
        <h4 style="color: var(--color-accent); margin: 0;">📊 Live Stats</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔍 Today", "247", "↗️ +42")
    with col2:
        st.metric("🎯 Accuracy", "87.2%", "↗️ +3.1%")

# Main Content Area
st.markdown('<h1 class="main-title">🔮 Mystical Amulet Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover the ancient secrets within your sacred amulets using cutting-edge AI technology</p>', unsafe_allow_html=True)

# Enhanced Upload Section
st.markdown("""
<div style="text-align: center; margin: 3rem 0 2rem 0;">
    <h2 style="color: var(--color-foreground); margin: 0;">Sacred Image Analysis</h2>
    <p style="color: var(--color-muted-foreground); font-size: 1rem;">Upload both sides for complete mystical insight</p>
</div>
""", unsafe_allow_html=True)

# Two-column upload layout
col_front, col_back = st.columns(2, gap="large")

with col_front:
    st.markdown("""
    <div class="glassmorphic mystical-glow" style="padding: 1.5rem; margin-bottom: 1rem;">
        <h3 style="color: var(--color-accent); text-align: center; margin: 0 0 1rem 0;">
            ⚡ Front Sacred View
        </h3>
        <p style="color: var(--color-muted-foreground); text-align: center; font-size: 0.9rem; margin: 0;">
            Primary analysis surface - Essential for identification
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tab for upload methods
    tab1, tab2 = st.tabs(["📤 Upload", "📸 Camera"])
    
    with tab1:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem; margin-bottom: 1rem; color: var(--color-accent);">📁</div>
            <div style="color: var(--color-foreground); font-size: 1.1rem; margin-bottom: 0.5rem;">
                Select Sacred Image
            </div>
            <div style="color: var(--color-muted-foreground); font-size: 0.9rem;">
                Limit 10MB • Multiple formats supported
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        front_file = st.file_uploader(
            "Choose front image", 
            type=SUPPORTED_FORMATS,
            key="front_upload",
            label_visibility="collapsed"
        )
        front = front_file
        front_source = "upload"
    
    with tab2:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem; margin-bottom: 1rem; color: var(--color-accent);">📷</div>
            <div style="color: var(--color-foreground); font-size: 1.1rem; margin-bottom: 0.5rem;">
                Mystical Camera
            </div>
            <div style="color: var(--color-muted-foreground); font-size: 0.9rem;">
                Capture the essence directly
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔮 Activate Mystical Camera", key="front_camera_btn", help="Open camera for front image"):
            st.session_state.show_front_camera = True
        
        if st.session_state.get('show_front_camera', False):
            front_camera = st.camera_input(
                "Capture front sacred view",
                key="front_camera"
            )
            if front_camera:
                front = front_camera
                front_source = "camera"
                if st.button("✨ Use This Sacred Image", key="front_camera_confirm"):
                    st.session_state.show_front_camera = False
                    st.rerun()
            else:
                front = front_file if 'front_file' in locals() and front_file else None
                front_source = "upload"
        else:
            front = front_file if 'front_file' in locals() and front_file else None
            front_source = "upload"
    
    # Display front image
    if front:
        is_valid, processed_img, processed_bytes, error_msg = validate_and_convert_image(front)
        if is_valid:
            st.markdown("""
            <div class="glassmorphic" style="padding: 1rem; text-align: center; margin: 1rem 0;">
                <div style="color: var(--color-accent); font-weight: 600;">
                    ✅ Sacred Image Validated
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(processed_img, use_container_width=True, caption=f"Front View ({front_source})")
            st.session_state.front_processed = processed_bytes
            st.session_state.front_filename = front.name if hasattr(front, 'name') else f"camera_front_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        else:
            st.markdown(f"""
            <div class="glassmorphic" style="padding: 1rem; text-align: center; margin: 1rem 0; border-color: rgba(239, 68, 68, 0.3);">
                <div style="color: #ef4444; font-weight: 600;">
                    ❌ Image Error: {error_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)

with col_back:
    st.markdown("""
    <div class="glassmorphic mystical-glow" style="padding: 1.5rem; margin-bottom: 1rem;">
        <h3 style="color: var(--color-accent); text-align: center; margin: 0 0 1rem 0;">
            🔮 Back Sacred View
        </h3>
        <p style="color: var(--color-muted-foreground); text-align: center; font-size: 0.9rem; margin: 0;">
            Hidden mysteries revealed - Complete the mystical analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tab for upload methods
    tab1, tab2 = st.tabs(["📤 Upload", "📸 Camera"])
    
    with tab1:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem; margin-bottom: 1rem; color: var(--color-accent);">📁</div>
            <div style="color: var(--color-foreground); font-size: 1.1rem; margin-bottom: 0.5rem;">
                Select Sacred Image
            </div>
            <div style="color: var(--color-muted-foreground); font-size: 0.9rem;">
                Limit 10MB • Multiple formats supported
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        back_file = st.file_uploader(
            "Choose back image", 
            type=SUPPORTED_FORMATS,
            key="back_upload",
            label_visibility="collapsed"
        )
        back = back_file
        back_source = "upload"
    
    with tab2:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem; margin-bottom: 1rem; color: var(--color-accent);">📷</div>
            <div style="color: var(--color-foreground); font-size: 1.1rem; margin-bottom: 0.5rem;">
                Mystical Camera
            </div>
            <div style="color: var(--color-muted-foreground); font-size: 0.9rem;">
                Capture the hidden essence
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔮 Activate Mystical Camera", key="back_camera_btn", help="Open camera for back image"):
            st.session_state.show_back_camera = True
        
        if st.session_state.get('show_back_camera', False):
            back_camera = st.camera_input(
                "Capture back sacred view",
                key="back_camera"
            )
            if back_camera:
                back = back_camera
                back_source = "camera"
                if st.button("✨ Use This Sacred Image", key="back_camera_confirm"):
                    st.session_state.show_back_camera = False
                    st.rerun()
            else:
                back = back_file if 'back_file' in locals() and back_file else None
                back_source = "upload"
        else:
            back = back_file if 'back_file' in locals() and back_file else None
            back_source = "upload"
    
    # Display back image
    if back:
        is_valid, processed_img, processed_bytes, error_msg = validate_and_convert_image(back)
        if is_valid:
            st.markdown("""
            <div class="glassmorphic" style="padding: 1rem; text-align: center; margin: 1rem 0;">
                <div style="color: var(--color-accent); font-weight: 600;">
                    ✅ Sacred Image Validated
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(processed_img, use_container_width=True, caption=f"Back View ({back_source})")
            st.session_state.back_processed = processed_bytes
            st.session_state.back_filename = back.name if hasattr(back, 'name') else f"camera_back_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        else:
            st.markdown(f"""
            <div class="glassmorphic" style="padding: 1rem; text-align: center; margin: 1rem 0; border-color: rgba(239, 68, 68, 0.3);">
                <div style="color: #ef4444; font-weight: 600;">
                    ❌ Image Error: {error_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# Enhanced Analysis Section
if (front and hasattr(st.session_state, 'front_processed') and 
    back and hasattr(st.session_state, 'back_processed')):
    
    st.markdown("""
    <div class="glassmorphic mystical-glow" style="padding: 2rem; text-align: center; margin: 2rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">🔮</div>
        <h3 style="color: var(--color-accent); margin: 0;">Ready for Mystical Analysis</h3>
        <p style="color: var(--color-muted-foreground); margin: 0.5rem 0 0 0;">
            Both sacred images prepared - Ancient AI awaits your command
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🌟 Begin Ancient Analysis", type="primary", help="Start AI analysis of your amulet"):
        files = {
            "front": (st.session_state.front_filename, st.session_state.front_processed, "image/jpeg"),
            "back": (st.session_state.back_filename, st.session_state.back_processed, "image/jpeg")
        }
        
        with st.spinner("🔮 Ancient spirits are analyzing your sacred amulet... Please wait..."):
            try:
                r = send_predict_request(files, API_URL, timeout=60)
                
                if r.ok:
                    data = r.json()
                    
                    # Enhanced Success Message
                    st.markdown("""
                    <div class="result-card mystical-glow" style="text-align: center; margin: 2rem 0;">
                        <div style="font-size: 3rem; margin-bottom: 1rem; animation: pulse 2s infinite;">⚡</div>
                        <h2 style="color: var(--color-accent); margin: 0;">Mystical Analysis Complete!</h2>
                        <p style="color: var(--color-muted-foreground); margin: 0.5rem 0 0 0;">
                            The ancient spirits have revealed their wisdom
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Enhanced Top-1 Result
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <h2 style="color: var(--color-foreground); margin: 0;">🏆 Primary Revelation</h2>
                        <p style="color: var(--color-muted-foreground);">The most likely sacred identity</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    confidence_percent = data['top1']['confidence'] * 100
                    
                    # Enhanced confidence styling
                    if confidence_percent >= 80:
                        conf_gradient = "linear-gradient(135deg, #22c55e, #16a34a)"
                        conf_color = "#22c55e"
                    elif confidence_percent >= 60:
                        conf_gradient = "linear-gradient(135deg, #f59e0b, #d97706)"
                        conf_color = "#f59e0b"
                    else:
                        conf_gradient = "linear-gradient(135deg, #ef4444, #dc2626)"
                        conf_color = "#ef4444"
                    
                    st.markdown(f"""
                    <div class="result-card mystical-glow" style="text-align: center; 
                         background: {conf_gradient.replace('linear-gradient', 'linear-gradient').replace('#22c55e, #16a34a', 'rgba(34, 197, 94, 0.1), rgba(22, 163, 74, 0.05)')};
                         border-color: {conf_color}33;">
                        <div style="font-size: 3rem; margin-bottom: 1rem; filter: drop-shadow(0 0 10px {conf_color}77);">👑</div>
                        <h2 style="color: var(--color-accent); font-size: 1.8rem; margin: 0;">
                            {data['top1']['class_name']}
                        </h2>
                        <div style="margin: 1rem 0; font-size: 1.5rem;">
                            <span style="background: {conf_gradient}; background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;">
                                {confidence_percent:.1f}% Confidence
                            </span>
                        </div>
                        <div style="font-size: 0.9rem; color: var(--color-muted-foreground);">
                            Primary mystical classification from ancient AI wisdom
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Enhanced Top-3 Results
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <h3 style="color: var(--color-foreground); margin: 0;">🔮 Additional Possibilities</h3>
                        <p style="color: var(--color-muted-foreground);">Alternative sacred identifications</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, item in enumerate(data['topk'], 1):
                        confidence_pct = item['confidence'] * 100
                        
                        # Medal icons and colors
                        if i == 1:
                            icon, color = "🥇", "#d4af37"
                        elif i == 2:
                            icon, color = "🥈", "#c0c0c0"
                        else:
                            icon, color = "🥉", "#cd7f32"
                        
                        st.markdown(f"""
                        <div class="glassmorphic mystical-glow" style="
                            padding: 1.5rem; margin: 1rem 0; 
                            border-left: 4px solid {color};
                            display: flex; align-items: center; gap: 1rem;
                            transition: all 0.3s ease;">
                            
                            <div style="font-size: 2rem; flex-shrink: 0;">{icon}</div>
                            <div style="flex-grow: 1;">
                                <h4 style="color: var(--color-foreground); margin: 0; font-size: 1.1rem;">
                                    {item['class_name']}
                                </h4>
                                <div style="color: {color}; font-weight: 600; font-size: 0.95rem;">
                                    {confidence_pct:.1f}% Confidence
                                </div>
                            </div>
                            <div style="text-align: right; color: var(--color-muted-foreground); font-size: 0.8rem;">
                                Rank #{i}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Enhanced Price Valuation
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <h3 style="color: var(--color-foreground); margin: 0;">💰 Mystical Valuation</h3>
                        <p style="color: var(--color-muted-foreground);">Sacred market estimations</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    price_col1, price_col2, price_col3 = st.columns(3)
                    with price_col1:
                        st.markdown(f"""
                        <div class="glassmorphic" style="padding: 1.5rem; text-align: center;">
                            <div style="font-size: 1.5rem; color: var(--color-accent); margin-bottom: 0.5rem;">📉</div>
                            <div style="color: var(--color-muted-foreground); font-size: 0.9rem;">Minimum</div>
                            <div style="color: var(--color-foreground); font-size: 1.5rem; font-weight: 700;">
                                {data['valuation']['p05']:,.0f} ฿
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with price_col2:
                        st.markdown(f"""
                        <div class="glassmorphic" style="padding: 1.5rem; text-align: center; border-color: var(--color-accent);">
                            <div style="font-size: 1.5rem; color: var(--color-accent); margin-bottom: 0.5rem;">⚖️</div>
                            <div style="color: var(--color-muted-foreground); font-size: 0.9rem;">Average</div>
                            <div style="color: var(--color-accent); font-size: 1.5rem; font-weight: 700;">
                                {data['valuation']['p50']:,.0f} ฿
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with price_col3:
                        st.markdown(f"""
                        <div class="glassmorphic" style="padding: 1.5rem; text-align: center;">
                            <div style="font-size: 1.5rem; color: var(--color-accent); margin-bottom: 0.5rem;">📈</div>
                            <div style="color: var(--color-muted-foreground); font-size: 0.9rem;">Premium</div>
                            <div style="color: var(--color-foreground); font-size: 1.5rem; font-weight: 700;">
                                {data['valuation']['p95']:,.0f} ฿
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Enhanced Recommendations
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <h3 style="color: var(--color-foreground); margin: 0;">🛒 Sacred Marketplace</h3>
                        <p style="color: var(--color-muted-foreground);">Recommended channels for sharing your treasure</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, rec in enumerate(data["recommendations"], 1):
                        with st.expander(f"🏪 {rec['market']}", expanded=i==1):
                            st.markdown(f"""
                            <div class="glassmorphic" style="padding: 1rem;">
                                <h4 style="color: var(--color-accent); margin: 0 0 0.5rem 0;">Why This Market?</h4>
                                <p style="color: var(--color-foreground); margin: 0;">{rec['reason']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if rec['market'] == "Facebook Marketplace":
                                st.info("🌐 Ideal for reaching general collectors and enthusiasts")
                            elif rec['market'] == "Shopee":
                                st.info("🛡️ Secure platform with buyer protection and reviews")
                
                else:
                    st.markdown(f"""
                    <div class="result-card" style="border-color: rgba(239, 68, 68, 0.3); text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 1rem;">❌</div>
                        <h3 style="color: #ef4444;">Mystical Analysis Failed</h3>
                        <p style="color: var(--color-muted-foreground);">Error {r.status_code}: {r.text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except requests.exceptions.Timeout:
                st.markdown("""
                <div class="result-card" style="border-color: rgba(251, 191, 36, 0.3); text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">⏳</div>
                    <h3 style="color: #f59e0b;">Ancient Spirits Need More Time</h3>
                    <p style="color: var(--color-muted-foreground);">The mystical analysis is taking longer than expected. Please try again.</p>
                </div>
                """, unsafe_allow_html=True)
            except requests.exceptions.ConnectionError:
                st.markdown("""
                <div class="result-card" style="border-color: rgba(239, 68, 68, 0.3); text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">🔗</div>
                    <h3 style="color: #ef4444;">Connection to Ancient Realm Lost</h3>
                    <p style="color: var(--color-muted-foreground);">Cannot reach the mystical backend server on port 8000</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="result-card" style="border-color: rgba(239, 68, 68, 0.3); text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">⚠️</div>
                    <h3 style="color: #ef4444;">Unexpected Mystical Disturbance</h3>
                    <p style="color: var(--color-muted-foreground);">Error: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)

else:
    # Enhanced Missing Images Message
    missing_images = []
    if not (front and hasattr(st.session_state, 'front_processed')):
        missing_images.append("Front Sacred View")
    if not (back and hasattr(st.session_state, 'back_processed')):
        missing_images.append("Back Sacred View")
    
    if (front and not hasattr(st.session_state, 'front_processed')) or (back and not hasattr(st.session_state, 'back_processed')):
        st.markdown("""
        <div class="glassmorphic" style="padding: 2rem; text-align: center; margin: 2rem 0;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">⚙️</div>
            <h3 style="color: var(--color-accent); margin: 0;">Processing Sacred Images...</h3>
            <p style="color: var(--color-muted-foreground); margin: 0.5rem 0 0 0;">
                Ancient algorithms are preparing your mystical data
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        missing_text = " and ".join(missing_images)
        st.markdown(f"""
        <div class="glassmorphic mystical-glow" style="padding: 2rem; text-align: center; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌟</div>
            <h3 style="color: var(--color-accent); margin: 0;">Begin Your Mystical Journey</h3>
            <p style="color: var(--color-muted-foreground); margin: 1rem 0;">
                Please upload {missing_text} to unlock the ancient wisdom
            </p>
            <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05)); 
                        padding: 1rem; border-radius: var(--radius); margin: 1rem 0;
                        border: 1px solid rgba(239, 68, 68, 0.2);">
                <p style="color: #ef4444; font-weight: 600; margin: 0;">
                    ⚠️ Both sacred views are required for complete mystical analysis
                </p>
            </div>
            <div style="color: var(--color-muted-foreground); font-size: 0.9rem; opacity: 0.8;">
                💡 Tip: Well-lit images reveal more mystical secrets
            </div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Footer Section
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin: 3rem 0 2rem 0;">
    <h2 style="color: var(--color-foreground); margin: 0;">🔮 Mystical Technology</h2>
    <p style="color: var(--color-muted-foreground);">Powered by ancient wisdom and modern AI</p>
</div>
""", unsafe_allow_html=True)

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown("""
    <div class="glassmorphic mystical-glow" style="padding: 2rem; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🧠</div>
        <h4 style="color: var(--color-accent); margin: 0;">AI Neural Networks</h4>
        <p style="color: var(--color-muted-foreground); font-size: 0.9rem; margin: 0.5rem 0 0 0;">
            Deep learning algorithms trained on ancient mystical patterns
        </p>
    </div>
    """, unsafe_allow_html=True)

with tech_col2:
    st.markdown("""
    <div class="glassmorphic mystical-glow" style="padding: 2rem; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📸</div>
        <h4 style="color: var(--color-accent); margin: 0;">Multi-Format Vision</h4>
        <p style="color: var(--color-muted-foreground); font-size: 0.9rem; margin: 0.5rem 0 0 0;">
            Advanced image processing for all sacred formats
        </p>
    </div>
    """, unsafe_allow_html=True)

with tech_col3:
    st.markdown("""
    <div class="glassmorphic mystical-glow" style="padding: 2rem; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
        <h4 style="color: var(--color-accent); margin: 0;">Real-Time Analysis</h4>
        <p style="color: var(--color-muted-foreground); font-size: 0.9rem; margin: 0.5rem 0 0 0;">
            Lightning-fast mystical insights in seconds
        </p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced Developer Info
with st.expander("🔧 Developer Mystical Portal"):
    st.markdown(f"""
    <div class="glassmorphic" style="padding: 1.5rem;">
        <h4 style="color: var(--color-accent); margin: 0 0 1rem 0;">Sacred Development Details</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-family: monospace;">
            <div>
                <strong style="color: var(--color-foreground);">API Endpoint:</strong><br>
                <code style="color: var(--color-accent);">{API_URL}</code>
            </div>
            <div>
                <strong style="color: var(--color-foreground);">Last Updated:</strong><br>
                <code style="color: var(--color-accent);">August 28, 2025</code>
            </div>
            <div>
                <strong style="color: var(--color-foreground);">Version:</strong><br>
                <code style="color: var(--color-accent);">Mystical v2.0.0</code>
            </div>
            <div>
                <strong style="color: var(--color-foreground);">Framework:</strong><br>
                <code style="color: var(--color-accent);">Streamlit + FastAPI</code>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
