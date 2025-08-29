
import streamlit as st
import requests
import sys
import os
from datetime import datetime
from PIL import Image

# Small helper to render reusable upload cards
def upload_card(title_th, key, help_text=""):
    st.markdown(f"""
    <div class="card" style="text-align:center; margin-bottom:0.75rem;">
        <h3 style="margin:0 0 .5rem 0; color: var(--card-fg);">{title_th}</h3>
        <p style="margin:0; color: var(--muted); font-size:0.9rem;">{help_text}</p>
    </div>
    """, unsafe_allow_html=True)
    file = st.file_uploader("", type=SUPPORTED_FORMATS, key=key, label_visibility="collapsed")
    return file

# Import custom components
try:
    from components.ui import (
        mystical_header,
        mystical_card,
        mystical_progress,
        mystical_alert,
        confidence_indicator,
        result_display_card,
        create_sidebar_navigation
    )
except ImportError:
    # Fallback if components are not available
    pass

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
API_URL = os.getenv("AMULET_API_URL", "http://localhost:8000")

# Page Configuration
st.set_page_config(
    page_title="Amulet-AI | Ancient Intelligence", 
    page_icon="𓁹", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Modern CSS theme (single palette)
st.markdown("""
<style>
:root{
    --bg:#fefdfc; --fg:#1b1b1f; --muted:#6b7280;
    --card:#2c2b31; --card-fg:#ffffff;
    --accent:#e6aa33; --banner:#2b314c; --surface:#b7bee7;
    --radius:14px;
    --success:#22c55e; --warning:#f59e0b; --danger:#ef4444;
    --medal-gold:#d4af37; --medal-silver:#c0c0c0; --medal-bronze:#cd7f32;

    /* Aliases */
    --color-background:var(--bg);
    --color-foreground:var(--fg);
    --color-muted-foreground:var(--muted);
    --color-card:var(--card);
    --color-accent:var(--accent);

    /* Confidence gradients */
    --success-grad:linear-gradient(135deg,var(--success), color-mix(in srgb,var(--success) 70%, black));
    --warning-grad:linear-gradient(135deg,var(--warning), color-mix(in srgb,var(--warning) 70%, black));
    --danger-grad: linear-gradient(135deg,var(--danger), color-mix(in srgb,var(--danger) 70%, black));
}

.stApp{ background:var(--bg); color:var(--fg); font-family:'Inter',system-ui,sans-serif; }
section[data-testid="stSidebar"]{ background:var(--banner)!important; color:#fff!important; }
h1,h2,h3,h4{ font-family: 'Playfair Display', serif; }
.card{ background:var(--card); color:var(--card-fg); border:1px solid #ffffff1f; border-radius:var(--radius); padding:1rem; }
.glassmorphic{ background:rgba(44,43,49,.96); color:var(--card-fg); border:1px solid color-mix(in srgb,var(--accent) 18%,transparent); border-radius:var(--radius); }
.textarea-surface{ background:var(--surface); border-radius:var(--radius); padding:.75rem; }
.btn-accent{ background:var(--accent); color:#261a00; border:none; border-radius:var(--radius); padding:.65rem 1rem; font-weight:600; }
.file-drop{ border:2px dashed color-mix(in srgb,var(--accent) 55%,transparent); border-radius:var(--radius); }
.upload-zone{ border:2px dashed color-mix(in srgb,var(--accent) 35%,transparent); border-radius:var(--radius); padding:1.25rem; }
.result-card{ background:#fff; border:1px solid color-mix(in srgb,var(--accent) 8%,transparent); border-radius:var(--radius); }
#MainMenu, footer, header { visibility: hidden; }
@media (max-width:768px){ .main-title{ font-size:2rem; } }
/* Helpers */
.muted{ color:var(--color-muted-foreground); }
.accent{ color:var(--color-accent); }
.success{ color:var(--success); }
.warning{ color:var(--warning); }
.danger{ color:var(--danger); }
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
                background: linear-gradient(135deg, rgba(230,170,51,0.1), rgba(230,170,51,0.05));
                border-radius: var(--radius); margin-bottom: 2rem;">
        <h2 style="color: var(--color-accent); margin: 0; font-family: var(--font-heading);">คู่มือการใช้งาน</h2>
        <p style="color: var(--color-muted-foreground); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
    <h2 style="color: var(--color-accent); margin: 0; font-family: var(--font-heading); font-weight: 700;">คู่มือการใช้งาน</h2>
    <p style="color: var(--color-muted-foreground); margin: 0.5rem 0 0 0; font-size: 0.9rem; font-weight: 700;">
            ปัญญาโบราณ & AI ยุคใหม่
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ขั้นตอนการวิเคราะห์", expanded=True):
        st.markdown("""
        ### ขั้นตอนศักดิ์สิทธิ์
        
        **1. เตรียมรูปภาพ**
        - ด้านหน้า (จำเป็น)
        - ด้านหลัง (จำเป็น)
        
        **2. วิธีอัปโหลด**
        - อัปโหลดไฟล์ หรือ
        - ถ่ายภาพด้วยกล้อง
        
        **3. วิเคราะห์ด้วย AI**
        - ประมวลผลด้วย Deep Learning
        - ตรวจจับลวดลาย
        - เทียบกับฐานข้อมูลประวัติศาสตร์
        
        **4. ผลลัพธ์**
        - ความมั่นใจในการจำแนก
        - ประเมินราคา
        - แนะนำตลาด
        """)

    with st.expander("ข้อมูลระบบ"):
        st.markdown("""
        ### เทคโนโลยีที่ใช้
        
        **AI Engine**
        - TensorFlow 2.x
        - โครงข่ายประสาทเทียมเฉพาะทาง
        - Transfer Learning
        
        **ฝั่งเซิร์ฟเวอร์**
        - FastAPI Framework
        - Python 3.9+
        - REST API
        
        **ฝั่งผู้ใช้**
        - Streamlit
        - UI/UX ทันสมัย
        - รองรับทุกอุปกรณ์
        
        ### ประสิทธิภาพ
        
        ความแม่นยำ: ~85%
        ประมวลผล: 30-60 วินาที
        ฐานข้อมูล: 5,000+ รายการ
        """)

    with st.expander("เคล็ดลับการถ่ายภาพ"):
        st.markdown("""
        ### แสงสว่าง
        
        ควรทำ:
        - ใช้แสงธรรมชาติ
        - แสงสม่ำเสมอทั่วทั้งวัตถุ
        - หลีกเลี่ยงเงาเข้ม
        
        หลีกเลี่ยง:
        - ใช้แฟลช
        - แสงไม่สม่ำเสมอ
        - พื้นผิวสะท้อนแสง
        
        ### มุมกล้อง
        
        เหมาะสม:
        - ถ่ายตรง 90°
        - ระยะห่าง 20-30 ซม.
        - วางวัตถุไว้กลางภาพ
        
        หลีกเลี่ยง:
        - มุมเอียง
        - ใกล้/ไกลเกินไป
        - วางวัตถุไม่ตรงกลาง
        
        ### พื้นหลัง
        
        แนะนำ:
        - สีขาว/ครีม เรียบ
        - พื้นผิวเรียบ
        - ไม่มีสิ่งรบกวน
        
        หลีกเลี่ยง:
        - พื้นหลังลาย
        - ฉากรก
        - วัสดุสะท้อนแสง
        """)

    with st.expander("หมายเหตุสำคัญ"):
        st.warning("""
        สถานะระบบ: ทดสอบเบต้า
        
        - ความแม่นยำ: ~80-85%
        - ใช้ข้อมูลจำลองบางส่วน
        - ผลลัพธ์เพื่อการอ้างอิงเท่านั้น
        
        ความเป็นส่วนตัว: ภาพจะถูกลบหลังวิเคราะห์เสร็จ
        """)

    # Enhanced Stats
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; 
                background: rgba(230,170,51,0.05); 
                border-radius: var(--radius); margin: 1rem 0;">
        <h4 style="color: var(--color-accent); margin: 0;">สถิติการใช้งาน</h4>
    <h4 style="color: var(--color-accent); margin: 0; font-weight: 700;">สถิติการใช้งาน</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("วันนี้", "247", "+42")
    with col2:
        st.metric("ความแม่นยำ", "87.2%", "+3.1%")

# Main Content Area
st.markdown('<h1 class="main-title">ระบบวิเคราะห์พระเครื่องอัตโนมัติ</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ค้นพบความลับในพระเครื่องของคุณด้วย AI อัจฉริยะ</p>', unsafe_allow_html=True)
st.markdown("""
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 2.5rem; margin-bottom: 2.5rem;">
    <div style="
    background: rgba(255,255,255,0.18);
    box-shadow: 0 8px 32px 0 color-mix(in srgb,var(--accent) 10%, rgba(31,38,135,0.08));
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 32px;
    border: 1.5px solid color-mix(in srgb,var(--accent) 40%, transparent);
    padding: 2.5rem 2rem 2rem 2rem;
    max-width: 600px;
    width: 100%;
    text-align: center;">
        <h1 style="
            color: var(--color-accent);
            font-size: 2.6rem;
            font-weight: 800;
            margin: 0 0 1rem 0;
            letter-spacing: 1px;">
            ระบบวิเคราะห์พระเครื่องอัตโนมัติ
        </h1>
        <p style="
            color: var(--color-foreground);
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: 0.5px;">
            ค้นพบความลับในพระเครื่องของคุณด้วย AI อัจฉริยะ
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced Upload Section
st.markdown("""
<div style="text-align: center; margin: 3rem 0 2rem 0;">
    <h2 style="color: var(--color-foreground); margin: 0;">วิเคราะห์ภาพพระเครื่อง</h2>
    <p style="color: var(--color-muted-foreground); font-size: 1rem;">อัปโหลดทั้งสองด้านเพื่อการวิเคราะห์ที่สมบูรณ์</p>
    <h2 style="color: var(--card-fg); margin: 0; font-weight: 700;">วิเคราะห์ภาพพระเครื่อง</h2>
    <p style="color: var(--color-muted-foreground); font-size: 1rem; font-weight: 700;">อัปโหลดทั้งสองด้านเพื่อการวิเคราะห์ที่สมบูรณ์</p>
</div>
""", unsafe_allow_html=True)

# Two-column upload layout
col_front, col_back = st.columns(2, gap="large")

with col_front:
    st.markdown("""
    <div class="glassmorphic mystical-glow" style="padding: 1.5rem; margin-bottom: 1rem;">
        <h3 style="color: var(--color-accent); text-align: center; margin: 0 0 1rem 0;">
    <h3 style="color: var(--color-accent); text-align: center; margin: 0 0 1rem 0; font-weight: 700;">
            ด้านหน้าพระเครื่อง
        </h3>
        <p style="color: var(--color-muted-foreground); text-align: center; font-size: 0.9rem; margin: 0;">
    <p style="color: var(--color-muted-foreground); text-align: center; font-size: 0.9rem; margin: 0; font-weight: 700;">
            พื้นที่หลักสำหรับวิเคราะห์ จำเป็นต้องใช้
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tab for upload methods
    tab1, tab2 = st.tabs(["อัปโหลดไฟล์", "ถ่ายภาพ"])

    with tab1:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 2.2rem; margin-bottom: 1rem; color: var(--color-accent);">เลือกไฟล์</div>
            <div style="color: var(--color-foreground); font-size: 1.1rem; margin-bottom: 0.5rem;">
            <div style="font-size: 2.2rem; margin-bottom: 1rem; color: var(--color-accent); font-weight: 700;">เลือกไฟล์</div>
            <div style="color: var(--card-fg); font-size: 1.1rem; margin-bottom: 0.5rem; font-weight: 700;">
                เลือกภาพด้านหน้า
            </div>
            <div style="color: var(--color-muted-foreground); font-size: 0.9rem;">
            <div style="color: var(--color-muted-foreground); font-size: 0.9rem; font-weight: 700;">
                จำกัด 10MB • รองรับหลายฟอร์แมต
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
    <h3 style="color: var(--color-accent); text-align: center; margin: 0 0 1rem 0; font-weight: 700;">
            ด้านหลังพระเครื่อง
        </h3>
        <p style="color: var(--color-muted-foreground); text-align: center; font-size: 0.9rem; margin: 0;">
    <p style="color: #b0b0b0; text-align: center; font-size: 0.9rem; margin: 0; font-weight: 700;">
            เผยความลับที่ซ่อนอยู่ - เพื่อการวิเคราะห์ที่สมบูรณ์
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tab for upload methods
    tab1, tab2 = st.tabs(["อัปโหลดไฟล์", "ถ่ายภาพ"])

    with tab1:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 2.2rem; margin-bottom: 1rem; color: var(--color-accent);">เลือกไฟล์</div>
            <div style="color: var(--color-foreground); font-size: 1.1rem; margin-bottom: 0.5rem;">
            <div style="font-size: 2.2rem; margin-bottom: 1rem; color: var(--color-accent); font-weight: 700;">เลือกไฟล์</div>
            <div style="color: #ffffff; font-size: 1.1rem; margin-bottom: 0.5rem; font-weight: 700;">
                เลือกภาพด้านหลัง
            </div>
            <div style="color: var(--color-muted-foreground); font-size: 0.9rem;">
            <div style="color: #b0b0b0; font-size: 0.9rem; font-weight: 700;">
                จำกัด 10MB • รองรับหลายฟอร์แมต
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
            <div style="font-size: 2.2rem; margin-bottom: 1rem; color: var(--color-accent);">ถ่ายภาพ</div>
            <div style="color: var(--color-foreground); font-size: 1.1rem; margin-bottom: 0.5rem;">
            <div style="font-size: 2.2rem; margin-bottom: 1rem; color: var(--color-accent); font-weight: 700;">ถ่ายภาพ</div>
            <div style="color: #ffffff; font-size: 1.1rem; margin-bottom: 0.5rem; font-weight: 700;">
                กล้องถ่ายภาพ
            </div>
            <div style="color: var(--color-muted-foreground); font-size: 0.9rem;">
            <div style="color: #b0b0b0; font-size: 0.9rem; font-weight: 700;">
                ถ่ายภาพด้านหลังโดยตรง
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("เปิดกล้องถ่ายภาพ", key="back_camera_btn", help="เปิดกล้องสำหรับภาพด้านหลัง"):
            st.session_state.show_back_camera = True

        if st.session_state.get('show_back_camera', False):
            back_camera = st.camera_input(
                "Capture back sacred view",
                key="back_camera"
            )
            if back_camera:
                back = back_camera
                back_source = "camera"
                if st.button("ใช้ภาพนี้", key="back_camera_confirm"):
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
                    ตรวจสอบภาพสำเร็จ
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.image(processed_img, use_container_width=True, caption=f"ด้านหลัง ({'กล้อง' if back_source=='camera' else 'อัปโหลด'})")
            st.session_state.back_processed = processed_bytes
            st.session_state.back_filename = back.name if hasattr(back, 'name') else f"camera_back_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        else:
            st.markdown(f"""
            <div class="glassmorphic" style="padding: 1rem; text-align: center; margin: 1rem 0; border-color: rgba(239, 68, 68, 0.3);">
                <div style="color: #ef4444; font-weight: 600;">
                    ข้อผิดพลาด: {error_msg}
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
    <h3 style="color: var(--color-accent); margin: 0; font-weight: 700;">พร้อมสำหรับการวิเคราะห์</h3>
    <p style="color: var(--color-muted-foreground); margin: 0.5rem 0 0 0; font-weight: 700;">
            อัปโหลดภาพทั้งสองด้านแล้ว สามารถเริ่มวิเคราะห์ได้ทันที
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("เริ่มวิเคราะห์พระเครื่อง", type="primary", help="เริ่มวิเคราะห์พระเครื่องด้วย AI"):
        files = {
            "front": (st.session_state.front_filename, st.session_state.front_processed, "image/jpeg"),
            "back": (st.session_state.back_filename, st.session_state.back_processed, "image/jpeg")
        }
        with st.spinner("กำลังวิเคราะห์พระเครื่อง กรุณารอสักครู่..."):
            try:
                r = send_predict_request(files, API_URL, timeout=60)
                if r.ok:
                    data = r.json() if r.content else {}

                    # Enhanced Success Message
                    st.markdown("""
                    <div class="result-card mystical-glow" style="text-align: center; margin: 2rem 0;">
                        <div style="font-size: 3rem; margin-bottom: 1rem; animation: pulse 2s infinite;">⚡</div>
                        <h2 style="color: var(--color-accent); margin: 0;">Mystical Analysis Complete!</h2>
                        <p style="color: var(--color-muted-foreground); margin: 0.5rem 0 0 0;">
                            The ancient spirits have revealed their wisdom
                        <h2 style="color: var(--color-accent); margin: 0; font-weight: 700;">วิเคราะห์เสร็จสิ้น</h2>
                        <p style="color: var(--color-muted-foreground); margin: 0.5rem 0 0 0; font-weight: 700;">
                            ผลลัพธ์การวิเคราะห์แสดงด้านล่างนี้
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Enhanced Top-1 Result
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <h2 style="color: var(--color-foreground); margin: 0;">🏆 Primary Revelation</h2>
                        <p style="color: var(--color-muted-foreground);">The most likely sacred identity</p>
                        <h2 style="color: var(--color-accent); margin: 0; font-weight: 700;">ผลลัพธ์หลัก</h2>
                        <p style="color: var(--color-muted-foreground); font-weight: 700;">พระเครื่องที่มีความเป็นไปได้สูงสุด</p>
                    </div>
                    """, unsafe_allow_html=True)

                    top1 = data.get('top1') or {}
                    topk = data.get('topk') or []
                    valuation = data.get('valuation') or {}
                    confidence_percent = float(top1.get('confidence', 0)) * 100.0

                    # Enhanced confidence styling
                    # use semantic token tiers for confidence visuals
                    conf_tier = "success" if confidence_percent >= 80 else "warning" if confidence_percent >= 60 else "danger"
                    grad_var = f"var(--{conf_tier}-grad)"
                    solid_var = f"var(--{conf_tier})"

                    st.markdown(f"""
                    <div class="result-card mystical-glow" style="text-align: center; 
                         background: color-mix(in srgb, {solid_var} 10%, white);
                         border-color: color-mix(in srgb, {solid_var} 20%, transparent);">
                        <div style="font-size: 3rem; margin-bottom: 1rem; filter: drop-shadow(0 0 10px color-mix(in srgb, {solid_var} 45%, transparent));">👑</div>
                        <h2 style="color: var(--color-accent); font-size: 1.8rem; margin: 0;">
                            {data.get('top1', {{}}).get('class_name', 'Unknown')}
                        </h2>
                        <div style="margin: 1rem 0; font-size: 1.5rem;">
                            <span style="background: {grad_var}; background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;">
                                {confidence_percent:.1f}% Confidence
                            </span>
                        </div>
                        <div style="font-size: 0.9rem; color: var(--color-muted-foreground);">
                            Primary mystical classification from ancient AI wisdom
                        <div style="font-size: 0.9rem; color: var(--color-muted-foreground); font-weight: 700;">
                            การจัดหมวดหมู่โดย AI
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

                    for i, item in enumerate(topk, 1):
                        confidence_pct = float(item.get('confidence', 0)) * 100.0

                        # Medal icons and colors
                        if i == 1:
                            icon, color = "🥇", "var(--medal-gold)"
                        elif i == 2:
                            icon, color = "🥈", "var(--medal-silver)"
                        else:
                            icon, color = "🥉", "var(--medal-bronze)"

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

                    for i, rec in enumerate(data.get("recommendations", []), 1):
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
        <div class="glassmorphic mystical-glow" style="padding: 2rem; text-align: center; margin: 2rem 0; background: var(--card-fg); border-radius: 12px;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌟</div>
            <h3 style="color: var(--color-accent); margin: 0;">Begin Your Mystical Journey</h3>
            <p style="color: var(--color-muted-foreground); margin: 1rem 0;">
                Please upload {missing_text} to unlock the ancient wisdom
            <h3 style="color: var(--warning); margin: 0; font-weight: 700;">เริ่มต้นการวิเคราะห์พระเครื่อง</h3>
            <p style="color: var(--color-muted-foreground); margin: 1rem 0; font-weight: 700;">
                กรุณาอัปโหลด{missing_text}เพื่อปลดล็อกการวิเคราะห์
            </p>
                <div style="background: linear-gradient(135deg, color-mix(in srgb,var(--danger) 30%, white), color-mix(in srgb,var(--danger) 10%, white)); 
                        padding: 1rem; border-radius: var(--radius); margin: 1rem 0;
                        border: 1px solid color-mix(in srgb,var(--danger) 20%, transparent);">
                <p style="color: var(--danger); font-weight: 600; margin: 0;">
                    ⚠️ Both sacred views are required for complete mystical analysis
                <p style="color: var(--danger); font-weight: 700; margin: 0;">
                    ⚠️ ต้องอัปโหลดทั้งสองด้านจึงจะสามารถวิเคราะห์ได้ครบถ้วน
                </p>
            </div>
            <div style="color: var(--color-muted-foreground); font-size: 0.9rem; opacity: 0.8;">
                💡 Tip: Well-lit images reveal more mystical secrets
            <div style="color: #ffd600; font-size: 0.9rem; opacity: 0.8; font-weight: 700;">
                💡 เคล็ดลับ: ภาพที่มีแสงสว่างเพียงพอจะช่วยให้วิเคราะห์ได้แม่นยำขึ้น
            </div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Footer Section
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin: 3rem 0 2rem 0;">
    <h2 style="color: var(--color-foreground); margin: 0;">🔮 Mystical Technology</h2>
    <p style="color: var(--color-muted-foreground);">Powered by ancient wisdom and modern AI</p>
<div style="text-align: center; margin: 3rem 0 2rem 0; background: #fff; border-radius: 12px; padding: 2rem 1rem;">
        <h2 style="color: var(--color-accent); margin: 0; font-weight: 700;">เทคโนโลยีวิเคราะห์พระเครื่อง</h2>
    <p style="color: var(--color-muted-foreground); font-weight: 700;">ขับเคลื่อนด้วยปัญญาโบราณและ AI สมัยใหม่</p>
</div>
""", unsafe_allow_html=True)

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown("""
    <div class="glassmorphic mystical-glow" style="padding: 2rem; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
    <div class="glassmorphic mystical-glow" style="padding: 2rem; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center; background: #fff; border-radius: 10px;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🧠</div>
        <h4 style="color: var(--color-accent); margin: 0;">AI Neural Networks</h4>
        <p style="color: var(--color-muted-foreground); font-size: 0.9rem; margin: 0.5rem 0 0 0;">
            Deep learning algorithms trained on ancient mystical patterns
    <h4 style="color: var(--color-accent); margin: 0; font-weight: 700;">ปัญญาประดิษฐ์ AI</h4>
        <p style="color: #b0b0b0; font-size: 0.9rem; margin: 0.5rem 0 0 0; font-weight: 700;">
            อัลกอริทึม Deep Learning ที่ฝึกกับลวดลายพระเครื่องโบราณ
        </p>
    </div>
    """, unsafe_allow_html=True)

with tech_col2:
    st.markdown("""
    <div class="glassmorphic mystical-glow" style="padding: 2rem; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
    <div class="glassmorphic mystical-glow" style="padding: 2rem; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center; background: #fff; border-radius: 10px;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📸</div>
        <h4 style="color: var(--color-accent); margin: 0;">Multi-Format Vision</h4>
        <p style="color: var(--color-muted-foreground); font-size: 0.9rem; margin: 0.5rem 0 0 0;">
            Advanced image processing for all sacred formats
    <h4 style="color: var(--color-accent); margin: 0; font-weight: 700;">รองรับหลายฟอร์แมต</h4>
        <p style="color: #b0b0b0; font-size: 0.9rem; margin: 0.5rem 0 0 0; font-weight: 700;">
            ประมวลผลภาพพระเครื่องได้ทุกฟอร์แมต
        </p>
    </div>
    """, unsafe_allow_html=True)

with tech_col3:
    st.markdown("""
    <div class="glassmorphic mystical-glow" style="padding: 2rem; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
    <div class="glassmorphic mystical-glow" style="padding: 2rem; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center; background: #fff; border-radius: 10px;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
        <h4 style="color: var(--color-accent); margin: 0;">Real-Time Analysis</h4>
        <p style="color: var(--color-muted-foreground); font-size: 0.9rem; margin: 0.5rem 0 0 0;">
            Lightning-fast mystical insights in seconds
    <h4 style="color: var(--color-accent); margin: 0; font-weight: 700;">วิเคราะห์แบบเรียลไทม์</h4>
        <p style="color: #b0b0b0; font-size: 0.9rem; margin: 0.5rem 0 0 0; font-weight: 700;">
            วิเคราะห์ผลได้อย่างรวดเร็วในไม่กี่วินาที
        </p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced Developer Info
with st.expander("🔧 ข้อมูลสำหรับนักพัฒนา"):
    st.markdown(f"""
    <div class="glassmorphic" style="padding: 1.5rem;">
        <h4 style="color: var(--color-accent); margin: 0 0 1rem 0;">Sacred Development Details</h4>
    <h4 style="color: var(--color-accent); margin: 0 0 1rem 0; font-weight: 700;">รายละเอียดสำหรับนักพัฒนา</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-family: monospace;">
            <div>
                <strong style="color: var(--color-foreground);">API Endpoint:</strong><br>
                <code style="color: var(--color-accent);">{API_URL}</code>
                <strong style="color: #222; font-weight: 700;">API Endpoint:</strong><br>
                <code style="color: var(--color-accent);">{API_URL}</code>
            </div>
            <div>
                <strong style="color: var(--color-foreground);">Last Updated:</strong><br>
                <code style="color: var(--color-accent);">August 28, 2025</code>
                <strong style="color: #222; font-weight: 700;">อัปเดตล่าสุด:</strong><br>
                <code style="color: var(--color-accent);">28 สิงหาคม 2025</code>
            </div>
            <div>
                <strong style="color: var(--color-foreground);">Version:</strong><br>
                <code style="color: var(--color-accent);">Mystical v2.0.0</code>
                <strong style="color: #222; font-weight: 700;">เวอร์ชัน:</strong><br>
                <code style="color: var(--color-accent);">Mystical v2.0.0</code>
            </div>
            <div>
                <strong style="color: var(--color-foreground);">Framework:</strong><br>
                <code style="color: var(--color-accent);">Streamlit + FastAPI</code>
                <strong style="color: #222; font-weight: 700;">เฟรมเวิร์ก:</strong><br>
                <code style="color: var(--color-accent);">Streamlit + FastAPI</code>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)