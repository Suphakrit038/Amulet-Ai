import streamlit as st
import requests
from frontend.utils import validate_and_convert_image, send_predict_request, SUPPORTED_FORMATS, FORMAT_DISPLAY
from datetime import datetime
from PIL import Image
import io

# เปลี่ยนจาก st.secrets.get() เป็นการกำหนดค่าตรงๆ
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    /* Main theme colors */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom header styling */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .custom-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .custom-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Upload sections */
    .upload-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px dashed #e1e8ed;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .upload-section:hover {
        border-color: #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
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
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Tips section */
    .tips-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .tip-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-top: 3px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Progress animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .analyzing {
        animation: pulse 1.5s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# ใช้ฟังก์ชันจาก utils แทน

st.set_page_config(
    page_title="Amulet-AI", 
    page_icon="🔍", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Header Section with custom styling
st.markdown("""
<div class="custom-header">
    <h1>🔍 Amulet-AI</h1>
    <p>วิเคราะห์พระเครื่องลึกลับด้วยปัญญาประดิษฐ์</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: #667eea;">📋 คู่มือใช้งาน</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("� วิธีการใช้งาน", expanded=True):
        st.markdown("""
        1. 📤 **อัปโหลด** ภาพด้านหน้าพระเครื่อง
        2. 📷 **ถ่ายรูป** หรือเลือกภาพด้านหลัง (ไม่บังคับ)
        3. 🔍 **กดปุ่ม** "วิเคราะห์ตอนนี้"
        4. ⏳ **รอผล** การวิเคราะห์
        5. 📊 **ดูผลลัพธ์** และคำแนะนำ
        """)
    
    with st.expander("🎯 ข้อมูลระบบ"):
        st.markdown("""
        - 🤖 **เทคโนโลยี**: TensorFlow + FastAPI
        - 📈 **ความแม่นยำ**: แสดง Top-3 ผลลัพธ์
        - 💰 **ประเมินราคา**: ช่วงราคาตลาดปัจจุบัน
        - 🛒 **คำแนะนำ**: ช่องทางขายที่เหมาะสม
        """)
    
    with st.expander("📸 เคล็ดลับถ่ายรูป"):
        st.markdown("""
        **แสงสว่าง** 💡
        - ใช้แสงธรรมชาติหรือแสงขาว
        - หลีกเลี่ยงแสงสะท้อนและเงา
        
        **มุมกล้อง** 📐
        - ถ่ายตรงไม่เอียง
        - ระยะใกล้พอเห็นรายละเอียด
        
        **พื้นหลัง** 🎨
        - ใช้พื้นเรียบสีขาวหรืออ่อน
        - ไม่มีสิ่งรบกวนในภาพ
        """)
    
    with st.expander("⚠️ ข้อจำกัด"):
        st.warning("""
        - ระบบอยู่ในช่วงทดสอบ
        - ใช้ข้อมูลจำลองในการวิเคราะห์
        - ความแม่นยำประมาณ 70-80%
        """)

# Main Content
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <h2 style="color: #667eea; margin-bottom: 0.5rem;">📤 อัปโหลดรูปภาพพระเครื่อง</h2>
    <p style="color: #6c757d; margin: 0;">รองรับไฟล์: <code>{}</code></p>
</div>
""".format(FORMAT_DISPLAY), unsafe_allow_html=True)

# Image input options
col_upload, col_camera = st.columns(2)

with col_upload:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: #495057; margin: 0;">📁 อัปโหลดจากไฟล์</h4>
        <p style="color: #6c757d; font-size: 0.9rem; margin: 0.5rem 0 0 0;">เลือกไฟล์จากเครื่องของคุณ</p>
    </div>
    """, unsafe_allow_html=True)

with col_camera:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: #495057; margin: 0;">📷 ถ่ายรูปด้วยกล้อง</h4>
        <p style="color: #6c757d; font-size: 0.9rem; margin: 0.5rem 0 0 0;">🔒 ขอสิทธิ์เมื่อกดใช้งาน</p>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**ภาพด้านหน้า** (บังคับ)")
    
    # Tab สำหรับเลือกวิธีการ input
    tab1, tab2 = st.tabs(["📁 อัปโหลด", "📷 ถ่ายรูป"])
    
    with tab1:
        front_file = st.file_uploader(
            "เลือกไฟล์ภาพด้านหน้า", 
            type=SUPPORTED_FORMATS,
            key="front_upload"
        )
        front = front_file
        front_source = "upload"
    
    with tab2:
        if st.button("📷 เปิดกล้องถ่ายรูป", key="front_camera_btn", use_container_width=True):
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
                if st.button("✅ ใช้รูปนี้", key="front_camera_confirm"):
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
                    ✅ ภาพถูกต้อง กำลังแสดงผล...
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced image display
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <h5 style="color: #495057; margin: 0;">🖼️ ภาพด้านหน้า ({front_source})</h5>
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
                    ❌ ไฟล์ภาพไม่ถูกต้อง: {error_msg}
                </div>
                <div style="color: #856404; font-size: 0.9rem; margin-top: 0.5rem;">
                    💡 ลองใช้รูปภาพอื่น หรือถ่ายรูปใหม่
                </div>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #fff3cd; 
                border: 1px solid #ffeaa7; border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: #856404; margin: 0;">📸 ภาพด้านหลัง</h4>
        <p style="color: #856404; font-size: 0.85rem; margin: 0.3rem 0 0 0;">
            (ไม่บังคับ - สำหรับการวิเคราะห์ที่ละเอียดยิ่งขึ้น)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📁 อัปโหลด", "📷 ถ่ายรูป"])
    
    with tab1:
        back_file = st.file_uploader(
            "เลือกไฟล์ภาพด้านหลัง", 
            type=SUPPORTED_FORMATS,
            key="back_upload"
        )
        back = back_file
        back_source = "upload"
    
    with tab2:
        if st.button("📷 เปิดกล้องถ่ายรูป", key="back_camera_btn", use_container_width=True):
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
                if st.button("✅ ใช้รูปนี้", key="back_camera_confirm"):
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
            st.image(processed_img, caption=f"ภาพด้านหลัง ({back_source})", use_container_width=True)
            # เก็บข้อมูลที่ประมวลผลแล้วไว้ใน session_state
            st.session_state.back_processed = processed_bytes
            st.session_state.back_filename = back.name if hasattr(back, 'name') else f"camera_back_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        else:
            st.error(f"❌ ไฟล์ภาพไม่ถูกต้อง: {error_msg}")
            st.warning("💡 ลองใช้รูปภาพอื่น หรือถ่ายรูปใหม่")

st.markdown("---")

# Analysis Section
if front and hasattr(st.session_state, 'front_processed'):
    # Enhanced analyze button section
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0 1rem 0;">
        <h3 style="color: #495057; margin: 0;">🚀 พร้อมวิเคราะห์แล้ว</h3>
        <p style="color: #6c757d; font-size: 0.9rem;">กดปุ่มด้านล่างเพื่อเริ่มการวิเคราะห์ด้วย AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 วิเคราะห์ตอนนี้", type="primary", use_container_width=True):
        # ใช้ไฟล์ที่ประมวลผลแล้ว
        files = {"front": (st.session_state.front_filename, st.session_state.front_processed, "image/jpeg")}
        if back and hasattr(st.session_state, 'back_processed'):
            files["back"] = (st.session_state.back_filename, st.session_state.back_processed, "image/jpeg")
        # Enhanced loading message
        with st.spinner("⚡ กำลังประมวลผลด้วย AI... โปรดรอสักครู่"):
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
                            ✅ วิเคราะห์เสร็จสิ้น!
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
                        <h2 style="color: #495057; margin: 0;">🎯 ผลการวิเคราะห์</h2>
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
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        <div style="font-size: 2rem; margin-bottom: 1rem;">🏆</div>
                        <h2 style="color: {conf_color}; margin: 0; font-size: 1.5rem;">
                            {data['top1']['class_name']}
                        </h2>
                        <div style="margin: 1rem 0; font-size: 1.2rem; color: {conf_color};">
                            <strong>ความน่าจะเป็น: {confidence_percent:.1f}%</strong>
                        </div>
                        <div style="font-size: 0.9rem; color: {conf_color}; opacity: 0.8;">
                            ผลการวิเคราะห์อันดับ 1 จากระบบ AI
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top-3 Results with enhanced styling
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 1.5rem 0;">
                        <h3 style="color: #495057; margin: 0;">📊 ตัวเลือกอื่นๆ (Top-3)</h3>
                        <p style="color: #6c757d; font-size: 0.9rem;">ผลการจัดอันดับทั้งหมดจากระบบ AI</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create enhanced styled results
                    for i, item in enumerate(data['topk'], 1):
                        confidence_pct = item['confidence'] * 100
                        
                        # Medal and styling based on rank
                        if i == 1:
                            icon = "🥇"
                            bg_gradient = "linear-gradient(135deg, #fff3e0, #ffe0b3)"
                            border_color = "#ffcc80"
                            text_color = "#e65100"
                        elif i == 2:
                            icon = "🥈"
                            bg_gradient = "linear-gradient(135deg, #f3e5f5, #ce93d8)"
                            border_color = "#ba68c8"
                            text_color = "#4a148c"
                        else:
                            icon = "🥉"
                            bg_gradient = "linear-gradient(135deg, #fff8e1, #ffecb3)"
                            border_color = "#ffcc02"
                            text_color = "#f57f17"
                        
                        st.markdown(f"""
                        <div style="padding: 1rem; margin: 0.8rem 0; border-radius: 10px;
                                    background: {bg_gradient}; 
                                    border: 1px solid {border_color};
                                    display: flex; align-items: center;">
                            <div style="font-size: 1.5rem; margin-right: 1rem;">{icon}</div>
                            <div style="flex-grow: 1;">
                                <div style="font-weight: bold; color: {text_color}; font-size: 1.1rem;">
                                    {item['class_name']}
                                </div>
                                <div style="color: {text_color}; font-size: 0.9rem; opacity: 0.8;">
                                    ความน่าจะเป็น: {confidence_pct:.1f}%
                                </div>
                            </div>
                            <div style="text-align: right; color: {text_color};">
                                <div style="font-size: 0.8rem; opacity: 0.7;">อันดับ {i}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_rank, col_name, col_conf = st.columns([0.5, 3, 1])
                        with col_rank:
                            st.markdown(f"**{icon}**")
                        with col_name:
                            st.markdown(f"**{item['class_name']}**")
                        with col_conf:
                            st.markdown(f"`{confidence_pct:.1f}%`")
                    
                    # Price Valuation
                    st.markdown("---")
                    st.subheader("💰 ช่วงราคาประเมิน")
                    
                    price_col1, price_col2, price_col3 = st.columns(3)
                    with price_col1:
                        st.metric(
                            label="💸 ราคาต่ำ (P05)",
                            value=f"{data['valuation']['p05']:,.0f} ฿",
                            help="ราคาต่ำสุดในตลาด"
                        )
                    with price_col2:
                        st.metric(
                            label="💵 ราคากลาง (P50)",
                            value=f"{data['valuation']['p50']:,.0f} ฿",
                            help="ราคาเฉลี่ยในตลาด"
                        )
                    with price_col3:
                        st.metric(
                            label="💳 ราคาสูง (P95)",
                            value=f"{data['valuation']['p95']:,.0f} ฿",
                            help="ราคาสูงสุดในตลาด"
                        )
                    
                    # Recommendations
                    st.markdown("---")
                    st.subheader("🛒 แนะนำช่องทางการขาย")
                    
                    for i, rec in enumerate(data["recommendations"], 1):
                        with st.expander(f"📍 {rec['market']}", expanded=i==1):
                            st.write(f"💡 **เหตุผล:** {rec['reason']}")
                            if rec['market'] == "Facebook Marketplace":
                                st.info("🔗 เหมาะสำหรับการขายให้คนทั่วไป")
                            elif rec['market'] == "Shopee":
                                st.info("🛍️ มีระบบรีวิวและการันตี")
                
                else:
                    st.error(f"❌ เกิดข้อผิดพลาด โปรดลองใหม่อีกครั้ง: {r.status_code} - {r.text}")
                    
            except requests.exceptions.Timeout:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fff3cd, #ffeaa7); 
                            border: 1px solid #ffeaa7; border-radius: 10px; 
                            padding: 1.5rem; margin: 1rem 0; text-align: center;">
                    <div style="color: #856404; font-size: 1.1rem; font-weight: bold;">
                        ⏰ การประมวลผลใช้เวลานานเกินไป
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
                        🔌 ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ได้
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
                        💥 เกิดข้อผิดพลาดไม่คาดคิด
                    </div>
                    <div style="color: #721c24; font-size: 0.9rem; margin-top: 0.5rem;">
                        รายละเอียด: {str(e)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    if front:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fff3cd, #ffeaa7); 
                    border: 1px solid #ffeaa7; border-radius: 10px; 
                    padding: 1.5rem; margin: 1rem 0; text-align: center;">
            <div style="color: #856404; font-size: 1.1rem; font-weight: bold;">
                🔄 กำลังประมวลผลรูปภาพ...
            </div>
            <div style="color: #856404; font-size: 0.9rem; margin-top: 0.5rem;">
                กรุณารอสักครู่ ระบบกำลังเตรียมข้อมูล
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #cce7ff, #b3daff); 
                    border: 1px solid #b3daff; border-radius: 10px; 
                    padding: 2rem; margin: 2rem 0; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">📋</div>
            <div style="color: #0056b3; font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">
                เริ่มต้นการวิเคราะห์
            </div>
            <div style="color: #0056b3; font-size: 0.95rem;">
                กรุณาอัปโหลดภาพด้านหน้าหรือถ่ายรูปก่อนเริ่มการวิเคราะห์
            </div>
            <div style="color: #0056b3; font-size: 0.8rem; margin-top: 0.8rem; opacity: 0.8;">
                💡 เคล็ดลับ: ภาพที่มีแสงสว่างเพียงพอจะให้ผลลัพธ์ที่ดีกว่า
            </div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Tips section
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin: 2rem 0 1.5rem 0;">
    <h2 style="color: #495057; margin: 0;">💡 เคล็ดลับการถ่ายภาพที่ดี</h2>
    <p style="color: #6c757d; font-size: 0.9rem;">วิธีการถ่ายภาพเพื่อให้ได้ผลการวิเคราะห์ที่แม่นยำที่สุด</p>
</div>
""", unsafe_allow_html=True)

col_tip1, col_tip2, col_tip3 = st.columns(3)

with col_tip1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e3f2fd, #bbdefb); 
                border-radius: 15px; padding: 1.5rem; height: 180px;
                text-align: center; border: 1px solid #90caf9;">
        <div style="font-size: 2rem; margin-bottom: 0.8rem;">📸</div>
        <h4 style="color: #1565c0; margin: 0.5rem 0;">แสงสว่าง</h4>
        <p style="color: #1565c0; font-size: 0.85rem; margin: 0; line-height: 1.4;">
            ถ่ายในที่แสงสว่างเพียงพอ<br>
            หลีกเลี่ยงแสงแรง<br>
            ใช้แสงธรรมชาติ
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_tip2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e5f5, #e1bee7); 
                border-radius: 15px; padding: 1.5rem; height: 180px;
                text-align: center; border: 1px solid #ce93d8;">
        <div style="font-size: 2rem; margin-bottom: 0.8rem;">🎯</div>
        <h4 style="color: #6a1b9a; margin: 0.5rem 0;">มุมกล้อง</h4>
        <p style="color: #6a1b9a; font-size: 0.85rem; margin: 0; line-height: 1.4;">
            ถ่ายตรงกลางวัตถุ<br>
            หลีกเลี่ยงมุมเอียง<br>
            ระยะ 20-30 ซม.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_tip3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fff3e0, #ffcc80); 
                border-radius: 15px; padding: 1.5rem; height: 180px;
                text-align: center; border: 1px solid #ffb74d;">
        <div style="font-size: 2rem; margin-bottom: 0.8rem;">🖼️</div>
        <h4 style="color: #e65100; margin: 0.5rem 0;">พื้นหลัง</h4>
        <p style="color: #e65100; font-size: 0.85rem; margin: 0; line-height: 1.4;">
            ใช้พื้นหลังเรียบ<br>
            สีขาวหรือสีอ่อน<br>
            ไม่มีสิ่งรบกวน
        </p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin: 2rem 0 1rem 0;">
    <h3 style="color: #495057; margin: 0;">🚀 เทคโนโลยี</h3>
    <p style="color: #6c757d; font-size: 0.9rem;">ระบบปัญญาประดิษฐ์ขั้นสูงสำหรับการวิเคราะห์พระเครื่อง</p>
</div>
""", unsafe_allow_html=True)

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🤖</div>
        <h4 style="color: #495057; margin: 0.5rem 0;">AI Technology</h4>
        <p style="color: #6c757d; font-size: 0.85rem; margin: 0;">TensorFlow + FastAPI</p>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📱</div>
        <h4 style="color: #495057; margin: 0.5rem 0;">Multi-Format</h4>
        <p style="color: #6c757d; font-size: 0.85rem; margin: 0;">JPG, PNG, HEIC & More</p>
    </div>
    """, unsafe_allow_html=True)

with col_info3:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📷</div>
        <h4 style="color: #495057; margin: 0.5rem 0;">Camera Ready</h4>
        <p style="color: #6c757d; font-size: 0.85rem; margin: 0;">ถ่ายรูปได้ทันที</p>
    </div>
    """, unsafe_allow_html=True)

# Development info
with st.expander("🔧 Developer Info"):
    st.write(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.write(f"🌐 API URL: {API_URL}")
    st.write("👨‍💻 Developed with Streamlit & FastAPI")