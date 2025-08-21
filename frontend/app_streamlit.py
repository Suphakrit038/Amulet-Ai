import streamlit as st
import requests
from datetime import datetime
from PIL import Image
import io

# รองรับ HEIC format
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "heic", "heif", "webp", "bmp", "tiff"]
    FORMAT_DISPLAY = "JPG, JPEG, PNG, HEIC, HEIF, WebP, BMP, TIFF"
except ImportError:
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
    FORMAT_DISPLAY = "JPG, JPEG, PNG, WebP, BMP, TIFF"

# เปลี่ยนจาก st.secrets.get() เป็นการกำหนดค่าตรงๆ
API_URL = "http://localhost:8000"

# ฟังก์ชันตรวจสอบและแปลงรูปภาพ
def validate_and_convert_image(uploaded_file):
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        
        # แปลงเป็น RGB (จำเป็นสำหรับบางฟอร์แมต)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # แปลงเป็น bytes สำหรับการส่งไป API
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        uploaded_file.seek(0)  # reset original file pointer
        return True, img, img_byte_arr, None
    except Exception as e:
        return False, None, None, str(e)

st.set_page_config(
    page_title="Amulet-AI", 
    page_icon="🔍", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Header Section
st.title("🔍 Amulet-AI")
st.markdown("### วิเคราะห์พระเครื่องลึกลับด้วยปัญญาประดิษฐ์")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 วิธีการใช้งาน")
    st.markdown("""
    1. 📤 อัปโหลดภาพด้านหน้าพระเครื่อง
    2. 📷 อัปโหลดภาพด้านหลัง (ไม่บังคับ)
    3. 🔍 กดปุ่ม "วิเคราะห์ตอนนี้"
    4. ⏳ รอผลการวิเคราะห์
    5. 📊 ดูผลลัพธ์และคำแนะนำ
    """)
    
    st.markdown("---")
    st.header("ℹ️ ข้อมูลเพิ่มเติม")
    st.info("⚡ ระบบใช้ AI เพื่อจำแนกประเภทพระเครื่อง")
    st.info("📈 แสดงความน่าจะเป็น Top-3")
    st.info("💰 ประเมินช่วงราคาตลาด")
    st.info("🛒 แนะนำช่องทางขาย")

# Main Content
st.subheader("📤 อัปโหลดรูปภาพพระเครื่อง")
st.info(f"📋 รองรับไฟล์: {FORMAT_DISPLAY}")

# Image input options
col_upload, col_camera = st.columns(2)

with col_upload:
    st.markdown("**📁 อัปโหลดจากไฟล์**")

with col_camera:
    st.markdown("**📷 ถ่ายรูปด้วยกล้อง**")
    st.caption("🔒 จะขอสิทธิ์กล้องเมื่อกดใช้งาน")

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
            st.image(processed_img, caption=f"ภาพด้านหน้า ({front_source})", use_container_width=True)
            # เก็บข้อมูลที่ประมวลผลแล้วไว้ใน session_state
            st.session_state.front_processed = processed_bytes
            st.session_state.front_filename = front.name if hasattr(front, 'name') else f"camera_front_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        else:
            st.error(f"❌ ไฟล์ภาพไม่ถูกต้อง: {error_msg}")
            st.warning("💡 ลองใช้รูปภาพอื่น หรือถ่ายรูปใหม่")

with col2:
    st.markdown("**ภาพด้านหลัง** (ไม่บังคับ)")
    
    # Tab สำหรับเลือกวิธีการ input
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
    if st.button("🔍 วิเคราะห์ตอนนี้", type="primary", use_container_width=True):
        # ใช้ไฟล์ที่ประมวลผลแล้ว
        files = {"front": (st.session_state.front_filename, st.session_state.front_processed, "image/jpeg")}
        if back and hasattr(st.session_state, 'back_processed'):
            files["back"] = (st.session_state.back_filename, st.session_state.back_processed, "image/jpeg")
        
        with st.spinner("⚡ กำลังประมวลผล..."):
            try:
                r = requests.post(f"{API_URL}/predict", files=files, timeout=60)
                
                if r.ok:
                    data = r.json()
                    st.success("✅ วิเคราะห์เสร็จสิ้น!")
                    
                    # Main Result
                    st.markdown("---")
                    st.subheader("🎯 ผลการวิเคราะห์")
                    
                    # Top-1 Result with styling
                    confidence_percent = data['top1']['confidence'] * 100
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: #f0f8ff; border-left: 5px solid #1f77b4;">
                        <h3 style="color: #1f77b4; margin: 0;">🏆 {data['top1']['class_name']}</h3>
                        <p style="font-size: 18px; margin: 10px 0;">ความน่าจะเป็น: <strong>{confidence_percent:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top-3 Results
                    st.markdown("---")
                    st.subheader("📊 ตัวเลือกอื่นๆ (Top-3)")
                    
                    # Create a styled table
                    for i, item in enumerate(data['topk'], 1):
                        confidence_pct = item['confidence'] * 100
                        if i == 1:
                            icon = "🥇"
                        elif i == 2:
                            icon = "🥈"
                        else:
                            icon = "🥉"
                        
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
                    st.error(f"❌ เกิดข้อผิดพลาด: {r.status_code} - {r.text}")
                    
            except requests.exceptions.Timeout:
                st.error("⏰ การประมวลผลใช้เวลานานเกินไป กรุณาลองใหม่")
            except requests.exceptions.ConnectionError:
                st.error("🔌 ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ได้ กรุณาตรวจสอบว่า Backend กำลังทำงานอยู่")
            except Exception as e:
                st.error(f"💥 เกิดข้อผิดพลาดไม่คาดคิด: {str(e)}")

else:
    if front:
        st.warning("🔄 กำลังประมวลผลรูปภาพ... กรุณารอสักครู่")
    else:
        st.info("📋 กรุณาอัปโหลดภาพด้านหน้าหรือถ่ายรูปก่อนเริ่มการวิเคราะห์")

# Tips section
st.markdown("---")
st.subheader("💡 เคล็ดลับการถ่ายภาพที่ดี")
col_tip1, col_tip2, col_tip3 = st.columns(3)

with col_tip1:
    st.markdown("""
    **📸 แสงสว่าง**
    - ใช้แสงธรรมชาติ
    - หลีกเลี่ยงแสงสะท้อน
    - ไม่ใช้ Flash
    """)

with col_tip2:
    st.markdown("""
    **🎯 มุมกล้อง**
    - ถ่ายตรงกับพระเครื่อง
    - เคลียร์ภาพพระทั้งหมด
    - หลีกเลี่ยงการเอียง
    """)

with col_tip3:
    st.markdown("""
    **🖼️ พื้นหลัง**
    - ใช้พื้นหลังเรียบ
    - สีขาวหรือสีอ่อน
    - ไม่มีสิ่งรบกวน
    """)

# Footer
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    <div style="text-align: center;">
        <h4>🤖 AI Technology</h4>
        <p>TensorFlow + FastAPI</p>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div style="text-align: center;">
        <h4>📱 Multi-Format</h4>
        <p>รองรับ HEIC, JPG, PNG+</p>
    </div>
    """, unsafe_allow_html=True)

with col_info3:
    st.markdown("""
    <div style="text-align: center;">
        <h4>📷 Camera Ready</h4>
        <p>ถ่ายรูปได้ทันที</p>
    </div>
    """, unsafe_allow_html=True)

# Development info
with st.expander("🔧 Developer Info"):
    st.write(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.write(f"🌐 API URL: {API_URL}")
    st.write("👨‍💻 Developed with Streamlit & FastAPI")