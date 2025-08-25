import streamlit as st
import requests
import io
from datetime import datetime
from PIL import Image

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
    page_title="🔍 Amulet-AI", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean CSS
st.markdown("""
<style>
    .main {
        padding: 2rem 1rem;
    }
    
    .header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .upload-box {
        border: 2px dashed #ddd;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .confidence-high { border-left: 4px solid #28a745; }
    .confidence-medium { border-left: 4px solid #ffc107; }
    .confidence-low { border-left: 4px solid #dc3545; }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        width: 100%;
    }
    
    .price-metric {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Functions
def validate_image(uploaded_file):
    """ตรวจสอบและแปลงไฟล์รูปภาพ"""
    try:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # แปลงเป็น bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        return image, img_byte_arr, None
    except Exception as e:
        return None, None, str(e)

# Main App
def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>🔍 Amulet-AI</h1>
        <p>ระบบวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload Section
    st.subheader("📤 อัปโหลดรูปภาพ")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📷 ภาพด้านหน้า** (บังคับ)")
        front_file = st.file_uploader(
            "เลือกไฟล์ภาพด้านหน้า", 
            type=['jpg', 'jpeg', 'png', 'heic'],
            key="front"
        )
        
        if front_file:
            image, img_bytes, error = validate_image(front_file)
            if image:
                st.image(image, caption="ภาพด้านหน้า", use_container_width=True)
                st.session_state.front_data = img_bytes
            else:
                st.error(f"❌ ไฟล์ไม่ถูกต้อง: {error}")
    
    with col2:
        st.markdown("**📷 ภาพด้านหลัง** (ไม่บังคับ)")
        back_file = st.file_uploader(
            "เลือกไฟล์ภาพด้านหลัง", 
            type=['jpg', 'jpeg', 'png', 'heic'],
            key="back"
        )
        
        if back_file:
            image, img_bytes, error = validate_image(back_file)
            if image:
                st.image(image, caption="ภาพด้านหลัง", use_container_width=True)
                st.session_state.back_data = img_bytes
            else:
                st.error(f"❌ ไฟล์ไม่ถูกต้อง: {error}")
    
    # Analysis Button
    if front_file and hasattr(st.session_state, 'front_data'):
        st.markdown("---")
        
        if st.button("🔍 วิเคราะห์ภาพ", type="primary"):
            with st.spinner("🤖 กำลังประมวลผล..."):
                try:
                    # เตรียมไฟล์
                    files = {"front": ("front.jpg", st.session_state.front_data, "image/jpeg")}
                    if hasattr(st.session_state, 'back_data'):
                        files["back"] = ("back.jpg", st.session_state.back_data, "image/jpeg")
                    
                    # เรียก API
                    response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                    
                    if response.ok:
                        data = response.json()
                        
                        # แสดงผลลัพธ์
                        st.success("✅ วิเคราะห์เสร็จสิ้น!")
                        
                        # ผลลัพธ์หลัก
                        st.subheader("🏆 ผลการวิเคราะห์")
                        
                        confidence = data['top1']['confidence'] * 100
                        if confidence >= 80:
                            conf_class = "confidence-high"
                        elif confidence >= 60:
                            conf_class = "confidence-medium"
                        else:
                            conf_class = "confidence-low"
                        
                        st.markdown(f"""
                        <div class="result-card {conf_class}">
                            <h3>🥇 {data['top1']['class_name']}</h3>
                            <p><strong>ความน่าจะเป็น: {confidence:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Top-3 Results
                        st.subheader("📊 ผลลัพธ์ทั้งหมด")
                        
                        for i, result in enumerate(data['topk'], 1):
                            conf_pct = result['confidence'] * 100
                            medal = ["🥇", "🥈", "🥉"][i-1]
                            
                            col_rank, col_name, col_conf = st.columns([0.5, 3, 1])
                            with col_rank:
                                st.markdown(f"**{medal}**")
                            with col_name:
                                st.write(result['class_name'])
                            with col_conf:
                                st.write(f"{conf_pct:.1f}%")
                        
                        # ราคาประเมิน
                        st.subheader("💰 ราคาประเมิน")
                        
                        price_col1, price_col2, price_col3 = st.columns(3)
                        
                        with price_col1:
                            st.metric("ราคาต่ำ (P05)", f"{data['valuation']['p05']:,} ฿")
                        with price_col2:
                            st.metric("ราคากลาง (P50)", f"{data['valuation']['p50']:,} ฿")
                        with price_col3:
                            st.metric("ราคาสูง (P95)", f"{data['valuation']['p95']:,} ฿")
                        
                        # คำแนะนำ
                        st.subheader("🛒 คำแนะนำการขาย")
                        
                        for rec in data.get('recommendations', []):
                            with st.expander(f"📍 {rec['market']}"):
                                st.write(f"💡 {rec['reason']}")
                    
                    else:
                        st.error(f"❌ เกิดข้อผิดพลาด: {response.status_code}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("🔌 ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้")
                except Exception as e:
                    st.error(f"💥 เกิดข้อผิดพลาด: {str(e)}")
    
    # Tips Section
    st.markdown("---")
    st.subheader("💡 เคล็ดลับการถ่ายภาพ")
    
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    
    with tip_col1:
        st.markdown("""
        **📸 แสงสว่าง**
        - ใช้แสงธรรมชาติ
        - หลีกเลี่ยงเงา
        - ไม่แสงแรงเกินไป
        """)
    
    with tip_col2:
        st.markdown("""
        **🎯 มุมกล้อง**
        - ถ่ายตรงไม่เอียง
        - ระยะ 20-30 ซม.
        - เห็นรายละเอียดชัด
        """)
    
    with tip_col3:
        st.markdown("""
        **🖼️ พื้นหลัง**
        - ใช้พื้นเรียบ
        - สีขาวหรือสีอ่อน
        - ไม่มีสิ่งรบกวน
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**🔧 เทคโนโลยี:** TensorFlow + FastAPI + Streamlit")
    
    # Warning
    st.warning("""
    ⚠️ **ข้อจำกัด:** ระบบใช้ข้อมูลทดสอบ ความแม่นยำอาจไม่สูง
    """)

if __name__ == "__main__":
    main()