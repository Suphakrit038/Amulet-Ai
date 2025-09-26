#!/usr/bin/env python3
"""
Production-Ready Frontend for Amulet-AI
Frontend ที่พร้อมใช้งานจริงสำหรับการจำแนกพระเครื่อง
"""

import streamlit as st
import requests
import io
from PIL import Image
import json

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Amulet-AI - ระบบจำแนกพระเครื่อง",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ตั้งค่า API
API_BASE_URL = "http://localhost:8000"

class AmuletFrontend:
    """Frontend class สำหรับ Amulet-AI"""
    
    def __init__(self):
        self.api_url = API_BASE_URL
        
    def check_api_health(self):
        """ตรวจสอบสถานะ API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200 and response.json().get("status") == "healthy"
        except:
            return False
    
    def get_classes_info(self):
        """ดึงข้อมูล classes"""
        try:
            response = requests.get(f"{self.api_url}/classes", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def predict_image(self, image_file):
        """ส่งรูปภาพไปทำนาย"""
        try:
            files = {"file": image_file}
            response = requests.post(f"{self.api_url}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}

def main():
    """ฟังก์ชันหลักของ Frontend"""
    
    # สร้าง instance
    frontend = AmuletFrontend()
    
    # Header
    st.title("🔮 Amulet-AI - ระบบจำแนกพระเครื่อง")
    st.markdown("### ระบบ AI สำหรับจำแนกประเภทพระเครื่องไทย")
    
    # ตรวจสอบสถานะ API
    api_healthy = frontend.check_api_health()
    
    with st.sidebar:
        st.header("ℹ️ ข้อมูลระบบ")
        
        if api_healthy:
            st.success("✅ API พร้อมใช้งาน")
        else:
            st.error("❌ API ไม่พร้อมใช้งาน")
            st.info("กรุณาเริ่มต้น API server ก่อนใช้งาน")
        
        # ดึงข้อมูล classes
        classes_info = frontend.get_classes_info()
        if classes_info:
            st.subheader("📋 ประเภทพระเครื่องที่รองรับ")
            for class_key, class_data in classes_info.get("classes", {}).items():
                st.write(f"**{class_data['thai']}**")
                st.write(f"- {class_data['description']}")
        
        st.markdown("---")
        st.subheader("🚀 วิธีใช้งาน")
        st.write("1. อัพโหลดรูปภาพพระเครื่อง")
        st.write("2. รอระบบประมวลผล")
        st.write("3. ดูผลการจำแนก")
        
        st.markdown("---")
        st.write("**เวอร์ชัน**: 2.0.0")
        st.write("**สถานะ**: Production Ready")
    
    # Main content
    if not api_healthy:
        st.error("⚠️ ไม่สามารถเชื่อมต่อกับ API ได้")
        st.info("กรุณาเริ่มต้น API server ด้วยคำสั่ง: `python backend/api/production_ready_api.py`")
        return
    
    # อัพโหลดไฟล์
    st.header("📤 อัพโหลดรูปภาพพระเครื่อง")
    
    uploaded_file = st.file_uploader(
        "เลือกรูปภาพพระเครื่อง",
        type=['jpg', 'jpeg', 'png'],
        help="รองรับไฟล์ JPG, JPEG, และ PNG"
    )
    
    if uploaded_file is not None:
        # แสดงรูปภาพที่อัพโหลด
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 รูปภาพที่อัพโหลด")
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)
            
            # ข้อมูลไฟล์
            st.write(f"**ชื่อไฟล์**: {uploaded_file.name}")
            st.write(f"**ประเภท**: {uploaded_file.type}")
            st.write(f"**ขนาด**: {uploaded_file.size:,} bytes")
        
        with col2:
            st.subheader("🔍 ผลการจำแนก")
            
            # ปุ่มทำนาย
            if st.button("🚀 เริ่มจำแนก", type="primary", use_container_width=True):
                
                with st.spinner("กำลังประมวลผล..."):
                    # รีเซ็ต file pointer
                    uploaded_file.seek(0)
                    
                    # ทำนาย
                    result = frontend.predict_image(uploaded_file)
                
                if "error" in result:
                    st.error(f"❌ เกิดข้อผิดพลาด: {result['error']}")
                else:
                    # แสดงผลลัพธ์
                    st.success("✅ จำแนกเสร็จสิ้น!")
                    
                    # ประเภทที่ทำนาย
                    st.markdown("### 🎯 ผลการทำนาย")
                    st.markdown(f"**ประเภท**: {result['thai_name']}")
                    st.markdown(f"**ความมั่นใจ**: {result['confidence_percentage']}")
                    
                    # Progress bar แสดงความมั่นใจ
                    confidence_value = result['confidence']
                    st.progress(confidence_value)
                    
                    # เกณฑ์ความมั่นใจ
                    if confidence_value >= 0.8:
                        st.success("🟢 ความมั่นใจสูง")
                    elif confidence_value >= 0.6:
                        st.warning("🟡 ความมั่นใจปานกลาง")
                    else:
                        st.error("🔴 ความมั่นใจต่ำ - ควรตรวจสอบอีกครั้ง")
                    
                    # ข้อมูลเพิ่มเติม
                    with st.expander("📊 ข้อมูลเพิ่มเติม"):
                        st.json({
                            "predicted_class": result["predicted_class"],
                            "thai_name": result["thai_name"],
                            "confidence": result["confidence"],
                            "model_version": result["model_version"],
                            "timestamp": result["timestamp"]
                        })
    
    # ส่วนข้อมูลเพิ่มเติม
    st.markdown("---")
    st.header("📖 เกี่ยวกับระบบ")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🤖 เทคโนโลยี AI")
        st.write("- Random Forest Classifier")
        st.write("- Feature Engineering")
        st.write("- Computer Vision")
        st.write("- Optimized for Small Dataset")
    
    with col2:
        st.subheader("📊 ประสิทธิภาพ")
        st.write("- ข้อมูลเทรน: 20 รูปต่อประเภท")
        st.write("- Features: HOG, ORB, Color, LBP")
        st.write("- Cross-validation: 3-fold")
        st.write("- Class balancing")
    
    with col3:
        st.subheader("🔧 คุณสมบัติ")
        st.write("- รองรับ 3 ประเภทพระเครื่อง")
        st.write("- API RESTful")
        st.write("- Real-time prediction")
        st.write("- Production ready")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <p>🔮 <strong>Amulet-AI v2.0.0</strong> - ระบบจำแนกพระเครื่องไทยด้วย AI</p>
        <p>พัฒนาด้วย ❤️ โดย AI Assistant</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()