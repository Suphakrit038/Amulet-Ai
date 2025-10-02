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

# Configuration
st.set_page_config(
    page_title="🔮 Amulet-AI - ระบบจำแนกพระเครื่อง",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Constants
API_BASE_URL = "http://localhost:8000"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Theme Colors (Thai Traditional)
COLORS = {
    'primary': '#8B4513',      # สีน้ำตาลทอง
    'secondary': '#DAA520',    # สีทอง
    'accent': '#FF6347',       # สีส้มแดง
    'success': '#32CD32',      # สีเขียว
    'warning': '#FFD700',      # สีเหลืองทอง
    'error': '#DC143C',        # สีแดง
    'info': '#4682B4',         # สีฟ้า
    'dark': '#2F4F4F',         # สีเทาเข้ม
    'light': '#F5F5DC'         # สีครีม
}

# Custom CSS - Clean and Modern
st.markdown("""
<style>
    /* Import Thai Font */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Prompt', sans-serif;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Header */
    .main-header {
        background: linear-gradient(90deg, #8B4513 0%, #DAA520 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(139, 69, 19, 0.3);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    /* Cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #DAA520;
    }
    
    .upload-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        margin: 2rem 0;
        border: 2px dashed #DAA520;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.15);
        margin: 1rem 0;
        max-width: 1200px;
        width: 98vw;
        margin: 0 auto;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #8B4513 0%, #DAA520 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 69, 19, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 69, 19, 0.4);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #8B4513 0%, #DAA520 100%);
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Status Messages */
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    /* Image Display */
    .image-container {
        text-align: center;
        margin: 1rem 0;
    }
    
    .image-container img {
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        max-width: 100%;
        height: auto;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Thai Text */
    .thai-text {
        font-family: 'Prompt', sans-serif;
        line-height: 1.6;
    }
    
    /* Loading Spinner */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #DAA520;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def encode_image_to_base64(image):
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

# Main App Layout
def main():
    # Header
    st.markdown("""
    <div class="main-header fade-in">
        <div class="main-title">🔮 Amulet-AI</div>
        <div class="main-subtitle">ระบบจำแนกพระเครื่องอัจฉริยะ - Thai Amulet Classification System</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card fade-in">
            <h3>🎯 การจำแนกแม่นยำ</h3>
            <p class="thai-text">ใช้ AI ขั้นสูงวิเคราะห์ลักษณะพระเครื่อง</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card fade-in">
            <h3>⚡ ประมวลผลเร็ว</h3>
            <p class="thai-text">ผลลัพธ์ภายในไม่กี่วินาที</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card fade-in">
            <h3>🔒 ปลอดภัย</h3>
            <p class="thai-text">ข้อมูลของคุณจะไม่ถูกเก็บบันทึก</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Image Upload Section
    st.markdown("""
    <div class="upload-card">
        <h2>📸 อัปโหลดรูปภาพพระเครื่อง</h2>
        <p class="thai-text">กรุณาเลือกรูปภาพพระเครื่องที่ต้องการจำแนกประเภท</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "เลือกไฟล์รูปภาพ",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="รองรับไฟล์ PNG, JPG, JPEG, BMP ขนาดไม่เกิน 10MB"
    )
    
    if uploaded_file is not None:
        # Validate file
        is_valid, message = validate_image(uploaded_file)
        
        if not is_valid:
            st.markdown(f"""
            <div class="error-message">
                <strong>❌ ข้อผิดพลาด:</strong> {message}
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Display success message
        st.markdown("""
        <div class="success-message">
            <strong>✅ สำเร็จ:</strong> ไฟล์ถูกต้องและพร้อมประมวลผล
        </div>
        """, unsafe_allow_html=True)
        
        # Process image
        try:
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for display
            display_image = resize_image(image.copy(), max_size=600)
            
            # Display image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(display_image, caption="รูปภาพที่อัปโหลด", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Prediction Section
            st.markdown("---")
            
            if st.button("🔍 เริ่มการวิเคราะห์", key="predict_btn"):
                with st.spinner("กำลังวิเคราะห์รูปภาพ..."):
                    # Try API first, fallback to mock
                    image_data = encode_image_to_base64(image)
                    success, result = predict_via_api(image_data)
                    
                    if not success:
                        # Show API error and use mock
                        st.markdown(f"""
                        <div class="warning-message">
                            <strong>⚠️ หมายเหตุ:</strong> {result}<br>
                            <em>ใช้ระบบจำลองสำหรับการทดสอบ</em>
                        </div>
                        """, unsafe_allow_html=True)
                        result = mock_prediction(image)
                    
                    # Display Results
                    if result.get("status") == "success":
                        st.markdown("""
                        <div class="result-card fade-in">
                            <h2>📊 ผลการวิเคราะห์</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        predictions = result.get("predictions", [])
                        top_prediction = predictions[0] if predictions else None
                        
                        if top_prediction:
                            # Top Prediction
                            confidence = top_prediction["confidence"]
                            class_name = top_prediction["class"]
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("""
                                <div class="metric-container">
                                    <h3>🏆 ประเภทพระเครื่อง</h3>
                                    <h2 style="color: #8B4513;">{}</h2>
                                </div>
                                """.format(class_name), unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("""
                                <div class="metric-container">
                                    <h3>📈 ความเชื่อมั่น</h3>
                                    <h2 style="color: #DAA520;">{:.1f}%</h2>
                                </div>
                                """.format(confidence), unsafe_allow_html=True)
                            
                            with col3:
                                processing_time = result.get("processing_time", 0)
                                st.markdown("""
                                <div class="metric-container">
                                    <h3>⚡ เวลาประมวลผล</h3>
                                    <h2 style="color: #32CD32;">{:.2f}s</h2>
                                </div>
                                """.format(processing_time), unsafe_allow_html=True)
                            
                            # Confidence Bar
                            st.markdown("### 📊 ระดับความเชื่อมั่นทั้งหมด")
                            
                            for i, pred in enumerate(predictions[:5]):  # Show top 5
                                conf = pred["confidence"]
                                name = pred["class"]
                                
                                # Create progress bar
                                st.write(f"**{name}**")
                                progress_col, percent_col = st.columns([4, 1])
                                
                                with progress_col:
                                    st.progress(conf / 100)
                                
                                with percent_col:
                                    st.write(f"{conf:.1f}%")
                            
                            # Additional Info
                            st.markdown("---")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("""
                                <div class="feature-card">
                                    <h4>ℹ️ ข้อมูลเพิ่มเติม</h4>
                                    <ul class="thai-text">
                                        <li>โมเดล: {}</li>
                                        <li>วันที่วิเคราะห์: {}</li>
                                        <li>ขนาดรูปภาพ: {}x{} พิกเซล</li>
                                    </ul>
                                </div>
                                """.format(
                                    result.get("model_version", "v2.1.0"),
                                    datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                    image.width, image.height
                                ), unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("""
                                <div class="feature-card">
                                    <h4>⚠️ คำแนะนำ</h4>
                                    <ul class="thai-text">
                                        <li>ผลการวิเคราะห์เป็นเพียงข้อมูลเบื้องต้น</li>
                                        <li>ควรปรึกษาผู้เชี่ยวชาญเพื่อยืนยัน</li>
                                        <li>ถ่ายภาพให้ชัดเจนเพื่อผลลัพธ์ที่แม่นยำ</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        else:
                            st.markdown("""
                            <div class="error-message">
                                <strong>❌ ข้อผิดพลาด:</strong> ไม่สามารถวิเคราะห์รูปภาพได้
                            </div>
                            """, unsafe_allow_html=True)
                    
                    else:
                        error_msg = result.get("error", "เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ")
                        st.markdown(f"""
                        <div class="error-message">
                            <strong>❌ ข้อผิดพลาด:</strong> {error_msg}
                        </div>
                        """, unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f"""
            <div class="error-message">
                <strong>❌ ข้อผิดพลาด:</strong> ไม่สามารถประมวลผลรูปภาพได้<br>
                <em>รายละเอียด: {str(e)}</em>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p class="thai-text">
            🔮 <strong>Amulet-AI</strong> | ระบบจำแนกพระเครื่องอัจฉริยะ<br>
            <em>พัฒนาด้วยเทคโนโลยี AI ขั้นสูงเพื่อการอนุรักษ์มรดกทางวัฒนธรรมไทย</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()