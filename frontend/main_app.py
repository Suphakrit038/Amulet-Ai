import streamlit as st
import requests
from datetime import datetime
from PIL import Image, ImageFilter, ImageStat
import io
import os
import sys
import base64
from io import BytesIO
from typing import Any
import json
import time
import numpy as np
from pathlib import Path
import logging

# Lazy import matplotlib to avoid circular import issues
matplotlib = None
plt = None

def get_matplotlib():
    """Lazy import matplotlib to avoid circular import issues"""
    global matplotlib, plt
    if matplotlib is None:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            st.warning("matplotlib ไม่พร้อมใช้งาน การแสดงกราฟบางอย่างจะถูกข้าม")
            return None, None
    return matplotlib, plt

# พยายามนำเข้า OpenCV หากไม่มีให้ใช้ตัวสำรอง
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("คำเตือน: ไม่พบ OpenCV การตรวจสอบคุณภาพภาพบางอย่างจะถูกข้าม")

# สำหรับฟีเจอร์เปรียบเทียบรูปภาพ
try:
    from frontend.comparison_module import FeatureExtractor, ImageComparer
except ImportError:
    # ถ้าไม่พบโมดูล ให้สร้าง dummy class ไว้ก่อน
    class FeatureExtractor:
        def __init__(self, *args, **kwargs):
            pass
    class ImageComparer:
        def __init__(self, *args, **kwargs):
            pass
        def compare_image(self, *args, **kwargs):
            return {"top_matches": []}

# ==========================================================
# Imports / Utils (prefer the first file's implementations)
# ==========================================================

# กำหนดค่าพื้นฐานสำหรับรูปแบบไฟล์ที่รองรับ
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    SUPPORTED_FORMATS = [
        "jpg",
        "jpeg",
        "png",
        "heic",
        "heif",
        "webp",
        "bmp",
        "tiff",
    ]
    FORMAT_DISPLAY = "JPG, JPEG, PNG, HEIC, HEIF, WebP, BMP, TIFF"
except Exception:
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
    FORMAT_DISPLAY = "JPG, JPEG, PNG, WebP, BMP, TIFF"

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from frontend.utils import validate_and_convert_image
except ImportError:
    st.error("ไม่สามารถโหลดโมดูล utils ได้")
except Exception:
    # ---- ส่วนสำรองจากไฟล์แรก ----
    pass

    MAX_FILE_SIZE_MB = 10

    def validate_and_convert_image(uploaded_file):
        """การตรวจสอบคุณภาพภาพแบบขั้นสูง พร้อมตรวจจับภาพเบลอ การตรวจสอบคุณภาพ และการตรวจสอบเนื้อหา"""
        try:
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

            if hasattr(uploaded_file, "read"):
                file_bytes = uploaded_file.read()
            else:
                file_bytes = getattr(uploaded_file, "getvalue", lambda: b"")()

            if not file_bytes:
                return False, None, None, "❌ ไฟล์ว่างเปล่าหรือไม่สามารถอ่านได้"

            if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
                return False, None, None, f"❌ ไฟล์ใหญ่เกินไป (> {MAX_FILE_SIZE_MB} MB)"

            filename = getattr(uploaded_file, "name", "") or ""
            if filename:
                ext = filename.rsplit(".", 1)[-1].lower()
                if ext not in SUPPORTED_FORMATS:
                    return False, None, None, f"❌ ไฟล์ประเภทนี้ไม่รองรับ: .{ext}"

            # เปิดและตรวจสอบภาพ
            img = Image.open(io.BytesIO(file_bytes))
            
            # 1. ตรวจสอบความละเอียดขั้นต่ำ
            width, height = img.size
            min_dimension = 200
            if width < min_dimension or height < min_dimension:
                return False, None, None, f"❌ ภาพมีความละเอียดต่ำเกินไป (ต้องมีขนาดอย่างน้อย {min_dimension}x{min_dimension} พิกเซล)"
            
            # แปลงเป็น RGB สำหรับการประมวลผล
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # 2. ตรวจสอบความสว่างและคอนทราสต์พื้นฐาน (ใช้ PIL)
            stat = ImageStat.Stat(img)
            mean_brightness = sum(stat.mean) / len(stat.mean)
            
            if mean_brightness < 30:
                return False, None, None, "❌ ภาพมืดเกินไป กรุณาถ่ายในที่ที่มีแสงเพียงพอ"
            elif mean_brightness > 240:
                return False, None, None, "❌ ภาพสว่างเกินไป อาจมีแสงแฟลชส่องจนเกิน"
            
            # 3. ตรวจสอบอัตราส่วนที่ผิดปกติ
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 3:
                return False, None, None, "❌ ภาพมีอัตราส่วนที่ผิดปกติ กรุณาถ่ายภาพพระเครื่องในมุมที่เหมาะสม"
            
            # 4. การตรวจจับภาพเบลอด้วย PIL โดยใช้ตัวกรองภาพ
            gray_img = img.convert('L')  # แปลงเป็นขาวดำ
            blur_img = gray_img.filter(ImageFilter.BLUR)
            
            # คำนวณความแตกต่างระหว่างภาพต้นฉบับและภาพที่เบลอ
            gray_array = np.array(gray_img)
            blur_array = np.array(blur_img)
            diff = np.mean(np.abs(gray_array.astype(float) - blur_array.astype(float)))
            
            if diff < 5:  # ความแตกต่างต่ำมากแสดงว่าภาพเบลออยู่แล้ว
                return False, None, None, f"❌ ภาพเบลอหรือไม่คมชัด กรุณาถ่ายภาพใหม่ให้คมชัดขึ้น"
            
            # 5. ตรวจสอบความหลากหลายของสี
            img_array = np.array(img)
            color_std = np.std(img_array)
            
            if color_std < 20:
                return False, None, None, "❌ ภาพมีรายละเอียดน้อยเกินไป อาจเป็นภาพสีเดียวหรือภาพเบลอ"
            
            # การตรวจสอบขั้นสูงด้วย OpenCV (หากมี)
            if OPENCV_AVAILABLE:
                # แปลงภาพ PIL เป็นรูปแบบ OpenCV
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # การตรวจจับภาพเบลอแบบขั้นสูงโดยใช้ Laplacian variance
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_threshold = 100
                
                if laplacian_var < blur_threshold:
                    return False, None, None, f"❌ ภาพเบลอหรือไม่คมชัด (คะแนนความคมชัด: {laplacian_var:.1f} ต้องมีมากกว่า {blur_threshold})"
                
                # การตรวจจับขอบเพื่อตรวจสอบเนื้อหา
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                
                if edge_density < 0.05:
                    return False, None, None, "❌ ภาพไม่มีรายละเอียดเพียงพอ อาจเกิดจากการเขย่าตัวหรือการเคลื่อนไหว"
                
                # การตรวจจับเส้นขอบเพื่อหาการมีอยู่ของวัตถุ
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                significant_contours = [c for c in contours if cv2.contourArea(c) > (width * height * 0.01)]
                
                if len(significant_contours) < 1:
                    return False, None, None, "❌ ไม่พบวัตถุที่มีรูปร่างชัดเจนในภาพ กรุณาถ่ายภาพพระเครื่องให้ชัดเจน"
                
                # ตรวจสอบความอิ่มตัวของสี
                hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
                mean_saturation = np.mean(hsv[:, :, 1])
                
                if mean_saturation > 200:
                    return False, None, None, "❌ ภาพมีสีสันจัดจ้านเกินไป อาจไม่ใช่ภาพพระเครื่องจริง"
                
                # การตรวจจับสัญญาณรบกวน
                noise_level = cv2.meanStdDev(gray)[1][0][0]
                if noise_level > 50:
                    return False, None, None, f"❌ ภาพมีสัญญาณรบกวนมากเกินไป (ระดับสัญญาณรบกวน: {noise_level:.1f})"
                
                quality_score = min(100, (laplacian_var / blur_threshold) * 50 + (edge_density * 1000))
            else:
                # คะแนนคุณภาพสำรองโดยไม่ใช้ OpenCV
                quality_score = min(100, diff * 5 + (color_std / 3))
            
            # หากผ่านการตรวจสอบทั้งหมด ให้เตรียมภาพ
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG", quality=95)
            img_byte_arr.seek(0)
            
            # ข้อความแสดงความสำเร็จพร้อมคะแนนคุณภาพ
            success_msg = f"✅ ภาพผ่านการตรวจสอบคุณภาพ (คะแนน: {quality_score:.1f}/100)"
            
            return True, img, img_byte_arr, success_msg

        except Exception as e:
            return False, None, None, f"❌ เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}"

# ==========================================================
# Utility Functions
# ==========================================================
def send_predict_request(files: dict, api_url: str, timeout: int = 60):
    """ส่งคำขอการทำนายไปยัง API
    
    Args:
        files (dict): Dictionary ของไฟล์ที่จะส่ง
        api_url (str): URL ของ API
        timeout (int): Timeout ในหน่วยวินาที
    
    Returns:
        requests.Response: Response จาก API
    """
    url = api_url.rstrip("/") + "/predict"
    prepared = {}
    for k, v in files.items():
        fname, fileobj, mime = v
        try:
            fileobj.seek(0)
        except Exception:
            pass
        prepared[k] = (fname, fileobj, mime)
    return requests.post(url, files=prepared, timeout=timeout)

# ==========================================================
# กำหนดค่า - ตอนนี้ใช้ AI Model จริงแล้ว! 🚀
# ==========================================================
# ใช้พอร์ตของ Production API (ดู backend/api/production_ready_api.py -> port=8000)
API_URL = "http://127.0.0.1:8000"  # Real AI Model Backend
# หากต้องการเปลี่ยนพอร์ต โปรดอัปเดตให้สอดคล้องกับฝั่ง Backend

# ค่าเริ่มต้นสำหรับเปรียบเทียบรูปภาพ
DEFAULT_MODEL_PATH = "training_output_improved/models/best_model.pth"
DEFAULT_DATABASE_DIR = "dataset_organized" 
DEFAULT_TOP_K = 5

# กำหนดค่าสำหรับแสดงผล
st.set_page_config(
    page_title="Amulet-AI",
    layout="wide"
)
# ==========================================================
# Global CSS/JS - simplified modern theme with Thai style
# ==========================================================
st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Playfair+Display:wght@600;700&display=swap" rel="stylesheet">
<style>
:root {
    --bg-1: #f7f7f9;
    --bg-2: #ececec;
    --primary: #222;
    --secondary: #555;
    --card: #fff;
    --muted: #888;
    --accent: #e5e7eb;
    --border: #e5e7eb;
    --shadow: 0 8px 24px rgba(0,0,0,0.07);
    --radius: 14px;
    --heading-font: 'Playfair Display', 'Inter', sans-serif;
    --body-font: 'Inter', 'Noto Sans Thai', sans-serif;
    --transition: all 0.3s cubic-bezier(.4,0,.2,1);
}
/* พื้นหลัง geometric pattern (SVG base64) */
html, body, .stApp {
    min-height: 100vh;
    background:
        linear-gradient(135deg, #f7f7f9 0%, #ececec 100%),
        repeating-linear-gradient(120deg, #e5e7eb 0px, #e5e7eb 2px, transparent 2px, transparent 40px),
        repeating-linear-gradient(60deg, #e5e7eb 0px, #e5e7eb 2px, transparent 2px, transparent 40px);
    background-blend-mode: lighten;
    background-size: cover;
    color: var(--primary);
    font-family: var(--body-font);
}
/* กล่องหลักทุกประเภทให้พื้นหลังขาวทึบและเงา */
.main .block-container,
.panel,
.card,
.app-header,
.upload-zone,
.comparison-grid,
.result-container {
    background: #ebebeb;
    color: var(--primary);
    border-radius: var(--radius);
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    border: 1px solid var(--border);
    padding: 2rem 1.5rem;
    margin: 1.5rem auto;
    transition: var(--transition);
}
/* เพิ่มความเด่นให้ header */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    padding: 2rem 2.5rem;
    border-radius: var(--radius);
    background: rgba(255,255,255,0.95);
    color: var(--primary);
    box-shadow: 0 6px 24px rgba(56,189,248,0.10);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    transition: var(--transition);
}
.header-text {
    flex: 2;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.header-subblock {
    margin-top: 1rem;
}
.crumbs {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 1.5rem;
}
.crumbs img {
    height: 70px;
    margin-left: 0.5rem;
    vertical-align: middle;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    background: #fff;
    padding: 0.5rem;
}
@media (max-width: 900px) {
    .app-header {
        flex-direction: column;
        align-items: flex-start;
        padding: 1rem;
        gap: 1rem;
    }
    .crumbs {
        justify-content: flex-start;
        gap: 1rem;
    }
    .crumbs img {
        height: 48px;
    }
}
</style>
<script>
document.addEventListener('DOMContentLoaded', function(){
    const containers = document.querySelectorAll('.stTabs');
    containers.forEach(function(c){
        ['dragenter','dragover','dragleave','drop'].forEach(ev=>c.addEventListener(ev, e=>{e.preventDefault();e.stopPropagation();}));
        c.addEventListener('dragover', ()=>c.classList.add('drag-over'));
        c.addEventListener('dragleave', ()=>c.classList.remove('drag-over'));
        c.addEventListener('drop', function(e){
            c.classList.remove('drag-over');
            const dt = e.dataTransfer; if(!dt) return; const files = dt.files; if(!files || files.length===0) return;
            const file = files[0]; if(!file.type.startsWith('image/')) return;
            const input = c.querySelector('input[type=file]'); if(!input) return;
            const data = new DataTransfer(); data.items.add(file); input.files = data.files; input.dispatchEvent(new Event('change',{bubbles:true}));
        });
    });
});
</script>

""",
        unsafe_allow_html=True,
)

# (CSS code removed from Python file; it should be placed inside a string and passed to st.markdown(..., unsafe_allow_html=True) if needed)

# ==========================================================
# ส่วนหัว - การออกแบบแบบขั้นสูง
# ==========================================================
def get_base64_image(image_path):
    """Get base64 encoded image with proper path resolution"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        # Create a placeholder if logo file is missing
        st.warning(f"Logo file not found: {image_path}. Using placeholder.")
        return ""

# Use absolute paths for logo files
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_depa_path = os.path.join(current_dir, "logo_depa.png")
logo_thai_austrian_path = os.path.join(current_dir, "logo_thai_austrian.gif")

logo_depa_b64 = get_base64_image(logo_depa_path)
logo_thai_austrian_b64 = get_base64_image(logo_thai_austrian_path)

st.markdown(
    f"""
<div class="app-header">
  <div class="header-text">
    <h1>Amulet-AI</h1>
    <p>ปัญญาโบราณสำหรับพระเครื่องพุทธไทย — ข้อมูลเชิงลึกเกี่ยวกับความแท้ ความเข้าใจในรูปแบบ</p>
    <div class="header-subblock">
      <span class="badge">จำแนกประเภทแม่นยำ</span>
      <span class="badge">ประเมินราคา</span>
      <span class="badge">มรดกวัฒนธรรม</span>
    </div>
  </div>
  <div class="crumbs" style="flex:1; text-align:right;">
    <span>หน้าหลัก</span>
    <img src="data:image/png;base64,{logo_depa_b64}" alt="depa" style="height:150px; margin-left:18px; vertical-align:middle; border-radius:8px;">
    <img src="data:image/gif;base64,{logo_thai_austrian_b64}" alt="thai-austrian" style="height:150px; margin-left:12px; vertical-align:middle; border-radius:8px;">
  </div>
</div>
""",
    unsafe_allow_html=True,
)



# ==========================================================
# หน้าหลัก / บทนำ
# ==========================================================
st.markdown('<div class="panel" style="text-align:center;">', unsafe_allow_html=True)
st.markdown("## ระบบวิเคราะห์พระเครื่องอัตโนมัติ")
st.markdown('<p class="muted">ค้นพบรายละเอียดที่ซ่อนอยู่ด้วย AI</p>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# คำแนะนำคุณภาพภาพ
# ==========================================================
with st.expander("📷 คำแนะนำการถ่ายภาพที่ได้คุณภาพ", expanded=False):
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                   border-radius: 12px; padding: 1.5rem; border: 1px solid #0284c7;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;">
                
                <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 8px; border-left: 4px solid #16a34a;">
                    <h4 style="color: #15803d; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
                        <span>🎯</span> การจัดองค์ประกอบ
                    </h4>
                    <ul style="color: #166534; font-size: 0.9rem; margin: 0; padding-left: 1rem;">
                        <li>วางพระเครื่องตรงกลางเฟรม</li>
                        <li>ให้เต็มภาพ มีรายละเอียดชัดเจน</li>
                        <li>หลีกเลี่ยงการครอบด้วยนิ้ว</li>
                        <li>ใช้พื้นหลังสีเดียว เรียบง่าย</li>
                    </ul>
                </div>
                
                <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b;">
                    <h4 style="color: #d97706; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
                        <span>💡</span> แสงและความคมชัด
                    </h4>
                    <ul style="color: #92400e; font-size: 0.9rem; margin: 0; padding-left: 1rem;">
                        <li>ถ่ายในที่ที่มีแสงธรรมชาติเพียงพอ</li>
                        <li>หลีกเลี่ยงการใช้แฟลช</li>
                        <li>ตั้งกล้องให้มั่นคง ไม่เขย่า</li>
                        <li>โฟกัสให้คมชัด</li>
                    </ul>
                </div>
                
                <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 8px; border-left: 4px solid #dc2626;">
                    <h4 style="color: #dc2626; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
                        <span>🚫</span> สิ่งที่ควรหลีกเลี่ยง
                    </h4>
                    <ul style="color: #991b1b; font-size: 0.9rem; margin: 0; padding-left: 1rem;">
                        <li>ภาพเบลอหรือไม่คมชัด</li>
                        <li>แสงน้อยเกินไป หรือ สว่างเกินไป</li>
                        <li>วัตถุอื่นที่ไม่ใช่พระเครื่อง</li>
                        <li>ภาพที่มีการแต่งสีมากเกินไป</li>
                    </ul>
                </div>
                
                <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 8px; border-left: 4px solid #7c3aed;">
                    <h4 style="color: #7c3aed; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
                        <span>⚡</span> ระบบตรวจสอบอัตโนมัติ
                    </h4>
                    <ul style="color: #6b21a8; font-size: 0.9rem; margin: 0; padding-left: 1rem;">
                        <li>ตรวจสอบความคมชัดอัตโนมัติ</li>
                        <li>วิเคราะห์คุณภาพแสง</li>
                        <li>ระบุเนื้อหาในภาพ</li>
                        <li>ให้คะแนนคุณภาพภาพ</li>
                    </ul>
                </div>
                
            </div>
            
            <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem; 
                       border: 1px solid rgba(59, 130, 246, 0.3);">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.2rem;">🛡️</span>
                    <span style="font-weight: 600; color: #1e40af;">ระบบป้องกันข้อมูลไม่คุณภาพ</span>
                </div>
                <p style="color: #1e40af; font-size: 0.9rem; margin: 0;">
                    ระบบจะตรวจสอบทุกภาพก่อนการวิเคราะห์ เพื่อให้แน่ใจว่าได้ผลลัพธ์ที่แม่นยำและเชื่อถือได้
                    หากภาพไม่ผ่านเกณฑ์ ระบบจะแจ้งให้ทราบพร้อมคำแนะนำการปรับปรุง
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

# ==========================================================
# ตัวช่วยอัปโหลด (สองคอลัมน์ รักษาการไหลที่ดียิ่งขึ้นจากไฟล์#1)
# ==========================================================
st.markdown(
    f"""
<div style=\"text-align:center; margin:1.25rem 0 .5rem\"> 
  <h3 style=\"margin:.25rem 0; color:var(--color-foreground)\">อัปโหลดรูปภาพพระเครื่อง</h3>
  <p class=\"muted\">รองรับไฟล์: <code>{FORMAT_DISPLAY}</code></p>
</div>
""",
    unsafe_allow_html=True,
)

col_upload, col_camera = st.columns(2)
with col_upload:
    st.markdown(
        """
        <div class="card" style="text-align:center;">
          <h4 class="accent" style="margin:0;">อัปโหลดจากไฟล์</h4>
          <p class="muted" style="margin:.35rem 0 0;">เลือกไฟล์จากเครื่องของคุณ</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_camera:
    st.markdown(
        """
        <div class="card" style="text-align:center;">
          <h4 class="accent" style="margin:0;">ถ่ายรูปด้วยกล้อง</h4>
          <p class="muted" style="margin:.35rem 0 0;">ขอสิทธิ์เมื่อกดใช้งาน</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

col1, col2 = st.columns(2)

# ---------------- Front ----------------
with col1:
    st.markdown(
        """
        <div style="text-align:center; padding:1rem; background:#e8f5e8; border:1px solid #c3e6c3; border-radius:.75rem; margin: .75rem 0;">
          <h4 style="color:#2d5016; margin:0;">ภาพด้านหน้า</h4>
          <p style="color:#2d5016; font-size:.85rem; margin:.3rem 0 0;">(บังคับ - สำหรับการวิเคราะห์พระเครื่อง)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["อัปโหลด", "ถ่ายรูป"])

    front = None
    front_source = "upload"

    with tab1:
        st.markdown(
            """
            <div class="upload-zone">
              <div style="font-weight:700; margin-bottom:.35rem;">เลือกไฟล์จากเครื่องของคุณ</div>
              <div class="muted" style="font-size:.9rem;">Limit 200MB per file • {}</div>
            </div>
            """.format(FORMAT_DISPLAY),
            unsafe_allow_html=True,
        )
        front_file = st.file_uploader(
            "เลือกไฟล์ภาพด้านหน้า",
            type=SUPPORTED_FORMATS,
            key="front_upload",
            label_visibility="collapsed",
        )
        if front_file:
            st.button("Browse files", key="front_browse", disabled=True)
        front = front_file
        front_source = "upload"

    with tab2:
        st.markdown(
            """
            <div class="upload-zone">
              <div style="font-weight:700; margin-bottom:.35rem;">กล้อง</div>
              <div class="muted" style="font-size:.9rem;">ขอสิทธิ์เมื่อกดใช้งาน</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("เปิดกล้องถ่ายรูป", key="front_camera_btn", use_container_width=True):
            st.session_state.show_front_camera = True

        if st.session_state.get("show_front_camera", False):
            front_camera = st.camera_input("ถ่ายรูปภาพด้านหน้า", key="front_camera")
            if front_camera:
                front = front_camera
                front_source = "camera"
                if st.button("ใช้รูปนี้", key="front_camera_confirm"):
                    st.session_state.show_front_camera = False
                    st.rerun()
            else:
                front = front_file if front_file else None
                front_source = "upload"
        else:
            front = front_file if front_file else None
            front_source = "upload"

    # Validate + preview
    if front:
        is_valid, processed_img, processed_bytes, error_msg = validate_and_convert_image(front)
        if is_valid:
            # Show success with quality information
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                           border: 1px solid #34d399; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.2rem;">✅</span>
                        <div>
                            <div style="font-weight: 600; color: #047857;">ภาพด้านหน้า: ผ่านการตรวจสอบ</div>
                            <div style="font-size: 0.85rem; color: #059669; margin-top: 0.2rem;">{error_msg}</div>
                        </div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            if processed_img is not None:
                st.image(processed_img, width=300, caption="ภาพที่อัปโหลด")
            else:
                st.warning("ไม่สามารถแสดงภาพได้")
            st.session_state.front_processed = processed_bytes
            st.session_state.front_filename = (
                front.name if hasattr(front, "name") else f"camera_front_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            )
        else:
            # Show detailed error with suggestions
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                           border: 1px solid #f87171; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                    <div style="display: flex; align-items: flex-start; gap: 0.5rem;">
                        <span style="font-size: 1.2rem;">❌</span>
                        <div>
                            <div style="font-weight: 600; color: #dc2626; margin-bottom: 0.5rem;">ภาพไม่ผ่านการตรวจสอบคุณภาพ</div>
                            <div style="color: #991b1b; margin-bottom: 0.8rem;">{error_msg}</div>
                            <div style="background: rgba(255,255,255,0.8); padding: 0.8rem; border-radius: 6px; 
                                       border-left: 3px solid #f59e0b;">
                                <div style="font-weight: 600; color: #92400e; margin-bottom: 0.3rem;">💡 คำแนะนำในการถ่ายภาพที่ดี:</div>
                                <ul style="color: #92400e; font-size: 0.85rem; margin: 0; padding-left: 1.2rem;">
                                    <li>ถ่ายในที่ที่มีแสงเพียงพอ แต่ไม่ใช้แฟลช</li>
                                    <li>ตั้งกล้องให้มั่นคง ไม่เขย่า</li>
                                    <li>ให้พระเครื่องอยู่ตรงกลางภาพ</li>
                                    <li>ระยะห่างพอดี ไม่ใกล้หรือไกลเกินไป</li>
                                    <li>พื้นหลังเรียบง่าย ไม่แต่งสี</li>
                                    <li>ให้พระเครื่องเต็มเฟรม มีรายละเอียดชัดเจน</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

# ---------------- Back ----------------
with col2:
    st.markdown(
        """
        <div style="text-align:center; padding:1rem; background:#e8f5e8; border:1px solid #c3e6c3; border-radius:.75rem; margin: .75rem 0;">
          <h4 style="color:#2d5016; margin:0;">ภาพด้านหลัง</h4>
          <p style="color:#2d5016; font-size:.85rem; margin:.3rem 0 0;">(บังคับ - สำหรับการวิเคราะห์ที่ละเอียดยิ่งขึ้น)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1b, tab2b = st.tabs(["อัปโหลด", "ถ่ายรูป"])

    back = None
    back_source = "upload"

    with tab1b:
        st.markdown(
            """
            <div class="upload-zone">
              <div style="font-weight:700; margin-bottom:.35rem;">เลือกไฟล์จากเครื่องของคุณ</div>
              <div class="muted" style="font-size:.9rem;">Limit 200MB per file • {}</div>
            </div>
            """.format(FORMAT_DISPLAY),
            unsafe_allow_html=True,
        )
        back_file = st.file_uploader(
            "เลือกไฟล์ภาพด้านหลัง",
            type=SUPPORTED_FORMATS,
            key="back_upload",
            label_visibility="collapsed",
        )
        if back_file:
            st.button("Browse files", key="back_browse", disabled=True)
        back = back_file
        back_source = "upload"

    with tab2b:
        st.markdown(
            """
            <div class="upload-zone">
              <div style="font-weight:700; margin-bottom:.35rem;">กล้อง</div>
              <div class="muted" style="font-size:.9rem;">ขอสิทธิ์เมื่อกดใช้งาน</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("เปิดกล้องถ่ายรูป", key="back_camera_btn", use_container_width=True):
            st.session_state.show_back_camera = True

        if st.session_state.get("show_back_camera", False):
            back_camera = st.camera_input("ถ่ายรูปภาพด้านหลัง", key="back_camera")
            if back_camera:
                back = back_camera
                back_source = "camera"
                if st.button("ใช้รูปนี้", key="back_camera_confirm"):
                    st.session_state.show_back_camera = False
                    st.rerun()
            else:
                back = back_file if back_file else None
                back_source = "upload"
        else:
            back = back_file if back_file else None
            back_source = "upload"

    # Validate + preview
    if back:
        is_valid, processed_img, processed_bytes, error_msg = validate_and_convert_image(back)
        if is_valid:
            # Show success with quality information
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                           border: 1px solid #34d399; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1.2rem;">✅</span>
                        <div>
                            <div style="font-weight: 600; color: #047857;">ภาพด้านหลัง: ผ่านการตรวจสอบ</div>
                            <div style="font-size: 0.85rem; color: #059669; margin-top: 0.2rem;">{error_msg}</div>
                        </div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            if processed_img is not None:
                st.image(processed_img, width=300, caption="ภาพที่อัปโหลด")
            else:
                st.warning("ไม่สามารถแสดงภาพได้")
            st.session_state.back_processed = processed_bytes
            st.session_state.back_filename = (
                back.name if hasattr(back, "name") else f"camera_back_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            )
        else:
            # Show detailed error with suggestions
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                           border: 1px solid #f87171; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                    <div style="display: flex; align-items: flex-start; gap: 0.5rem;">
                        <span style="font-size: 1.2rem;">❌</span>
                        <div>
                            <div style="font-weight: 600; color: #dc2626; margin-bottom: 0.5rem;">ภาพไม่ผ่านการตรวจสอบคุณภาพ</div>
                            <div style="color: #991b1b; margin-bottom: 0.8rem;">{error_msg}</div>
                            <div style="background: rgba(255,255,255,0.8); padding: 0.8rem; border-radius: 6px; 
                                       border-left: 3px solid #f59e0b;">
                                <div style="font-weight: 600; color: #92400e; margin-bottom: 0.3rem;">💡 คำแนะนำในการถ่ายภาพที่ดี:</div>
                                <ul style="color: #92400e; font-size: 0.85rem; margin: 0; padding-left: 1.2rem;">
                                    <li>ถ่ายในที่ที่มีแสงเพียงพอ แต่ไม่ใช้แฟลช</li>
                                    <li>ตั้งกล้องให้มั่นคง ไม่เขย่า</li>
                                    <li>ให้พระเครื่องอยู่ตรงกลางภาพ</li>
                                    <li>ระยะห่างพอดี ไม่ใกล้หรือไกลเกินไป</li>
                                    <li>พื้นหลังเรียบง่าย ไม่แต่งสี</li>
                                    <li>ให้พระเครื่องเต็มเฟรม มีรายละเอียดชัดเจน</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

st.markdown("---")

# ==========================================================
# Analyze Button + Result Flow (kept from file#1 structure)
# ==========================================================
if (
    ("front_processed" in st.session_state)
    and ("back_processed" in st.session_state)
):
    st.markdown(
        """
        <div class="panel" style="text-align:center; border-left: 4px solid #28a745; background-color: #f8f9fa;">
          <h4 style="margin:.25rem 0 0; color: #212529;">พร้อมวิเคราะห์แล้ว</h4>
          <p style="margin:.25rem 0 .5rem; color: #495057;">กดปุ่มด้านล่างเพื่อเริ่มการวิเคราะห์ด้วย AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("วิเคราะห์และเปรียบเทียบตอนนี้", type="primary", use_container_width=True, key="analyze_button"):
        # Add processing animation class
        st.markdown('<script>document.querySelector("[data-testid=\'stButton\'] button").classList.add("processing-btn");</script>', unsafe_allow_html=True)
        
        files = {
            "front": (
                st.session_state.front_filename,
                st.session_state.front_processed,   
                "image/jpeg",
            ),
            "back": (
                st.session_state.back_filename,
                st.session_state.back_processed,
                "image/jpeg",
            ),
        }
        with st.spinner("กำลังประมวลผลด้วย AI... โปรดรอสักครู่"):
            try:
                r = send_predict_request(files, API_URL, timeout=60)
                if r is None:
                    st.error("ไม่สามารถเชื่อมต่อกับ API ได้")
                elif r.ok:
                    data = r.json()

                    # ---- Enhanced Result Display with animations ----
                    st.markdown("---")
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.success("วิเคราะห์เสร็จสิ้น")
                    
                    # AI Mode indicator
                    ai_mode = data.get("ai_mode", "real_model")
                    processing_time = data.get("processing_time", 0)
                    
                    col_header1, col_header2 = st.columns([3, 1])
                    with col_header1:
                        st.markdown("## ผลการวิเคราะห์หลัก")
                    with col_header2:
                        st.info(f"โหมด: {ai_mode}")
                        st.info(f"เวลา: {processing_time:.2f}s")
                    
                    # ---- Display Top 3 Results with enhanced styling ----
                    topk_results = data.get("topk", [])
                    if topk_results:
                        st.markdown("### ผลการจำแนกประเภท TOP 3")
                        
                        for i, result in enumerate(topk_results[:3]):
                            confidence = float(result.get("confidence", 0.0))
                            conf_pct = confidence * 100.0
                            class_name = result.get("class_name", "Unknown")
                            
                            # Determine confidence level and styling
                            if conf_pct >= 80:
                                confidence_class = "confidence-high"
                                confidence_label = "สูง"
                                border_color = "#16a34a"
                                bg_gradient = "linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%)"
                            elif conf_pct >= 60:
                                confidence_class = "confidence-medium"
                                confidence_label = "ปานกลาง"
                                border_color = "#d97706"
                                bg_gradient = "linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)"
                            else:
                                confidence_class = "confidence-low"
                                confidence_label = "ต่ำ"
                                border_color = "#dc2626"
                                bg_gradient = "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
                            
                            # Create result card using Streamlit components
                            rank_text = ["อันดับ 1", "อันดับ 2", "อันดับ 3"][i] if i < 3 else f"อันดับ {i+1}"

                            # Use expander for each result
                            with st.expander(f"{rank_text}: {class_name}", expanded=(i==0)):
                                col_a, col_b = st.columns([2, 1])

                                with col_a:
                                    st.markdown(f"**ความเชื่อมั่น:** {confidence_label} ({conf_pct:.1f}%)")
                                    st.markdown(f"**คะแนนความน่าจะเป็น:** {confidence:.4f}")

                                with col_b:
                                    # Create a simple progress bar
                                    st.progress(conf_pct / 100)
                                    st.caption(f"{conf_pct:.1f}%")

                    
                    
                    # ---- Performance Summary ----
                    if topk_results:
                        top_result = topk_results[0]
                        top_class = top_result.get("class_name", "")
                        top_confidence = float(top_result.get("confidence", 0.0)) * 100.0
                        
                        st.markdown("### สรุปการวิเคราะห์")
                        summary_col1, summary_col2 = st.columns(2)

                        with summary_col1:
                            st.success(f"**ผลการวินิจฉัย:** {top_class}")
                            st.info(f"**ความเชื่อมั่น:** {top_confidence:.1f}%")

                        with summary_col2:
                            quality = "เยี่ยม" if top_confidence > 85 else "ดี" if top_confidence > 70 else "พอใช้"
                            st.info(f"**คุณภาพการวิเคราะห์:** {quality}")
                            st.info(f"**เวลาประมวลผล:** {processing_time:.2f} วินาที")
                        
                        # ---- Detection Details ----
                        st.markdown("### รายละเอียดการตรวจจับ")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**ประเภทพระเครื่อง:** {top_class}")
                            st.info(f"**ความเชื่อมั่น AI:** {top_confidence:.2f}%")

                        with col2:
                            shape = top_class.split('_')[0].title() if '_' in top_class else top_class
                            st.info(f"**รูปทรง:** {shape}")
                            st.info(f"**จำนวนคลาสที่เปรียบเทียบ:** {len(topk_results)} คลาส")

                        st.markdown("**วิธีการตรวจสอบเพิ่มเติม:**")
                        st.markdown("""
                        - ตรวจสอบรายละเอียดลวดลายด้วยแว่นขยาย
                        - เปรียบเทียบน้ำหนักและขนาดกับข้อมูลมาตรฐาน
                        - ปรึกษาผู้เชี่ยวชาญพระเครื่องเพื่อยืนยันผล
                        - ตรวจสอบประวัติและแหล่งที่มาของพระเครื่อง
                        """)
                    
                    
                    
                    
                    # ---- Detection Error Analysis (if available) ----
                    if "errors" in data and data["errors"]:
                        st.markdown("### วิเคราะห์ข้อผิดพลาดในการตรวจจับ")
                        
                        errors = data["errors"]
                        
                        # Create a table for error analysis
                        error_table = "| ลำดับ | ข้อผิดพลาด | คำอธิบาย |\n|---|---|---|\n"
                        
                        for i, error in enumerate(errors):
                            error_type = error.get("type", "ไม่ระบุ")
                            error_desc = error.get("description", "ไม่มีคำอธิบาย")
                            
                            error_table += f"| {i+1} | {error_type} | {error_desc} |\n"
                        
                        st.markdown(error_table)
                    
                    
                    # ---- Performance Metrics (if available) ----
                    if "metrics" in data and data["metrics"]:
                        st.markdown("### เมตริกการประเมินผล")
                        
                        metrics = data["metrics"]
                        
                        # Create a radar chart for metrics
                        try:
                            import plotly.express as px
                            import pandas as pd

                            # Prepare data for radar chart
                            metrics_df = pd.DataFrame(metrics)

                            fig = px.line_polar(
                                metrics_df,
                                r="ค่า",
                                theta="ชื่อเมตริก",
                                line_close=True,
                                template="plotly_white",
                                title="เมตริกการประเมินผลการตรวจจับ"
                            )

                            fig.update_traces(fill="toself")

                            st.plotly_chart(fig, use_container_width=True)
                        except ImportError:
                            st.warning("plotly หรือ pandas ไม่พร้อมใช้งาน การแสดงกราฟเมตริกจะถูกข้าม")
                        except Exception as e:
                            st.warning(f"ไม่สามารถแสดงกราฟเมตริกการประเมินผลได้: {e}")
                    

                    # ---- Professional Valuation Display ----
                    st.markdown("### ประเมินราคาตลาด")
                    v = data.get("valuation", {})
                    if v:
                        # Use a styled container for price information
                        st.markdown("""
                        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 0.5rem; padding: 1rem;">
                            <h4 style="margin-top: 0; color: #495057;">ช่วงราคาประเมิน</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            low_price = v.get('p05', 0)
                            st.metric("ราคาต่ำสุด", f"฿{low_price:,}" if low_price else "–")
                        with col2:
                            mid_price = v.get('p50', 0)
                            st.metric("ราคาเฉลี่ย", f"฿{mid_price:,}" if mid_price else "–")
                        with col3:
                            high_price = v.get('p95', 0)
                            st.metric("ราคาสูงสุด", f"฿{high_price:,}" if high_price else "–")
                        
                        # Confidence indicator
                        val_confidence = v.get('confidence', 'medium')
                        confidence_text = {
                            'high': 'สูง', 
                            'medium': 'ปานกลาง', 
                            'low': 'ต่ำ'
                        }.get(val_confidence, 'ไม่ระบุ')
                        
                        st.info(f"ความเชื่อมั่นในการประเมิน: **{confidence_text}**")
                        
                        # Add pricing notes
                        if v.get('notes'):
                            st.markdown("**หมายเหตุเกี่ยวกับราคา:**")
                            st.markdown(v.get('notes'))
                    else:
                        st.warning("ไม่มีข้อมูลการประเมินราคาสำหรับพระเครื่องนี้")

                    # ---- Professional Recommendations ----
                    st.markdown("### แนะนำตลาดและช่องทางการขาย")
                    recs = data.get("recommendations", [])
                    if recs:
                        for i, rec in enumerate(recs):
                            market_name = rec.get("market", "Market")
                            rating = rec.get("rating", 0)
                            distance = rec.get("distance", 0)
                            
                            # Market type description
                            market_type = "ออนไลน์" if distance == 0 else "ออฟไลน์"
                            rating_text = f"{rating}/5.0"
                            
                            with st.expander(f"{market_name} - {rating_text} ({market_type})", expanded=(i==0)):
                                st.markdown(f"""
                                <div style="padding: 10px 0;">
                                    <p><strong>เหตุผล:</strong> {rec.get('reason','')}</p>
                                    {'<p><strong>ระยะทาง:</strong> ' + str(distance) + ' กิโลเมตร</p>' if distance > 0 else '<p><strong>ประเภท:</strong> ออนไลน์</p>'}
                                    <p><strong>คะแนนแนะนำ:</strong> {rating}/5.0</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("ยังไม่มีคำแนะนำตลาดในขณะนี้")
                        
                    # Timestamp info
                    timestamp = data.get("timestamp", "")
                    if timestamp:
                        st.caption(f"วิเคราะห์เมื่อ: {timestamp}")
                        
                else:
                    st.error(f"เกิดข้อผิดพลาดจาก API: {r.status_code}")
                    st.markdown(f"""
                    <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 0.5rem; padding: 1rem;">
                        <h4 style="margin-top: 0; color: #721c24;">รายละเอียดข้อผิดพลาด</h4>
                        <pre style="background: #f5c6cb; padding: 10px; border-radius: 0.25rem;">{r.text}</pre>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except requests.exceptions.Timeout:
                st.warning("การประมวลผลใช้เวลานานเกินไป กรุณาลองใหม่อีกครั้งหรือลดขนาดไฟล์")
            except requests.exceptions.ConnectionError:
                st.error("ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ได้ กรุณาตรวจสอบว่า Backend ทำงานอยู่ที่พอร์ต 8000 หรือ 8001")
            except Exception as e:
                st.error("เกิดข้อผิดพลาดที่ไม่คาดคิด")
                st.markdown(f"""
                <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 0.5rem; padding: 1rem;">
                    <h4 style="margin-top: 0; color: #721c24;">รายละเอียดข้อผิดพลาด</h4>
                    <pre style="background: #f5c6cb; padding: 10px; border-radius: 0.25rem;">{str(e)}</pre>
                </div>
                """, unsafe_allow_html=True)
else:
    # Missing inputs guidance (kept concise)
    missing = []
    if "front_processed" not in st.session_state:
        missing.append("ภาพด้านหน้า")
    if "back_processed" not in st.session_state:
        missing.append("ภาพด้านหลัง")
    st.info("กรุณาอัปโหลด " + " และ ".join(missing) + " เพื่อเริ่มวิเคราะห์")

# ==========================================================
# เปรียบเทียบรูปภาพ - ประกาศฟังก์ชันก่อนเรียกใช้
# ==========================================================
def show_comparison_tab():
    """แสดงส่วนการเปรียบเทียบรูปภาพ"""
    st.markdown('<h1 style="text-align: center; margin-bottom: 1rem; color: #1E3A8A;">📸 ระบบเปรียบเทียบรูปภาพพระเครื่อง</h1>', unsafe_allow_html=True)
    
    # โหลดค่าเริ่มต้น
    config = {
       
       
        "model_path": DEFAULT_MODEL_PATH,
        "database_dir": DEFAULT_DATABASE_DIR,
        "top_k": DEFAULT_TOP_K
    }
    
    # โหลดค่าจาก config.json ถ้ามี
    config_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                for key in ["model_path", "database_dir", "top_k"]:
                    if key in loaded_config:
                        config[key] = loaded_config[key]
        except Exception as e:
            st.warning(f"ไม่สามารถโหลดไฟล์ config.json: {e}")
    
    # การตั้งค่าการเปรียบเทียบ  
    with st.expander("⚙️ การตั้งค่าการเปรียบเทียบรูปภาพ", expanded=False):
        st.markdown("### การตั้งค่าเปรียบเทียบรูปภาพ")
        
        # Model selection
        model_path = st.text_input(
            "ที่อยู่ไฟล์โมเดล",
            value=config["model_path"]
        )
        
        # Database selection
        database_dir = st.text_input(
            "ที่อยู่ฐานข้อมูลรูปภาพ",
            value=config["database_dir"]
        )
        
        # Top-k selection
        top_k = st.slider(
            "จำนวนภาพที่เหมือนที่สุดที่ต้องการแสดง",
            min_value=1,
            max_value=10,
            value=config["top_k"]
        )
        
        # Save config button
        if st.button("บันทึกการตั้งค่า"):
            new_config = {
                "model_path": model_path,
                "database_dir": database_dir,
                "top_k": top_k
            }
            
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(new_config, f, indent=2, ensure_ascii=False)
                st.success("บันทึกการตั้งค่าเรียบร้อยแล้ว!")
            except Exception as e:
                st.error(f"ไม่สามารถบันทึกการตั้งค่า: {e}")
        
        # Instructions
        st.markdown("""
        <div style="background-color: #EFF6FF; border-radius: 10px; padding: 1rem; margin-top: 1rem; border: 1px solid #BFDBFE;">
            <h3>วิธีใช้งาน</h3>
            <ol>
                <li>อัพโหลดรูปภาพพระเครื่องที่ต้องการเปรียบเทียบ</li>
                <li>รอระบบวิเคราะห์และค้นหาภาพที่คล้ายกัน</li>
                <li>ระบบจะแสดงผลการเปรียบเทียบและค่าความเหมือนของแต่ละภาพ</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Information about similarity score
        st.markdown("""
        <div style="background-color: #EFF6FF; border-radius: 10px; padding: 1rem; margin-top: 1rem; border: 1px solid #BFDBFE;">
            <h3>ค่าความเหมือน</h3>
            <p><span style="color: #10B981; font-weight: bold;">0.85 - 1.00</span>: เหมือนมาก</p>
            <p><span style="color: #F59E0B; font-weight: bold;">0.70 - 0.84</span>: เหมือนปานกลาง</p>
            <p><span style="color: #EF4444; font-weight: bold;">0.00 - 0.69</span>: เหมือนน้อย</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown('<h2 style="color: #2563EB; margin-top: 1rem; margin-bottom: 1rem;">อัพโหลดรูปภาพพระเครื่องที่ต้องการเปรียบเทียบ</h2>', unsafe_allow_html=True)
    
    # Upload image
    uploaded_file = st.file_uploader("เลือกรูปภาพ", type=["jpg", "jpeg", "png"], key="comparison_uploader")
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Save image temporarily
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        temp_dir = Path(root_dir) / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / "temp_upload.jpg"
        image.save(temp_path)
        
        st.markdown('<div style="border: 1px solid #E5E7EB; border-radius: 10px; padding: 0.5rem; background-color: #F9FAFB;">', unsafe_allow_html=True)
        st.image(image, caption="รูปภาพที่อัพโหลด", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Compare button
        if st.button("เปรียบเทียบรูปภาพ", key="compare_btn"):
            with st.spinner("กำลังวิเคราะห์รูปภาพ..."):
                try:
                    # Prepare paths
                    model_path_abs = Path(root_dir) / model_path
                    database_dir_abs = Path(root_dir) / database_dir
                    
                    # Initialize image comparer
                    comparer = ImageComparer(model_path_abs, database_dir_abs)
                    
                    # Compare image
                    start_time = time.time()
                    result = comparer.compare_image(temp_path, top_k=top_k)
                    elapsed_time = time.time() - start_time
                    
                    # Display results
                    st.markdown(f'<h2 style="color: #2563EB; margin-top: 1rem; margin-bottom: 1rem;">ผลการเปรียบเทียบ (ใช้เวลา {elapsed_time:.2f} วินาที)</h2>', unsafe_allow_html=True)
                    
                    # Create plot for comparison
                    def plot_comparison(query_img, match_results):
                        """Create a matplotlib figure for comparison"""
                        matplotlib, plt_local = get_matplotlib()
                        if plt_local is None:
                            return None

                        n_matches = len(match_results)
                        fig, axes = plt_local.subplots(1, n_matches + 1, figsize=(12, 4))

                        # Show query image
                        axes[0].imshow(query_img)
                        axes[0].set_title("ภาพที่อัพโหลด")
                        axes[0].axis('off')

                        # Show matches
                        for i, match in enumerate(match_results):
                            img = Image.open(match["path"]).convert('RGB')
                            similarity = match["similarity"]
                            class_name = match["class"]

                            axes[i+1].imshow(img)
                            axes[i+1].set_title(f"{class_name}\nความเหมือน: {similarity:.2f}")
                            axes[i+1].axis('off')

                        plt_local.tight_layout()

                        # Convert plot to image
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        plt_local.close(fig)

                        return buf
                    
                   
                    # Display comparison results
                    def get_similarity_class(similarity):
                        """Classify similarity score into categories"""
                        if similarity >= 0.85:
                            return "high"
                        elif similarity >= 0.7:
                            return "medium"
                        else:
                            return "low"
                    
                    # Create columns for results
                    for i, match in enumerate(result["top_matches"]):
                        similarity = match["similarity"]
                        similarity_class = get_similarity_class(similarity)
                        similarity_color = "#10B981" if similarity_class == "high" else "#F59E0B" if similarity_class == "medium" else "#EF4444"
                        
                        st.markdown(f"""
                        <div style="background-color: #EFF6FF; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; border: 1px solid #BFDBFE;">
                            <h4>{i+1}. {match['class']}</h4>
                            <p>ค่าความเหมือน: <span style="color: {similarity_color}; font-weight: bold;">{similarity:.4f}</span></p>
                        </div>
 """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการเปรียบเทียบรูปภาพ: {e}")
                    logging.error(f"Error in image comparison: {e}", exc_info=True)
                finally:
                    # Remove temporary file
                    if temp_path.exists():
                        try:
                            os.remove(temp_path)
                        except:
                            pass

# ==========================================================
# Developer Info (single block)
# ==========================================================
with st.expander("ข้อมูลสำหรับนักพัฒนา"):
    st.markdown(
        f"""
<div style="font-family: monospace; background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border: 1px solid #dee2e6;">
<p><strong>API Endpoint:</strong> <code>{API_URL}</code></p>  
<p><strong>Framework:</strong> Streamlit + FastAPI</p>  
<p><strong>Last updated:</strong> {datetime.now():%Y-%m-%d %H:%M}</p>
</div>
""", unsafe_allow_html=True
    )
    
    # API connection diagnostic
    if st.button("ทดสอบการเชื่อมต่อกับ API"):
        try:
            health_response = requests.get(f"{API_URL}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("API พร้อมใช้งาน")
            else:
                st.error(f"API ส่งค่า error: {health_response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("ไม่สามารถเชื่อมต่อกับ API ได้")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {str(e)}")

# ==========================================================
# เครดิตท้ายเว็บ
# ==========================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #e0f2fe 0%, #f1f5f9 100%);
            border-radius: 12px; padding: 1.5rem; border: 1px solid #f1f5f9; margin-top: 2rem; text-align: center;">
    <h4 style="color: #374151; margin-top: 0;">ขอขอบคุณ</h4>
    <p style="font-size: 1.1rem; color: #374151; margin-bottom: 0.5rem;">
        คณะกรรมการจากสำนักงานส่งเสริมเศรษฐกิจดิจิทัล (depa)<br>
        ที่ได้มอบโอกาสอันมีค่าให้แก่ทีม <strong>Taxes1112</strong> จากวิทยาลัยเทคนิคสัตหีบ<br>
        ในการเข้าร่วมโครงการและนำเสนอผลงานด้านนวัตกรรมดิจิทัลในครั้งนี้
    </p>
    <p style="color: #64748b; font-size: 1rem;">
        ซึ่งนับเป็นประสบการณ์ที่สำคัญในการพัฒนาศักยภาพของนักศึกษา<br>
        และเป็นแรงบันดาลใจในการต่อยอดความรู้ไปสู่การสร้างสรรค์ผลงานในอนาคต
    </p>
</div>
""", unsafe_allow_html=True)