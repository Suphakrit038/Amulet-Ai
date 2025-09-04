import streamlit as st
import requests
from datetime import datetime
from PIL import Image
import io
import os
import sys
import base64
from io import BytesIO
from typing import Any
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

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
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from frontend.utils import (
        validate_and_convert_image,
        send_predict_request,
        SUPPORTED_FORMATS,
        FORMAT_DISPLAY,
    )
except Exception:
    # ---- Fallbacks from the first file ----
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

    MAX_FILE_SIZE_MB = 10

    def validate_and_convert_image(uploaded_file):
        """Validate uploaded image, enforce size and extension limits, convert to RGB JPEG bytes."""
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
                return False, None, None, "Empty file or unreadable upload"

            if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
                return False, None, None, f"File too large (> {MAX_FILE_SIZE_MB} MB)"

            filename = getattr(uploaded_file, "name", "") or ""
            if filename:
                ext = filename.rsplit(".", 1)[-1].lower()
                if ext not in SUPPORTED_FORMATS:
                    return False, None, None, f"Unsupported file extension: .{ext}"

            img = Image.open(io.BytesIO(file_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG", quality=95)
            img_byte_arr.seek(0)

            return True, img, img_byte_arr, None
        except Exception as e:
            return False, None, None, str(e)

    def send_predict_request(files: dict, api_url: str, timeout: int = 60):
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
# Config - ตอนนี้ใช้ AI Model จริงแล้ว! 🚀
# ==========================================================
API_URL = "http://127.0.0.1:8001"  # Real AI Model Backend  
# API_URL = "http://127.0.0.1:8000"  # Mock API (หากต้องการทดสอบ)

# ค่าเริ่มต้นสำหรับเปรียบเทียบรูปภาพ
DEFAULT_MODEL_PATH = "training_output_improved/models/best_model.pth"
DEFAULT_DATABASE_DIR = "dataset_organized" 
DEFAULT_TOP_K = 5

# กำหนดค่าสำหรับแสดงผล
st.set_page_config(
    page_title=" Amulet-AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Global CSS (kept base from file#1, add light header classes from file#2)
# ==========================================================
st.markdown(
    """
<style>
/* ---- Light theme vars (from file#2) ---- */
:root {
  --color-background:#fdfdfd; --color-foreground:#1a1a1c;
  --color-card:#ffffff; --color-card-foreground:#1a1a1c;
  --color-muted:#f7f7f9; --color-muted-foreground:#555;
  --color-accent:#d4af37; --color-accent-foreground:#0a0a0b;
  --color-border:#e8e8ec; --color-input:#fff; --color-ring:#d4af37;
  --color-success:#16a34a; --color-warning:#d97706; --color-danger:#dc2626;
  --radius:.75rem; --radius-lg:1rem; --shadow-lg:0 8px 24px rgba(0,0,0,.08);
}
body { background:var(--color-background); color:var(--color-foreground); font-family:'Inter', system-ui, sans-serif; }
h1,h2,h3,h4 { font-family:'Playfair Display', serif; letter-spacing:-.02em; }
.muted{ color:var(--color-muted-foreground); }
.accent{ color:var(--color-accent); }
.card { background:var(--color-card); border:1px solid var(--color-border); border-radius:.75rem; padding:1rem; box-shadow:var(--shadow-lg); }
.panel{ background:#fff; border:1px solid var(--color-border); border-radius:.75rem; padding:1rem; }

/* ---- Expanded header classes (from file#2) ---- */
.app-header { display:flex; align-items:center; gap:1rem; padding:1rem 1.25rem; background:#fff; border:1px solid var(--color-border); border-radius:.75rem; }
.header-text h1 { margin:.1rem 0; font-size:2rem; }
.header-text p { margin:0; font-size:.95rem; color:var(--color-muted-foreground) }
.header-subblock { display:flex; gap:1rem; margin-top:.35rem; flex-wrap:wrap; }
.badge { display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .5rem; border-radius:.5rem; background: #fff7e0; border:1px solid #f3e2a6; color:#7a5b00; font-size:.8rem; }
.crumbs { margin-left:auto; color:var(--color-muted-foreground); font-size:.95rem; display:flex; gap:.5rem; align-items:center; }

/* ---- Animated/visual styles (kept from file#1) ---- */
.block-container { max-width:95% !important; padding-left:2rem !important; padding-right:2rem !important; }
.upload-section, .result-card, .tips-container, .tip-card { width:100% !important; max-width:none !important; }
/* (The rest of file#1's long animations omitted for brevity; kept functional ones below) */
.upload-zone{ background:#fff; border:2px dashed rgba(212,175,55,.35); border-radius:.75rem; padding:1.25rem; text-align:center; transition:.25s ease; }
.upload-zone:hover{ border-color: rgba(212,175,55,.6); background: #fdf9f2; transform: translateY(-2px); }
.hr { border-top:1px solid var(--color-border); margin:1.25rem 0; }
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Playfair+Display:wght@600;700&display=swap" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

# ==========================================================
# Header (use file#2's expanded copy but keep overall layout from file#1)
# ==========================================================
st.markdown(
    """
<div class="app-header">
  <div class="header-text">
    <h1> Amulet-AI</h1>
    <p>
ปัญญาโบราณสำหรับพระเครื่องพุทธไทย — ข้อมูลเชิงลึกเกี่ยวกับความแท้ ความเข้าใจในรูปแบบ</p>
    <div class="header-subblock">
      <span class="badge">Accurate Classification</span>
      <span class="badge">Price Estimation</span>
      <span class="badge">Cultural Heritage</span>
    </div>
  </div>
  <div class="crumbs">
    <span>หน้าหลัก</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)



# ==========================================================
# Hero / Intro
# ==========================================================
st.markdown('<div class="panel" style="text-align:center;">', unsafe_allow_html=True)
st.markdown("## ระบบวิเคราะห์พระเครื่องอัตโนมัติ")
st.markdown('<p class="muted">ค้นพบรายละเอียดที่ซ่อนอยู่ด้วย AI</p>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# Upload Helpers (two columns, keep richer flow from file#1)
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
            st.success("ภาพด้านหน้า: พร้อมใช้งาน")
            st.image(processed_img, width=300, caption=f"ภาพด้านหน้า ({front_source})")
            st.session_state.front_processed = processed_bytes
            st.session_state.front_filename = (
                front.name if hasattr(front, "name") else f"camera_front_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            )
        else:
            st.error(f"ไฟล์ภาพไม่ถูกต้อง: {error_msg}")

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
            st.success("ภาพด้านหลัง: พร้อมใช้งาน")
            st.image(processed_img, width=300, caption=f"ภาพด้านหลัง ({back_source})")
            st.session_state.back_processed = processed_bytes
            st.session_state.back_filename = (
                back.name if hasattr(back, "name") else f"camera_back_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            )
        else:
            st.error(f"ไฟล์ภาพไม่ถูกต้อง: {error_msg}")

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
    if st.button("วิเคราะห์ตอนนี้", type="primary", use_container_width=True):
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
                if r.ok:
                    data = r.json()

                    # ---- Professional Result Display ----
                    st.markdown("---")
                    st.success("วิเคราะห์เสร็จสิ้น")
                    
                    # AI Mode indicator
                    ai_mode = data.get("ai_mode", "mock_data")
                    processing_time = data.get("processing_time", 0)
                    
                    col_header1, col_header2 = st.columns([3, 1])
                    with col_header1:
                        st.markdown("## ผลการวิเคราะห์หลัก")
                    with col_header2:
                        st.info(f"โหมด: {ai_mode}")
                        st.info(f"เวลา: {processing_time:.2f}s")
                    
                    top1 = data.get("top1", {})
                    conf_pct = float(top1.get("confidence", 0.0)) * 100.0
                    class_name = top1.get("class_name", "Unknown")
                    
                    # Professional confidence display
                    confidence_label = "สูง" if conf_pct > 80 else "ปานกลาง" if conf_pct > 60 else "ต่ำ"
                    confidence_color = "#28a745" if conf_pct > 80 else "#ffc107" if conf_pct > 60 else "#dc3545"
                    
                    st.markdown(f"""
                    <div style="border-left: 5px solid {confidence_color}; padding-left: 15px;">
                        <h3 style="margin-top: 0;"><strong>{class_name}</strong></h3>
                        <p>ความเชื่อมั่น: <strong>{confidence_label}</strong> ({conf_pct:.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # ---- Professional Top-K Table ----
                    st.markdown("### รายงานความน่าจะเป็นทั้งหมด")
                    topk_data = []
                    for i, item in enumerate(data.get("topk", [])[:3], 1):
                        p = float(item.get("confidence", 0.0)) * 100.0
                        rank = f"อันดับ {i}"
                        topk_data.append({
                            "อันดับ": rank,
                            "พระเครื่อง": item.get('class_name','—'),
                            "ความน่าจะเป็น": f"{p:.1f}%",
                            "คะแนน": f"{item.get('confidence', 0):.3f}"
                        })
                    
                    if topk_data:
                        st.table(topk_data)

                    # Add reference images section
                    st.markdown("### ภาพตัวอย่างพระเครื่อง")
                    
                    # Check if reference images are available in the API response
                    if "reference_images" in data and data["reference_images"]:
                        ref_images = data.get("reference_images", {})
                        
                        # Display reference image for top prediction
                        top_class = top1.get("class_name", "")
                        if ref_images:
                            st.markdown("### เปรียบเทียบกับพระเครื่องในฐานข้อมูล")
                            
                            # แยกรูปอ้างอิงตามมุมมอง (front/back)
                            front_ref_images = {}
                            back_ref_images = {}
                            
                            for key, ref_data in ref_images.items():
                                view_type = ref_data.get("view_type", "unknown")
                                if view_type == "front":
                                    front_ref_images[key] = ref_data
                                elif view_type == "back":
                                    back_ref_images[key] = ref_data
                                else:
                                    # ถ้าไม่ระบุมุมมอง ให้ใส่ในกลุ่ม front เป็นค่าเริ่มต้น
                                    front_ref_images[key] = ref_data
                            
                            # แสดงการเปรียบเทียบด้านหน้า
                            st.markdown("#### เปรียบเทียบด้านหน้า")
                            col_user_front, col_ref_front = st.columns(2)
                            
                            with col_user_front:
                                st.markdown("##### พระเครื่องของคุณ")
                                # Using the processed front image stored in session state
                                if "front_processed" in st.session_state:
                                    front_img = Image.open(st.session_state.front_processed)
                                    st.image(front_img, width=300)
                                
                            with col_ref_front:
                                st.markdown(f"##### ตัวอย่าง {top_class}")
                                # ตรวจสอบว่ามีรูปอ้างอิงด้านหน้าหรือไม่
                                if front_ref_images:
                                    first_key = list(front_ref_images.keys())[0]
                                    ref_data = front_ref_images[first_key]
                                    
                                    # แปลง base64 เป็นรูปภาพ
                                    if "image_b64" in ref_data:
                                        img_bytes = base64.b64decode(ref_data["image_b64"])
                                        img = Image.open(BytesIO(img_bytes))
                                        st.image(img, width=300)
                                        st.caption(f"ไฟล์: {ref_data.get('filename', 'ไม่ทราบชื่อ')}")
                                    else:
                                        st.info("ไม่สามารถแสดงรูปอ้างอิงด้านหน้าได้")
                                else:
                                    st.info(f"ไม่มีรูปอ้างอิงด้านหน้าสำหรับ {top_class}")
                            
                            # แสดงการเปรียบเทียบด้านหลัง (ถ้ามี)
                            if "back_processed" in st.session_state and back_ref_images:
                                st.markdown("#### เปรียบเทียบด้านหลัง")
                                col_user_back, col_ref_back = st.columns(2)
                                
                                with col_user_back:
                                    st.markdown("##### พระเครื่องของคุณ")
                                    back_img = Image.open(st.session_state.back_processed)
                                    st.image(back_img, width=300)
                                
                                with col_ref_back:
                                    st.markdown(f"##### ตัวอย่าง {top_class}")
                                    first_key = list(back_ref_images.keys())[0]
                                    ref_data = back_ref_images[first_key]
                                    
                                    if "image_b64" in ref_data:
                                        img_bytes = base64.b64decode(ref_data["image_b64"])
                                        img = Image.open(BytesIO(img_bytes))
                                        st.image(img, width=300)
                                        st.caption(f"ไฟล์: {ref_data.get('filename', 'ไม่ทราบชื่อ')}")
                                    else:
                                        st.info("ไม่สามารถแสดงรูปอ้างอิงด้านหลังได้")
                            
                            # แสดงรูปเพิ่มเติม (ถ้ามี)
                            remaining_refs = {k: v for k, v in ref_images.items() 
                                             if k not in list(front_ref_images.keys())[:1] + list(back_ref_images.keys())[:1]}
                            
                            if remaining_refs:
                                st.markdown("#### รูปอ้างอิงเพิ่มเติม")
                                
                                # แบ่งเป็นแถวละ 3 รูป
                                for i in range(0, len(remaining_refs), 3):
                                    chunk = list(remaining_refs.items())[i:i+3]
                                    cols = st.columns(len(chunk))
                                    
                                    for j, (key, ref_data) in enumerate(chunk):
                                        with cols[j]:
                                            if "image_b64" in ref_data:
                                                img_bytes = base64.b64decode(ref_data["image_b64"])
                                                img = Image.open(BytesIO(img_bytes))
                                                st.image(img, width=200)
                                                view_type = ref_data.get("view_type", "unknown")
                                                st.caption(f"มุมมอง: {view_type}")
                            
                            # ข้อมูลเพิ่มเติมเกี่ยวกับพระเครื่อง
                            st.markdown("#### ข้อมูลเพิ่มเติม")
                            st.markdown(f"""
                            พระเครื่องประเภท **{top_class}** มีลักษณะเฉพาะตัวดังนี้:
                            
                            - รูปทรง: {top_class.split('_')[0].capitalize()}
                            - ลักษณะพิเศษ: มีความคล้ายกับภาพอ้างอิงร้อยละ {conf_pct:.1f}
                            - วัสดุที่ใช้: ดินผสมใบโพธิ์
                            - แหล่งที่มา: ผลิตโดยช่างฝีมือที่มีประสบการณ์
                            """)
                        else:
                            st.info(f"ไม่มีภาพตัวอย่างสำหรับ {top_class} ในฐานข้อมูล")
                    else:
                        st.info("ระบบยังไม่มีภาพตัวอย่างสำหรับการเปรียบเทียบ")

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
    
    # Sidebar สำหรับตั้งค่า
    with st.sidebar:
        st.title("⚙️ การตั้งค่าเปรียบเทียบรูปภาพ")
        
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
                        n_matches = len(match_results)
                        fig, axes = plt.subplots(1, n_matches + 1, figsize=(12, 4))
                        
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
                        
                        plt.tight_layout()
                        
                        # Convert plot to image
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        plt.close(fig)
                        
                        return buf
                    
                    # Plot comparison
                    comparison_img = plot_comparison(image, result["top_matches"])
                    st.image(comparison_img, use_column_width=True)
                    
                    # Display table of results
                    st.markdown('<h3 style="color: #2563EB; margin-top: 1rem; margin-bottom: 1rem;">รายละเอียดความเหมือน</h3>', unsafe_allow_html=True)
                    
                    # Get similarity class function
                    def get_similarity_class(similarity):
                        """Get CSS class for similarity score"""
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
    else:
        # Display sample or instructions
        st.markdown("""
        <div style="background-color: #EFF6FF; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; border: 1px solid #BFDBFE;">
            <h3>กรุณาอัพโหลดรูปภาพพระเครื่องที่ต้องการเปรียบเทียบ</h3>
            <p>ระบบจะวิเคราะห์และค้นหาภาพที่คล้ายกันจากฐานข้อมูล</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    # แสดงหน้าเปรียบเทียบรูปภาพ
    show_comparison_tab()

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
            with st.spinner("กำลังทดสอบการเชื่อมต่อ..."):
                health_response = requests.get(f"{API_URL}/health", timeout=5)
                if health_response.status_code == 200:
                    st.success(f"เชื่อมต่อสำเร็จ! API พร้อมใช้งาน - Status: {health_response.status_code}")
                    st.json(health_response.json())
                else:
                    st.error(f"เชื่อมต่อสำเร็จแต่ API ส่งค่า error: {health_response.status_code}")
                    st.code(health_response.text)
        except requests.exceptions.ConnectionError:
            st.error("ไม่สามารถเชื่อมต่อกับ API ได้ - กรุณาตรวจสอบว่า backend API เปิดใช้งานอยู่ที่ " + API_URL)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {str(e)}")

# ==========================================================
# เปรียบเทียบรูปภาพ
# ==========================================================

# ==========================================================
# Main application UI
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
    
    # Sidebar สำหรับตั้งค่า
    with st.sidebar:
        st.title("⚙️ การตั้งค่าเปรียบเทียบรูปภาพ")
        
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
                        n_matches = len(match_results)
                        fig, axes = plt.subplots(1, n_matches + 1, figsize=(12, 4))
                        
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
                        
                        plt.tight_layout()
                        
                        # Convert plot to image
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        plt.close(fig)
                        
                        return buf
                    
                    # Plot comparison
                    comparison_img = plot_comparison(image, result["top_matches"])
                    st.image(comparison_img, use_column_width=True)
                    
                    # Display table of results
                    st.markdown('<h3 style="color: #2563EB; margin-top: 1rem; margin-bottom: 1rem;">รายละเอียดความเหมือน</h3>', unsafe_allow_html=True)
                    
                    # Get similarity class function
                    def get_similarity_class(similarity):
                        """Get CSS class for similarity score"""
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
    else:
        # Display sample or instructions
        st.markdown("""
        <div style="background-color: #EFF6FF; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; border: 1px solid #BFDBFE;">
            <h3>กรุณาอัพโหลดรูปภาพพระเครื่องที่ต้องการเปรียบเทียบ</h3>
            <p>ระบบจะวิเคราะห์และค้นหาภาพที่คล้ายกันจากฐานข้อมูล</p>
        </div>
        """, unsafe_allow_html=True)

# เพิ่มแท็บสำหรับแยกส่วนวิเคราะห์และเปรียบเทียบ
tab1, tab2 = st.tabs(["📊 วิเคราะห์พระเครื่อง", "🔍 เปรียบเทียบรูปภาพ"])

with tab1:
    # ==========================================================
    # Hero / Intro
    # ==========================================================
    st.markdown('<div class="panel" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown("## ระบบวิเคราะห์พระเครื่องอัตโนมัติ")
    st.markdown('<p class="muted">ค้นพบรายละเอียดที่ซ่อนอยู่ด้วย AI</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)