import streamlit as st
import requests
from datetime import datetime
from PIL import Image
import io
import os
import sys
from typing import Any

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

st.set_page_config(
    page_title=" Amulet-AI",
    layout="wide",
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
  <div class="crumbs"><span>Dashboard</span></div>
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
                        if top_class in ref_images and ref_images[top_class]:
                            st.markdown(f"**ตัวอย่างพระเครื่อง {top_class}**")
                            st.image(
                                ref_images[top_class], 
                                caption=f"ตัวอย่างพระเครื่อง: {top_class}",
                                width=300
                            )
                            
                            # Add comparison columns for user image and reference image
                            st.markdown("### เปรียบเทียบกับพระเครื่องของคุณ")
                            col_user, col_ref = st.columns(2)
                            
                            with col_user:
                                st.markdown("#### พระเครื่องของคุณ")
                                # Using the processed front image stored in session state
                                if "front_processed" in st.session_state:
                                    front_img = Image.open(st.session_state.front_processed)
                                    st.image(front_img, width=300)
                                
                            with col_ref:
                                st.markdown(f"#### ตัวอย่าง {top_class}")
                                st.image(ref_images[top_class], width=300)
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