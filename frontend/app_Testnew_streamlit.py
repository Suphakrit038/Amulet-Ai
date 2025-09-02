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
# Config (prefer the first file's constants)
# ==========================================================
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Amulet-AI",
    page_icon="⟐",
    layout="wide",
    initial_sidebar_state="expanded",
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
.logo { width:44px; height:44px; border-radius:12px; display:grid; place-items:center; background:var(--color-accent); color:#fff; font-weight:800; }
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
  <div class="logo">⟐</div>
  <div class="header-text">
    <h1>Amulet-AI</h1>
    <p>Ancient Intelligence for Thai Buddhist Amulets — authenticity insights, pattern understanding, and market guidance.</p>
    <div class="header-subblock">
      <span class="badge">Accurate Classification</span>
      <span class="badge">Price Estimation</span>
      <span class="badge">Cultural Heritage</span>
    </div>
  </div>
  <div class="crumbs"><span>Dashboard</span><span>›</span><span style="color:var(--color-foreground)">Analysis</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# Sidebar (kept from file#1; minor tidy)
# ==========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem .75rem; background:#fff; border:1px solid var(--color-border); border-radius:.75rem; margin-bottom:1rem;">
            <h3 style="margin:.25rem 0 0; color:var(--color-accent)">คู่มือใช้งาน</h3>
            <p class="muted" style="margin:.35rem 0 0;">ปัญญาโบราณ & AI ยุคใหม่</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🎯 วิธีการใช้งาน", expanded=True):
        st.markdown(
            """
**1. เตรียมภาพ**  
- ภาพด้านหน้า (บังคับ)  
- ภาพด้านหลัง (บังคับ)

**2. อัปโหลดภาพ**  
- คลิกปุ่ม "อัปโหลด" หรือ  
- ใช้กล้องถ่ายภาพใหม่

**3. ตรวจสอบภาพ**  
- ดูตัวอย่างภาพ  
- ตรวจสอบความชัด

**4. วิเคราะห์**  
- กดปุ่ม "วิเคราะห์ตอนนี้"  
- รอระบบประมวลผล (30–60 วินาที)

**5. ผลลัพธ์**  
- การจำแนกประเภท  
- ประเมินราคา  
- ข้อแนะนำการขาย
"""
        )

    with st.expander("⚡ ข้อมูลระบบ"):
        st.markdown(
            """
**เทคโนโลยี**  
- TensorFlow 2.x (Transfer Learning)  
- FastAPI (Python 3.9+)  
- Streamlit UI (Responsive)

**ฟีเจอร์**  
- ✅ Top-3 Classification  
- ✅ Valuation Range  
- ✅ Market Recommendation

**ประสิทธิภาพ**  
- ~85% accuracy • 30–60s • 5,000+ DB
"""
        )

    with st.expander("📷 เคล็ดลับถ่ายรูป"):
        st.markdown(
            """
**แสง**: ใช้แสงธรรมชาติ หลีกเลี่ยงเงาเข้ม/แฟลช  
**มุม**: ถ่ายตรง 90° ระยะ 20–30 ซม.  
**พื้นหลัง**: สีเรียบ ไม่สะท้อน
"""
        )

    with st.expander("⚠️ ข้อจำกัดและข้อควรรู้"):
        st.info("สถานะระบบ: เบต้า • ผลลัพธ์เพื่ออ้างอิง • ภาพถูกลบหลังวิเคราะห์")

    st.markdown("---")
    st.markdown("#### สถิติการใช้งาน")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("วันนี้", "247", "+42")
    with c2:
        st.metric("ความแม่นยำ", "87.2%", "+3.1%")

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
            st.image(processed_img, use_column_width=True, caption=f"ภาพด้านหน้า ({front_source})")
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
            st.image(processed_img, use_column_width=True, caption=f"ภาพด้านหลัง ({back_source})")
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
        <div class="panel" style="text-align:center;">
          <h4 style="margin:.25rem 0 0;">พร้อมวิเคราะห์แล้ว</h4>
          <p class="muted" style="margin:.25rem 0 .5rem;">กดปุ่มด้านล่างเพื่อเริ่มการวิเคราะห์ด้วย AI</p>
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
        with st.spinner("🔍 กำลังประมวลผลด้วย AI Enhanced Mock Data... โปรดรอสักครู่"):
            try:
                r = send_predict_request(files, API_URL, timeout=60)
                if r.ok:
                    data = r.json()

                    # ---- Enhanced Primary Result Display ----
                    st.markdown("---")
                    st.success("✅ วิเคราะห์เสร็จสิ้น!")
                    
                    # AI Mode indicator
                    ai_mode = data.get("ai_mode", "mock_data")
                    processing_time = data.get("processing_time", 0)
                    
                    col_header1, col_header2 = st.columns([3, 1])
                    with col_header1:
                        st.markdown("## 🎯 ผลการวิเคราะห์หลัก")
                    with col_header2:
                        st.info(f"🤖 โหมด: {ai_mode}")
                        st.info(f"⏱️ เวลา: {processing_time:.2f}s")
                    
                    top1 = data.get("top1", {})
                    conf_pct = float(top1.get("confidence", 0.0)) * 100.0
                    class_name = top1.get("class_name", "Unknown")
                    
                    # Enhanced confidence display
                    confidence_color = "🟢" if conf_pct > 80 else "🟡" if conf_pct > 60 else "🔴"
                    st.markdown(
                        f"### {confidence_color} **{class_name}**"
                    )
                    st.markdown(f"**ความน่าจะเป็น:** {conf_pct:.1f}%")
                    
                    # Progress bar for confidence
                    st.progress(conf_pct/100, text=f"ความมั่นใจ: {conf_pct:.1f}%")

                    # ---- Enhanced Top-K Table ----
                    st.markdown("### 📊 รายงานความน่าจะเป็นทั้งหมด")
                    topk_data = []
                    for i, item in enumerate(data.get("topk", [])[:3], 1):
                        p = float(item.get("confidence", 0.0)) * 100.0
                        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                        topk_data.append({
                            "อันดับ": f"{emoji} #{i}",
                            "พระเครื่อง": item.get('class_name','—'),
                            "ความน่าจะเป็น": f"{p:.1f}%",
                            "คะแนน": f"{item.get('confidence', 0):.3f}"
                        })
                    
                    if topk_data:
                        st.table(topk_data)

                    # ---- Enhanced Valuation Display ----
                    st.markdown("### 💰 ประเมินราคาตลาด")
                    v = data.get("valuation", {})
                    if v:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            low_price = v.get('p05', 0)
                            st.metric("💵 ราคาต่ำสุด", f"฿{low_price:,}" if low_price else "–")
                        with col2:
                            mid_price = v.get('p50', 0)
                            st.metric("💸 ราคาเฉลี่ย", f"฿{mid_price:,}" if mid_price else "–")
                        with col3:
                            high_price = v.get('p95', 0)
                            st.metric("💎 ราคาสูงสุด", f"฿{high_price:,}" if high_price else "–")
                        
                        # Confidence indicator
                        val_confidence = v.get('confidence', 'medium')
                        confidence_emoji = "🎯" if val_confidence == "high" else "⚡" if val_confidence == "medium" else "⚠️"
                        st.info(f"{confidence_emoji} ความเชื่อมั่นในการประเมิน: **{val_confidence.upper()}**")

                    # ---- Enhanced Recommendations ----
                    st.markdown("### 🏪 แนะนำตลาดและช่องทางการขาย")
                    recs = data.get("recommendations", [])
                    if recs:
                        for i, rec in enumerate(recs):
                            market_name = rec.get("market", "Market")
                            rating = rec.get("rating", 0)
                            distance = rec.get("distance", 0)
                            
                            # Market type emoji
                            market_emoji = "🌐" if distance == 0 else "🏪"
                            rating_stars = "⭐" * int(rating) + "☆" * (5-int(rating))
                            
                            with st.expander(f"{market_emoji} {market_name} {rating_stars} ({rating}/5.0)", expanded=(i==0)):
                                st.write(f"**📝 เหตุผล:** {rec.get('reason','')}")
                                if distance > 0:
                                    st.write(f"**📍 ระยะทาง:** {distance} กิโลเมตร")
                                else:
                                    st.write(f"**💻 ประเภท:** ออนไลน์")
                                    
                                # Add recommendation score
                                st.progress(rating/5.0, text=f"คะแนนแนะนำ: {rating}/5.0")
                    else:
                        st.warning("⚠️ ยังไม่มีคำแนะนำตลาดในขณะนี้")
                        
                    # Timestamp info
                    timestamp = data.get("timestamp", "")
                    if timestamp:
                        st.caption(f"🕒 วิเคราะห์เมื่อ: {timestamp}")
                        
                else:
                    st.error(f"❌ เกิดข้อผิดพลาดจาก API: {r.status_code}")
                    st.write(f"📄 รายละเอียด: {r.text}")
                    
            except requests.exceptions.Timeout:
                st.warning("การประมวลผลใช้เวลานานเกินไป ลองใหม่อีกครั้งหรือลดขนาดไฟล์")
            except requests.exceptions.ConnectionError:
                st.error("เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ กรุณาตรวจสอบว่า Backend ทำงานอยู่ที่พอร์ต 8000")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดไม่คาดคิด: {e}")
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
**API Endpoint:** `{API_URL}`  
**Framework:** Streamlit + FastAPI  
**Last updated:** {datetime.now():%Y-%m-%d %H:%M}  
"""
    )
