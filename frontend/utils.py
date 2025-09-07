"""
โมดูล Utils (ความเข้ากันได้แบบเดิม + ฟังก์ชันใหม่)
ไฟล์นี้ส่งออกฟังก์ชันจาก amulet_utils.py เพื่อความเข้ากันได้แบบเดิม
และเพิ่มฟังก์ชันใหม่แบบโมดูลาร์เพื่อการจัดระเบียบโค้ดที่ดีขึ้น
"""

import logging
from PIL import Image
import io
import requests
from datetime import datetime
try:
    from .amulet_unified import (
        validate_and_convert_image,
        send_predict_request as legacy_send_predict_request,
        SUPPORTED_FORMATS,
        FORMAT_DISPLAY,
        MAX_FILE_SIZE_MB,
        find_reference_images,
        load_reference_images_for_comparison,
        compare_with_database,
        get_unified_result,
        format_comparison_results,
        get_dataset_info,
        process_image_with_api,
        check_api_connection,
        get_default_result
    )
except ImportError:
    # Fallback for direct import
    import amulet_unified
    validate_and_convert_image = amulet_unified.validate_and_convert_image
    legacy_send_predict_request = amulet_unified.send_predict_request
    SUPPORTED_FORMATS = amulet_unified.SUPPORTED_FORMATS
    FORMAT_DISPLAY = amulet_unified.FORMAT_DISPLAY
    MAX_FILE_SIZE_MB = amulet_unified.MAX_FILE_SIZE_MB
    find_reference_images = amulet_unified.find_reference_images
    load_reference_images_for_comparison = amulet_unified.load_reference_images_for_comparison
    compare_with_database = amulet_unified.compare_with_database
    get_unified_result = amulet_unified.get_unified_result
    format_comparison_results = amulet_unified.format_comparison_results
    get_dataset_info = amulet_unified.get_dataset_info
    process_image_with_api = amulet_unified.process_image_with_api
    check_api_connection = amulet_unified.check_api_connection
    get_default_result = amulet_unified.get_default_result

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_MB = 10
SUPPORTED = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]


def validate_bytes(b: bytes) -> tuple[bool, str]:
    """ตรวจสอบไฟล์ภาพว่าไม่เกินขนาดและเป็นไฟล์ที่เปิดได้"""
    if not b:
        return False, "ไฟล์ว่าง"
    if len(b) > MAX_MB * 1024 * 1024:
        return False, f"ไฟล์ใหญ่เกิน {MAX_MB} MB"
    try:
        Image.open(io.BytesIO(b)).verify()
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False, f"ไฟล์ภาพไม่ถูกต้อง: {e}"
    return True, None


def send_predict_request(files: dict, api_url: str, timeout: int = 60) -> requests.Response | None:
    """ส่งคำขอวิเคราะห์ไปยัง API backend"""
    url = api_url.rstrip("/") + "/predict"
    prepared = {}
    for k, v in files.items():
        fname, fileobj, mime = v
        try:
            fileobj.seek(0)
        except Exception:
            pass
        prepared[k] = (fname, fileobj, mime)
    try:
        response = requests.post(url, files=prepared, timeout=timeout)
        logger.info(f"API request to {url} returned status {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return None


def handle_upload(label: str, key: str) -> tuple[Image.Image | None, io.BytesIO | None, str | None, str | None]:
    """จัดการการอัปโหลดและตรวจสอบภาพ"""
    import streamlit as st  # Import here to avoid circular imports
    uploaded = st.file_uploader(label, type=SUPPORTED, key=key)
    if uploaded:
        bytes_data = uploaded.read()
        ok, msg = validate_bytes(bytes_data)
        uploaded.seek(0)
        if ok:
            try:
                img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
                processed = io.BytesIO()
                img.save(processed, format="JPEG", quality=90)
                processed.seek(0)
                filename = getattr(uploaded, "name", f"{key}_{datetime.now():%Y%m%d_%H%M%S}.jpg")
                return img, processed, filename, None
            except Exception as e:
                logger.error(f"Image processing failed: {e}")
                return None, None, None, f"ไม่สามารถอ่านภาพ: {e}"
        else:
            return None, None, None, msg
    return None, None, None, None
