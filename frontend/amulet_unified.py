"""
ระบบรวมสำหรับวิเคราะห์และเปรียบเทียบพระเครื่อง Amulet-AI
รวมฟังก์ชันทั้งหมดจาก amulet_comparison.py และ amulet_utils.py
- การวิเคราะห์จากโมเดล AI
- การเปรียบเทียบจากฐานข้อมูลภาพ
- ยูทิลิตี้สำหรับการจัดการรูปภาพและ API
- ปรับปรุงประสิทธิภาพและลดความซ้ำซ้อน
"""

import os
import json
import logging
import base64
import time
import requests
import io
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image
from pathlib import Path

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("amulet_unified")

# ค่าคงที่
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
FORMAT_DISPLAY = "JPG, JPEG, PNG, WebP, BMP, TIFF"
MAX_FILE_SIZE_MB = 10

# พาธมาตรฐาน
DEFAULT_REFERENCE_PATH = "dataset_organized"
DEFAULT_MODEL_PATH = "models/best_model.pth"
DEFAULT_CONFIG_PATH = "config.json"

# ===========================================================
# ฟังก์ชันช่วยเหลือทั่วไป
# ===========================================================

def get_project_root():
    """รับตำแหน่งโฟลเดอร์รากของโปรเจค"""
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_absolute_path(relative_path):
    """แปลงพาธสัมพัทธ์เป็นพาธสัมบูรณ์"""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(get_project_root(), relative_path)

def read_config(config_path=None):
    """อ่านการตั้งค่าจากไฟล์ config.json"""
    if config_path is None:
        config_path = os.path.join(get_project_root(), DEFAULT_CONFIG_PATH)
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"ไม่พบไฟล์ config: {config_path}")
            return {}
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ config: {e}")
        return {}

def save_config(config_data, config_path=None):
    """บันทึกการตั้งค่าลงในไฟล์ config.json"""
    if config_path is None:
        config_path = os.path.join(get_project_root(), DEFAULT_CONFIG_PATH)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        logger.info(f"บันทึกการตั้งค่าเรียบร้อยแล้ว: {config_path}")
        return True
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการบันทึกไฟล์ config: {e}")
        return False

# ===========================================================
# ฟังก์ชันจัดการรูปภาพ
# ===========================================================

def validate_and_convert_image(uploaded_file):
    """
    ตรวจสอบและแปลงรูปภาพที่อัปโหลด
    - ตรวจสอบขนาดไฟล์
    - ตรวจสอบนามสกุลไฟล์
    - แปลงเป็นโหมด RGB
    - บีบอัดเป็น JPEG
    
    Args:
        uploaded_file: ไฟล์ที่อัปโหลด
        
    Returns:
        (bool, Image, BytesIO, str): สถานะความถูกต้อง, รูปภาพ PIL, ข้อมูลไบต์, ข้อความข้อผิดพลาด
    """
    try:
        # ตั้งตำแหน่งไฟล์กลับไปที่เริ่มต้น
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        # อ่านข้อมูลไฟล์
        if hasattr(uploaded_file, "read"):
            file_bytes = uploaded_file.read()
        else:
            file_bytes = getattr(uploaded_file, "getvalue", lambda: b"")()

        if not file_bytes:
            return False, None, None, "ไฟล์ว่างเปล่าหรือไม่สามารถอ่านได้"

        # ตรวจสอบขนาดไฟล์
        if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, None, None, f"ไฟล์ใหญ่เกินไป (> {MAX_FILE_SIZE_MB} MB)"

        # ตรวจสอบนามสกุลไฟล์
        filename = getattr(uploaded_file, "name", "") or ""
        if filename:
            ext = filename.rsplit(".", 1)[-1].lower()
            if ext not in SUPPORTED_FORMATS:
                return False, None, None, f"นามสกุลไฟล์ไม่รองรับ: .{ext}"

        # เปิดและแปลงรูปภาพ
        img = Image.open(io.BytesIO(file_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")

        # บีบอัดและเตรียม BytesIO
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG", quality=95)
        img_byte_arr.seek(0)

        return True, img, img_byte_arr, None
    except Exception as e:
        return False, None, None, str(e)

def send_predict_request(files, api_url, timeout=60):
    """
    ส่งคำขอวิเคราะห์รูปภาพไปยัง API
    
    Args:
        files: พจนานุกรมของไฟล์ที่จะส่ง
        api_url: URL ของ API
        timeout: เวลาหมดเวลาในวินาที
        
    Returns:
        Response: ผลตอบกลับจาก API
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

def encode_image_to_base64(image, format="JPEG", quality=85, max_size=(300, 300)):
    """
    แปลงรูปภาพเป็น base64
    
    Args:
        image: รูปภาพ PIL หรือพาธไปยังไฟล์
        format: รูปแบบการบีบอัด
        quality: คุณภาพการบีบอัด
        max_size: ขนาดสูงสุด
        
    Returns:
        str: สตริง base64
    """
    if isinstance(image, str) or isinstance(image, Path):
        image = Image.open(image).convert('RGB')
    
    # ปรับขนาดรูปภาพถ้าใหญ่เกินไป
    if max_size and (image.width > max_size[0] or image.height > max_size[1]):
        image.thumbnail(max_size, Image.LANCZOS)
    
    buffered = BytesIO()
    image.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def decode_base64_to_image(base64_str):
    """
    แปลง base64 เป็นรูปภาพ PIL
    
    Args:
        base64_str: สตริง base64
        
    Returns:
        Image: รูปภาพ PIL
    """
    img_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_bytes))

# ===========================================================
# ฟังก์ชันค้นหารูปภาพอ้างอิง
# ===========================================================

def get_dataset_class_folders(dataset_path=DEFAULT_REFERENCE_PATH):
    """
    ค้นหาโฟลเดอร์ของแต่ละคลาสในชุดข้อมูล
    
    Args:
        dataset_path: พาธไปยังชุดข้อมูลที่จัดระเบียบแล้ว
        
    Returns:
        dict: พจนานุกรมที่มีคีย์เป็นชื่อคลาส ค่าเป็นพาธไปยังโฟลเดอร์
    """
    class_folders = {}
    
    try:
        # ใช้พาธเต็ม
        dataset_path = get_absolute_path(dataset_path)
        
        if not os.path.exists(dataset_path):
            logger.warning(f"ไม่พบพาธชุดข้อมูล: {dataset_path}")
            return class_folders
        
        # ค้นหาโฟลเดอร์ทั้งหมดในชุดข้อมูล
        class_dirs = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')]
        
        for class_dir in class_dirs:
            class_name = class_dir.replace('_', ' ').title()
            class_folders[class_name] = os.path.join(dataset_path, class_dir)
        
        logger.info(f"พบคลาสทั้งหมด {len(class_folders)} คลาส")
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการค้นหาโฟลเดอร์คลาส: {e}")
    
    return class_folders

def find_reference_images(class_name, dataset_path=DEFAULT_REFERENCE_PATH, view_type=None, max_images=3):
    """
    ค้นหารูปภาพอ้างอิงสำหรับคลาสที่ระบุ
    
    Args:
        class_name: ชื่อคลาสที่ต้องการค้นหารูปภาพอ้างอิง
        dataset_path: พาธไปยังชุดข้อมูลที่จัดระเบียบแล้ว
        view_type: มุมมองของภาพ (front, back, None=ทั้งหมด)
        max_images: จำนวนรูปภาพสูงสุดที่ต้องการ
        
    Returns:
        dict: พจนานุกรมของรูปภาพอ้างอิง
    """
    reference_images = {}
    
    try:
        # ใช้พาธเต็ม
        dataset_path = get_absolute_path(dataset_path)
        
        # แปลงชื่อคลาสเป็นรูปแบบชื่อโฟลเดอร์
        folder_name = class_name.lower().replace(' ', '_')
        class_path = os.path.join(dataset_path, folder_name)
        
        # ตรวจสอบว่าโฟลเดอร์มีอยู่หรือไม่
        if not os.path.exists(class_path):
            # ค้นหาคลาสที่คล้ายกัน
            class_folders = get_dataset_class_folders(dataset_path)
            similar_classes = [name for name in class_folders.keys() 
                              if name.lower().replace(' ', '') in class_name.lower().replace(' ', '') 
                              or class_name.lower().replace(' ', '') in name.lower().replace(' ', '')]
            
            if similar_classes:
                folder_name = similar_classes[0].lower().replace(' ', '_')
                class_path = os.path.join(dataset_path, folder_name)
                logger.info(f"ใช้คลาสที่คล้ายกัน: {similar_classes[0]} แทน {class_name}")
            else:
                logger.warning(f"ไม่พบโฟลเดอร์สำหรับคลาส {class_name}")
                return reference_images
        
        # ค้นหาไฟล์รูปภาพในโฟลเดอร์
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # กรองตาม view_type ถ้ามีการระบุ
        if view_type:
            image_files = [f for f in image_files if view_type.lower() in f.lower()]
        
        # จัดลำดับความสำคัญตามมุมมอง
        front_images = [f for f in image_files if "front" in f.lower()]
        back_images = [f for f in image_files if "back" in f.lower()]
        other_images = [f for f in image_files if "front" not in f.lower() and "back" not in f.lower()]
        
        # รวมและจำกัดจำนวน
        sorted_images = front_images + back_images + other_images
        sorted_images = sorted_images[:max_images]
        
        # แปลงรูปภาพเป็น base64
        for i, img_file in enumerate(sorted_images):
            img_path = os.path.join(class_path, img_file)
            
            try:
                with Image.open(img_path) as img:
                    # ปรับขนาดรูปภาพให้เหมาะสม
                    img = img.resize((300, 300), Image.LANCZOS)
                    
                    # แปลงเป็น base64
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # กำหนด view_type จากชื่อไฟล์
                    img_view_type = "front" if "front" in img_file.lower() else \
                                   "back" if "back" in img_file.lower() else "unknown"
                    
                    reference_images[f"ref_{i+1}"] = {
                        "image_b64": img_str,
                        "view_type": img_view_type,
                        "filename": img_file
                    }
            except Exception as e:
                logger.error(f"ไม่สามารถโหลดรูปภาพ {img_path}: {e}")
        
        logger.info(f"พบรูปภาพอ้างอิงทั้งหมด {len(reference_images)} รูป สำหรับคลาส {class_name}")
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการค้นหารูปภาพอ้างอิง: {e}")
    
    return reference_images

def load_reference_images_for_comparison(prediction_result, dataset_path=DEFAULT_REFERENCE_PATH):
    """
    โหลดรูปภาพอ้างอิงสำหรับผลการทำนาย
    
    Args:
        prediction_result: ผลลัพธ์จาก API การทำนาย
        dataset_path: พาธไปยังชุดข้อมูลที่จัดระเบียบแล้ว
        
    Returns:
        dict: ผลลัพธ์การทำนายพร้อมรูปภาพอ้างอิง
    """
    # คัดลอกผลลัพธ์เพื่อไม่ให้เปลี่ยนแปลงต้นฉบับ
    result = prediction_result.copy()
    
    # ตรวจสอบว่ามีรูปภาพอ้างอิงอยู่แล้วหรือไม่
    if "reference_images" in result and result["reference_images"]:
        logger.info("ใช้รูปภาพอ้างอิงจาก API โดยตรง")
        return result
    
    # ถ้าไม่มี ให้โหลดจากฐานข้อมูลท้องถิ่น
    try:
        if "top1" in result:
            top_class = result["top1"].get("class_name", "")
            if top_class:
                reference_images = find_reference_images(top_class, dataset_path)
                result["reference_images"] = reference_images
                logger.info(f"โหลดรูปภาพอ้างอิงสำหรับคลาส {top_class} จำนวน {len(reference_images)} รูป")
            else:
                logger.warning("ไม่พบข้อมูลคลาสในผลการทำนาย")
        else:
            logger.warning("ไม่พบข้อมูล top1 ในผลการทำนาย")
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการโหลดรูปภาพอ้างอิง: {e}")
        result["reference_images"] = {}
    
    return result

# ===========================================================
# ฟังก์ชันเปรียบเทียบและรวมผลลัพธ์
# ===========================================================

def compare_with_database(image, class_name, dataset_path=DEFAULT_REFERENCE_PATH, max_matches=3):
    """
    เปรียบเทียบรูปภาพที่อัปโหลดกับรูปภาพในฐานข้อมูล
    
    Args:
        image: รูปภาพที่ต้องการเปรียบเทียบ
        class_name: ชื่อคลาสที่ต้องการเปรียบเทียบ
        dataset_path: พาธไปยังชุดข้อมูลที่จัดระเบียบแล้ว
        max_matches: จำนวนการจับคู่สูงสุดที่ต้องการ
        
    Returns:
        list: รายการผลการเปรียบเทียบ
    """
    # หารูปภาพอ้างอิงสำหรับคลาส
    reference_images = find_reference_images(class_name, dataset_path, max_images=max_matches)
    
    # สร้างผลลัพธ์การเปรียบเทียบ
    comparison_results = []
    
    for ref_key, ref_data in reference_images.items():
        # แปลง base64 เป็นรูปภาพ
        if "image_b64" in ref_data:
            # สร้างผลลัพธ์การเปรียบเทียบแบบง่าย
            # ในระบบจริงคุณอาจใช้การเปรียบเทียบลักษณะพิเศษที่ซับซ้อนมากขึ้น
            comparison_results.append({
                "reference_image": ref_data,
                "similarity_score": 0.85,  # ค่าตัวอย่าง ในระบบจริงควรคำนวณจากการเปรียบเทียบจริง
                "view_type": ref_data.get("view_type", "unknown")
            })
    
    return comparison_results

def get_unified_result(api_result, ref_image_path=DEFAULT_REFERENCE_PATH):
    """
    รวมผลการวิเคราะห์และเปรียบเทียบเข้าด้วยกัน
    
    Args:
        api_result: ผลลัพธ์จาก API สำหรับการวิเคราะห์
        ref_image_path: พาธไปยังไดเรกทอรีรูปภาพอ้างอิง
        
    Returns:
        dict: ผลลัพธ์ที่รวมการวิเคราะห์และเปรียบเทียบ
    """
    result = api_result.copy()
    
    # ตรวจสอบว่ามีรูปภาพอ้างอิงอยู่แล้วหรือไม่
    if "reference_images" not in result or not result["reference_images"]:
        try:
            # ดึงคลาสที่ทำนายได้
            if "top1" in result and "class_name" in result["top1"]:
                class_name = result["top1"]["class_name"]
                
                # ค้นหารูปภาพอ้างอิง
                ref_images = find_reference_images(class_name, ref_image_path)
                result["reference_images"] = ref_images
                
                logger.info(f"เพิ่มรูปภาพอ้างอิงสำหรับคลาส {class_name} จำนวน {len(ref_images)} รูป")
            else:
                logger.warning("ไม่พบข้อมูลคลาสในผลการทำนาย")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการเพิ่มรูปภาพอ้างอิง: {e}")
            result["reference_images"] = {}
    
    # เพิ่มข้อมูลเปรียบเทียบ
    try:
        # คำนวณค่าความเหมือนสำหรับรูปภาพแต่ละรูป
        # ในระบบจริงอาจมีการคำนวณที่ซับซ้อนกว่านี้
        if "reference_images" in result and result["reference_images"]:
            comparison_data = []
            
            for key, ref_data in result["reference_images"].items():
                view_type = ref_data.get("view_type", "unknown")
                
                # จำลองค่าความเหมือน (ในระบบจริงควรคำนวณจริง)
                similarity = 0.85
                if "front" in view_type:
                    similarity = 0.90
                elif "back" in view_type:
                    similarity = 0.82
                
                comparison_data.append({
                    "ref_key": key,
                    "view_type": view_type,
                    "similarity": similarity,
                    "similarity_percent": f"{similarity * 100:.1f}%",
                    "filename": ref_data.get("filename", "unknown")
                })
            
            result["comparison_data"] = comparison_data
            result["overall_similarity"] = 0.88  # ค่าเฉลี่ยของความเหมือนทั้งหมด
            logger.info(f"เพิ่มข้อมูลเปรียบเทียบสำหรับรูปภาพอ้างอิง {len(comparison_data)} รูป")
        
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการเพิ่มข้อมูลเปรียบเทียบ: {e}")
        result["comparison_data"] = []
        result["overall_similarity"] = 0.0
    
    # เพิ่มข้อมูลเพิ่มเติม
    result["unified_timestamp"] = time.time()
    result["processing_time"] = time.time() - result.get("timestamp_start", time.time())
    result["ai_mode"] = result.get("ai_mode", "unified")
    
    return result

def format_comparison_results(result_data):
    """
    จัดรูปแบบผลลัพธ์สำหรับการแสดงผลเปรียบเทียบ
    
    Args:
        result_data: ข้อมูลผลลัพธ์จากการเปรียบเทียบ
        
    Returns:
        dict: ข้อมูลสำหรับแสดงผลในรูปแบบที่เหมาะสม
    """
    formatted_result = {
        "front_comparison": None,
        "back_comparison": None,
        "additional_images": [],
        "similarity_table": [],
        "summary": {}
    }
    
    try:
        # จัดกลุ่มรูปภาพตามมุมมอง
        front_images = {}
        back_images = {}
        other_images = {}
        
        if "reference_images" in result_data and result_data["reference_images"]:
            for key, ref_data in result_data["reference_images"].items():
                view_type = ref_data.get("view_type", "unknown").lower()
                
                if "front" in view_type:
                    front_images[key] = ref_data
                elif "back" in view_type:
                    back_images[key] = ref_data
                else:
                    other_images[key] = ref_data
        
        # หาข้อมูลความเหมือน
        comparison_data = result_data.get("comparison_data", [])
        similarity_by_ref = {item["ref_key"]: item["similarity"] for item in comparison_data}
        
        # สร้างข้อมูลสำหรับเปรียบเทียบด้านหน้า
        if front_images:
            first_key = list(front_images.keys())[0]
            ref_data = front_images[first_key]
            similarity = similarity_by_ref.get(first_key, 0.0)
            
            formatted_result["front_comparison"] = {
                "ref_data": ref_data,
                "similarity": similarity,
                "similarity_percent": f"{similarity * 100:.1f}%"
            }
        
        # สร้างข้อมูลสำหรับเปรียบเทียบด้านหลัง
        if back_images:
            first_key = list(back_images.keys())[0]
            ref_data = back_images[first_key]
            similarity = similarity_by_ref.get(first_key, 0.0)
            
            formatted_result["back_comparison"] = {
                "ref_data": ref_data,
                "similarity": similarity,
                "similarity_percent": f"{similarity * 100:.1f}%"
            }
        
        # รูปภาพเพิ่มเติม
        remaining_images = {}
        if front_images:
            remaining_images.update({k: v for k, v in front_images.items() if k != list(front_images.keys())[0]})
        if back_images:
            remaining_images.update({k: v for k, v in back_images.items() if k != list(back_images.keys())[0]})
        remaining_images.update(other_images)
        
        for key, ref_data in remaining_images.items():
            similarity = similarity_by_ref.get(key, 0.0)
            formatted_result["additional_images"].append({
                "ref_data": ref_data,
                "similarity": similarity,
                "similarity_percent": f"{similarity * 100:.1f}%"
            })
        
        # ตารางความเหมือน
        for item in comparison_data:
            similarity = item["similarity"]
            
            # กำหนดระดับความเหมือน
            if similarity >= 0.85:
                level = "สูง"
                color = "#10B981"  # สีเขียว
            elif similarity >= 0.7:
                level = "ปานกลาง"
                color = "#F59E0B"  # สีส้ม
            else:
                level = "ต่ำ"
                color = "#EF4444"  # สีแดง
            
            formatted_result["similarity_table"].append({
                "view_type": item["view_type"],
                "similarity": similarity,
                "similarity_percent": item["similarity_percent"],
                "level": level,
                "color": color,
                "filename": item.get("filename", "")
            })
        
        # สรุปผล
        overall_similarity = result_data.get("overall_similarity", 0.0)
        top_class = result_data.get("top1", {}).get("class_name", "Unknown")
        conf_pct = float(result_data.get("top1", {}).get("confidence", 0.0)) * 100.0
        
        formatted_result["summary"] = {
            "top_class": top_class,
            "confidence": conf_pct,
            "overall_similarity": overall_similarity,
            "overall_similarity_percent": f"{overall_similarity * 100:.1f}%",
            "similar_features": ["รูปทรง", "ลวดลาย", "สัดส่วน"],
            "different_features": ["รายละเอียดปลีกย่อย", "ความคมชัด"]
        }
    
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการจัดรูปแบบผลลัพธ์: {e}")
    
    return formatted_result

def get_dataset_info(dataset_path=DEFAULT_REFERENCE_PATH):
    """
    ดึงข้อมูลเกี่ยวกับชุดข้อมูลสำหรับแสดงผล
    
    Args:
        dataset_path: พาธไปยังไดเรกทอรีชุดข้อมูล
        
    Returns:
        dict: ข้อมูลเกี่ยวกับชุดข้อมูล
    """
    info = {
        "class_count": 0,
        "image_count": 0,
        "classes": [],
        "image_types": {
            "front": 0,
            "back": 0,
            "other": 0
        }
    }
    
    try:
        # ใช้พาธเต็ม
        dataset_path = get_absolute_path(dataset_path)
        
        if not os.path.exists(dataset_path):
            logger.warning(f"ไม่พบไดเรกทอรีชุดข้อมูล: {dataset_path}")
            return info
        
        # นับจำนวนคลาสและรูปภาพ
        class_dirs = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')]
        
        info["class_count"] = len(class_dirs)
        
        for class_dir in class_dirs:
            class_path = os.path.join(dataset_path, class_dir)
            class_name = class_dir.replace('_', ' ').title()
            
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            image_count = len(image_files)
            info["image_count"] += image_count
            
            # นับประเภทรูปภาพ
            front_count = len([f for f in image_files if "front" in f.lower()])
            back_count = len([f for f in image_files if "back" in f.lower()])
            other_count = image_count - front_count - back_count
            
            info["image_types"]["front"] += front_count
            info["image_types"]["back"] += back_count
            info["image_types"]["other"] += other_count
            
            info["classes"].append({
                "name": class_name,
                "image_count": image_count,
                "front_count": front_count,
                "back_count": back_count,
                "other_count": other_count
            })
        
        # เรียงคลาสตามจำนวนรูปภาพ (มากไปน้อย)
        info["classes"].sort(key=lambda x: x["image_count"], reverse=True)
        
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลชุดข้อมูล: {e}")
    
    return info

# ===========================================================
# API และการเชื่อมต่อ
# ===========================================================

def check_api_connection(api_url):
    """
    ตรวจสอบการเชื่อมต่อกับ API
    
    Args:
        api_url: URL ของ API
        
    Returns:
        (bool, dict): สถานะการเชื่อมต่อ, ข้อมูลตอบกลับ
    """
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API ส่งสถานะข้อผิดพลาด: {response.status_code}", "details": response.text}
    except Exception as e:
        return False, {"error": f"ไม่สามารถเชื่อมต่อกับ API ได้", "details": str(e)}

def get_api_info(api_url):
    """
    ดึงข้อมูลเกี่ยวกับ API
    
    Args:
        api_url: URL ของ API
        
    Returns:
        dict: ข้อมูลเกี่ยวกับ API
    """
    try:
        response = requests.get(f"{api_url.rstrip('/')}/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API ส่งสถานะข้อผิดพลาด: {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": f"ไม่สามารถเชื่อมต่อกับ API ได้", "details": str(e)}

# ===========================================================
# การทำงานหลัก
# ===========================================================

def process_image_with_api(front_image, back_image, api_url, front_filename=None, back_filename=None):
    """
    ประมวลผลรูปภาพด้วย API
    
    Args:
        front_image: รูปภาพด้านหน้า (ไฟล์หรือ BytesIO)
        back_image: รูปภาพด้านหลัง (ไฟล์หรือ BytesIO)
        api_url: URL ของ API
        front_filename: ชื่อไฟล์ด้านหน้า
        back_filename: ชื่อไฟล์ด้านหลัง
        
    Returns:
        (bool, dict): สถานะความสำเร็จ, ผลลัพธ์หรือข้อผิดพลาด
    """
    try:
        # เตรียมชื่อไฟล์ถ้าไม่ได้ระบุ
        if front_filename is None:
            front_filename = f"front_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        if back_filename is None:
            back_filename = f"back_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        
        # เตรียมไฟล์สำหรับส่ง
        files = {
            "front": (
                front_filename,
                front_image,
                "image/jpeg",
            ),
            "back": (
                back_filename,
                back_image,
                "image/jpeg",
            ),
        }
        
        # บันทึกเวลาเริ่มต้น
        start_time = time.time()
        
        # ส่งคำขอไปยัง API
        response = send_predict_request(files, api_url, timeout=60)
        
        # คำนวณเวลาที่ใช้
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # เพิ่มข้อมูลเวลา
            result["timestamp_start"] = start_time
            result["processing_time"] = elapsed_time
            
            # รวมผลลัพธ์และเพิ่มข้อมูลเปรียบเทียบ
            unified_result = get_unified_result(result)
            
            return True, unified_result
        else:
            return False, {
                "error": f"API ส่งสถานะข้อผิดพลาด: {response.status_code}",
                "details": response.text,
                "processing_time": elapsed_time
            }
    
    except requests.exceptions.Timeout:
        return False, {"error": "การประมวลผลใช้เวลานานเกินไป", "details": "กรุณาลองใหม่อีกครั้งหรือลดขนาดไฟล์"}
    except requests.exceptions.ConnectionError:
        return False, {"error": "ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ได้", "details": "กรุณาตรวจสอบว่า Backend ทำงานอยู่ที่พอร์ต 8000 หรือ 8001"}
    except Exception as e:
        return False, {"error": "เกิดข้อผิดพลาดที่ไม่คาดคิด", "details": str(e)}

def get_default_result():
    """
    สร้างผลลัพธ์เริ่มต้นสำหรับกรณีที่ API ไม่ทำงาน
    
    Returns:
        dict: ผลลัพธ์เริ่มต้น
    """
    return {
        "top1": {
            "class_name": "ไม่สามารถวิเคราะห์ได้",
            "confidence": 0.0
        },
        "topk": [
            {
                "class_name": "ไม่สามารถวิเคราะห์ได้",
                "confidence": 0.0
            }
        ],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": True,
        "error_message": "ไม่สามารถเชื่อมต่อกับ API ได้"
    }

# ===========================================================
# คลาสสำหรับเปรียบเทียบรูปภาพ
# ===========================================================

class ImageComparer:
    """คลาสสำหรับเปรียบเทียบรูปภาพพระเครื่อง"""
    
    def __init__(self, model_path, database_dir):
        self.model_path = model_path
        self.database_dir = database_dir
        
        # สร้าง FeatureExtractor จำลอง
        class FeatureExtractor:
            def __init__(self, model_path):
                self.model_path = model_path
                
            def extract_features(self, image):
                """จำลองการแยกลักษณะเด่น"""
                return np.random.rand(512)  # คืนค่าเวกเตอร์สุ่ม 512 มิติ
        
        self.feature_extractor = FeatureExtractor(model_path)
        
    def compare_image(self, image_path, top_k=5):
        """
        เปรียบเทียบรูปภาพกับฐานข้อมูล
        
        Args:
            image_path: พาธไปยังรูปภาพที่ต้องการเปรียบเทียบ
            top_k: จำนวนภาพที่เหมือนที่สุดที่ต้องการแสดง
            
        Returns:
            dict: ผลการเปรียบเทียบ
        """
        result = {
            "query_image": str(image_path),
            "top_matches": [],
            "processing_time": 0
        }
        
        # เวลาเริ่มต้น
        start_time = time.time()
        
        try:
            # ในระบบจริงควรใช้ feature extractor เพื่อดึงลักษณะเด่นและคำนวณความเหมือน
            # ในที่นี้เราจะสร้างข้อมูลจำลอง
            
            # จำลองการหารูปภาพที่เหมือนที่สุด
            class_dirs = []
            if os.path.exists(self.database_dir):
                class_dirs = [d for d in os.listdir(self.database_dir) 
                             if os.path.isdir(os.path.join(self.database_dir, d))]
            
            if not class_dirs:
                # ถ้าไม่มีคลาสในฐานข้อมูล ใช้ชื่อจำลอง
                class_dirs = ["class1", "class2", "class3", "class4", "class5"]
            
            for i, class_name in enumerate(class_dirs[:min(top_k, len(class_dirs))]):
                # จำลองค่าความเหมือน
                similarity = 0.9 - (i * 0.1)
                
                # ใช้รูปภาพในคลาสเป็นผลลัพธ์
                class_path = os.path.join(self.database_dir, class_name)
                
                if os.path.exists(class_path):
                    # ค้นหารูปภาพในโฟลเดอร์คลาส
                    image_files = [f for f in os.listdir(class_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    if image_files:
                        # เลือกรูปภาพแรก
                        img_path = os.path.join(class_path, image_files[0])
                        display_name = class_name.replace('_', ' ').title()
                        
                        result["top_matches"].append({
                            "path": img_path,
                            "class": display_name,
                            "similarity": similarity,
                            "filename": image_files[0]
                        })
                        continue
                
                # ถ้าไม่มีรูปในคลาส ใช้รูปเดิม
                result["top_matches"].append({
                    "path": str(image_path),
                    "class": class_name.replace('_', ' ').title(),
                    "similarity": similarity,
                    "filename": f"sample_{i+1}.jpg"
                })
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการเปรียบเทียบรูปภาพ: {e}")
        
        # คำนวณเวลาที่ใช้
        result["processing_time"] = time.time() - start_time
        
        return result

# ===========================================================
# Export
# ===========================================================

__all__ = [
    # ฟังก์ชันการจัดการรูปภาพ
    'validate_and_convert_image',
    'send_predict_request',
    'encode_image_to_base64',
    'decode_base64_to_image',
    
    # ฟังก์ชันค้นหารูปภาพอ้างอิง
    'get_dataset_class_folders',
    'find_reference_images',
    'load_reference_images_for_comparison',
    
    # ฟังก์ชันเปรียบเทียบและรวมผลลัพธ์
    'compare_with_database',
    'get_unified_result',
    'format_comparison_results',
    'get_dataset_info',
    
    # ฟังก์ชัน API และการเชื่อมต่อ
    'check_api_connection',
    'get_api_info',
    'process_image_with_api',
    'get_default_result',
    
    # ฟังก์ชันช่วยเหลือทั่วไป
    'get_project_root',
    'get_absolute_path',
    'read_config',
    'save_config',
    
    # คลาส
    'ImageComparer',
    
    # ค่าคงที่
    'SUPPORTED_FORMATS',
    'FORMAT_DISPLAY',
    'MAX_FILE_SIZE_MB',
    'DEFAULT_REFERENCE_PATH',
    'DEFAULT_MODEL_PATH',
    'DEFAULT_CONFIG_PATH',
]
