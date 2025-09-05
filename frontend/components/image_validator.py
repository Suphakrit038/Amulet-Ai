"""
คอมโพเนนต์สำหรับการตรวจสอบความถูกต้องของรูปภาพ
"""

import io
import logging
import sys
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional

# Add frontend to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import IMAGE_SETTINGS
except ImportError:
    # Fallback configuration
    IMAGE_SETTINGS = {
        "SUPPORTED_FORMATS": ["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        "MAX_FILE_SIZE_MB": 10,
        "IMAGE_QUALITY": 95,
        "THUMBNAIL_SIZE": (300, 300)
    }

logger = logging.getLogger(__name__)

class ImageValidator:
    """คลาสสำหรับตรวจสอบความถูกต้องของรูปภาพ"""
    
    def __init__(self):
        self.supported_formats = IMAGE_SETTINGS["SUPPORTED_FORMATS"]
        self.max_file_size = IMAGE_SETTINGS["MAX_FILE_SIZE_MB"] * 1024 * 1024
        self.image_quality = IMAGE_SETTINGS["IMAGE_QUALITY"]
    
    def validate_and_convert(self, uploaded_file) -> Tuple[bool, Optional[Image.Image], Optional[io.BytesIO], Optional[str]]:
        """
        ตรวจสอบและแปลงรูปภาพที่อัปโหลด
        
        Args:
            uploaded_file: ไฟล์ที่อัปโหลด
            
        Returns:
            (is_valid, pil_image, jpeg_bytesio, error_message)
        """
        try:
            # ตั้งตำแหน่งไฟล์กลับไปที่เริ่มต้น
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

            # อ่านข้อมูลไฟล์
            if hasattr(uploaded_file, 'read'):
                file_bytes = uploaded_file.read()
            else:
                file_bytes = getattr(uploaded_file, 'getvalue', lambda: b'')()

            if not file_bytes:
                return False, None, None, 'ไฟล์ว่างเปล่าหรือไม่สามารถอ่านได้'

            # ตรวจสอบขนาดไฟล์
            if len(file_bytes) > self.max_file_size:
                return False, None, None, f'ไฟล์ใหญ่เกินไป (> {IMAGE_SETTINGS["MAX_FILE_SIZE_MB"]} MB)'

            # ตรวจสอบนามสกุลไฟล์
            filename = getattr(uploaded_file, 'name', '') or ''
            if filename:
                ext = filename.rsplit('.', 1)[-1].lower()
                if ext not in self.supported_formats:
                    return False, None, None, f'นามสกุลไฟล์ไม่รองรับ: .{ext}'

            # เปิดและแปลงรูปภาพ
            img = Image.open(io.BytesIO(file_bytes))
            
            # ตรวจสอบความเสียหายของรูปภาพ
            try:
                img.verify()
                # โหลดรูปภาพใหม่หลังจาก verify
                img = Image.open(io.BytesIO(file_bytes))
            except Exception as e:
                return False, None, None, f'รูปภาพเสียหาย: {str(e)}'
            
            # แปลงเป็น RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # แปลงเป็น JPEG bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=self.image_quality)
            img_byte_arr.seek(0)

            logger.info(f"ตรวจสอบรูปภาพสำเร็จ: {filename}, ขนาด: {len(file_bytes)} bytes")
            return True, img, img_byte_arr, None
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตรวจสอบรูปภาพ: {str(e)}")
            return False, None, None, str(e)

    def validate_dimensions(self, image: Image.Image, min_width: int = 224, min_height: int = 224) -> Tuple[bool, str]:
        """
        ตรวจสอบขนาดของรูปภาพ
        
        Args:
            image: รูปภาพ PIL
            min_width: ความกว้างขั้นต่ำ
            min_height: ความสูงขั้นต่ำ
            
        Returns:
            (is_valid, message)
        """
        if image.width < min_width or image.height < min_height:
            return False, f'รูปภาพเล็กเกินไป (ขั้นต่ำ {min_width}x{min_height})'
        
        return True, 'ขนาดรูปภาพเหมาะสม'

# สร้าง instance สำหรับใช้งาน
image_validator = ImageValidator()

# Export ฟังก์ชันสำหรับใช้งานแบบเก่า
def validate_and_convert_image(uploaded_file):
    """ฟังก์ชันเพื่อความเข้ากันได้แบบเก่า"""
    return image_validator.validate_and_convert(uploaded_file)
