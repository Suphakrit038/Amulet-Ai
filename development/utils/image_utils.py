"""
Image Processing Utilities
ฟังก์ชันช่วยสำหรับการประมวลผลภาพ
"""
import io
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def validate_image(file_data):
    """ตรวจสอบความถูกต้องของไฟล์ภาพ"""
    try:
        if hasattr(file_data, 'seek'):
            file_data.seek(0)
        
        img = Image.open(file_data)
        
        # ตรวจสอบรูปแบบที่รองรับ
        if img.format not in ['JPEG', 'PNG', 'HEIC', 'WEBP', 'BMP', 'TIFF']:
            return False, f"Unsupported format: {img.format}"
        
        # ตรวจสอบขนาดไฟล์
        if hasattr(file_data, 'seek'):
            file_data.seek(0, 2)  # ไปท้ายไฟล์
            size = file_data.tell()
            file_data.seek(0)  # กลับไปจุดเริ่มต้น
            
            if size > 10 * 1024 * 1024:  # 10MB
                return False, "File size too large (max 10MB)"
        
        return True, None
    except Exception as e:
        return False, str(e)

def preprocess_image(image, target_size=(224, 224)):
    """ปรับปรุงภาพสำหรับการวิเคราะห์"""
    # แปลงเป็น RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # ปรับขนาด
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # เพิ่มความคมชัด
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    
    # ปรับความสว่าง
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    
    return image

def image_to_bytes(image, format='JPEG', quality=95):
    """แปลงภาพเป็น bytes"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format, quality=quality)
    img_byte_arr.seek(0)
    return img_byte_arr
