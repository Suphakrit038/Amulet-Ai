"""
การตั้งค่าระบบ Frontend แบบรวมศูนย์
"""

import os
from pathlib import Path

# ค่าพื้นฐาน
API_URL = os.environ.get("AMULET_API_URL", "http://127.0.0.1:8001")
DATABASE_DIR = os.environ.get("AMULET_DB_DIR", "data_base")
MODEL_PATH = os.environ.get("AMULET_MODEL_PATH", "frontend/models/best_model.pth")

# ค่าสำหรับการวิเคราะห์ภาพ
IMAGE_SETTINGS = {
    "SUPPORTED_FORMATS": ["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
    "FORMAT_DISPLAY": "JPG, JPEG, PNG, WebP, BMP, TIFF",
    "MAX_FILE_SIZE_MB": 10,
    "IMAGE_QUALITY": 95,
    "THUMBNAIL_SIZE": (300, 300)
}

# ค่าสำหรับ AI
AI_SETTINGS = {
    "TOP_K": 3,
    "CONFIDENCE_THRESHOLD": 0.7,
    "MIN_SIMILARITY_SCORE": 0.65,
    "BATCH_SIZE": 32,
    "NUM_WORKERS": 4
}

# ค่าสำหรับ UI
UI_SETTINGS = {
    "PRIMARY_COLOR": "#2563EB",
    "SECONDARY_COLOR": "#1E3A8A", 
    "SUCCESS_COLOR": "#10B981",
    "WARNING_COLOR": "#F59E0B",
    "ERROR_COLOR": "#EF4444",
    "BACKGROUND_COLOR": "#F9FAFB",
    "CARD_COLOR": "#FFFFFF"
}

# ค่าสำหรับการแคช
CACHE_SETTINGS = {
    "ENABLE_CACHING": True,
    "CACHE_TTL": 3600,  # 1 ชั่วโมง
    "MAX_CACHE_SIZE": 100,  # MB
    "IMAGE_CACHE_TTL": 7200,  # 2 hours
    "API_CACHE_TTL": 1800,  # 30 minutes
    "ENABLE_COMPRESSION": True,
    "COMPRESSION_QUALITY": 85
}

# ค่าสำหรับการล็อก
LOGGING_SETTINGS = {
    "LOG_LEVEL": "INFO",
    "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "LOG_FILE": "logs/amulet_ai.log",
    "MAX_LOG_SIZE": 10 * 1024 * 1024,  # 10MB
    "BACKUP_COUNT": 5,
    "ENABLE_CONSOLE": True,
    "ENABLE_FILE": True
}

# Performance Monitoring
PERFORMANCE_SETTINGS = {
    "ENABLE_METRICS": True,
    "METRICS_RETENTION_DAYS": 30,
    "ENABLE_PROFILING": False,  # Only for development
    "MAX_RESPONSE_TIME": 30,  # seconds
    "MAX_IMAGE_SIZE": 10 * 1024 * 1024,  # 10MB
    "ENABLE_ANALYTICS": True
}

# Production Settings
PRODUCTION_SETTINGS = {
    "DEBUG": False,
    "ENABLE_HOT_RELOAD": False,
    "ENABLE_PROFILER": False,
    "SECURE_HEADERS": True,
    "RATE_LIMITING": True,
    "MAX_REQUESTS_PER_MINUTE": 60
}

# ฟังก์ชันช่วยเหลือ
def get_project_root() -> Path:
    """รับตำแหน่งโฟลเดอร์รากของโปรเจค"""
    return Path(__file__).parent.parent

def get_absolute_path(relative_path: str) -> str:
    """แปลงพาธสัมพัทธ์เป็นพาธสัมบูรณ์"""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(get_project_root(), relative_path)

def validate_config():
    """ตรวจสอบการตั้งค่าที่จำเป็น"""
    errors = []
    
    # ตรวจสอบไฟล์โมเดล
    model_path = get_absolute_path(MODEL_PATH)
    if not os.path.exists(model_path):
        errors.append(f"ไม่พบไฟล์โมเดล: {model_path}")
    
    # ตรวจสอบไดเรกทอรีฐานข้อมูล
    db_path = get_absolute_path(DATABASE_DIR)
    if not os.path.exists(db_path):
        errors.append(f"ไม่พบไดเรกทอรีฐานข้อมูล: {db_path}")
    
    # ตรวจสอบไดเรกทอรีล็อก
    log_dir = os.path.dirname(get_absolute_path(LOGGING_SETTINGS["LOG_FILE"]))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    return errors

# ตรวจสอบการตั้งค่าเมื่อโหลดโมดูล
_config_errors = validate_config()
if _config_errors:
    import logging
    logger = logging.getLogger(__name__)
    for error in _config_errors:
        logger.warning(error)
