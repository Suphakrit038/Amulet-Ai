"""
สคริปต์สำหรับทำความสะอาดและลบไฟล์ที่ไม่จำเป็นหลังจากรวมระบบ
"""

import os
import shutil
import logging
from pathlib import Path

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cleanup")

# รายการไฟล์ที่ไม่จำเป็นอีกต่อไป
FILES_TO_REMOVE = [
    "frontend/comparison_module.py",
    "frontend/comparison_connector.py",
    "frontend/compare_functions.py",
    "frontend/unified_tools.py",
    "frontend/app_comparison.py",
    "frontend/comparison_app.log",
]

# โฟลเดอร์ที่ไม่จำเป็นอีกต่อไป
FOLDERS_TO_REMOVE = [
    "frontend/models",
]

def cleanup_files():
    """ลบไฟล์ที่ไม่จำเป็นอีกต่อไป"""
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # ลบไฟล์
    for file_path in FILES_TO_REMOVE:
        full_path = project_root / file_path
        if full_path.exists():
            try:
                os.remove(full_path)
                logger.info(f"ลบไฟล์: {file_path}")
            except Exception as e:
                logger.error(f"ไม่สามารถลบไฟล์ {file_path}: {e}")
        else:
            logger.warning(f"ไม่พบไฟล์: {file_path}")
    
    # ลบโฟลเดอร์
    for folder_path in FOLDERS_TO_REMOVE:
        full_path = project_root / folder_path
        if full_path.exists():
            try:
                shutil.rmtree(full_path)
                logger.info(f"ลบโฟลเดอร์: {folder_path}")
            except Exception as e:
                logger.error(f"ไม่สามารถลบโฟลเดอร์ {folder_path}: {e}")
        else:
            logger.warning(f"ไม่พบโฟลเดอร์: {folder_path}")

def create_missing_directories():
    """สร้างโฟลเดอร์ที่จำเป็นหากยังไม่มี"""
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # รายการโฟลเดอร์ที่จำเป็น
    required_folders = [
        "logs",
        "temp",
        "data_base",
    ]
    
    for folder in required_folders:
        folder_path = project_root / folder
        if not folder_path.exists():
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"สร้างโฟลเดอร์: {folder}")
            except Exception as e:
                logger.error(f"ไม่สามารถสร้างโฟลเดอร์ {folder}: {e}")

def run_cleanup():
    """ดำเนินการทำความสะอาดทั้งหมด"""
    logger.info("เริ่มต้นการทำความสะอาดระบบ...")
    
    # ลบไฟล์และโฟลเดอร์ที่ไม่จำเป็น
    cleanup_files()
    
    # สร้างโฟลเดอร์ที่จำเป็น
    create_missing_directories()
    
    logger.info("ทำความสะอาดระบบเสร็จสิ้น")

if __name__ == "__main__":
    run_cleanup()
