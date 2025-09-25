"""
🚀 Amulet-AI API Launcher
เครื่องมือเริ่มระบบ API อย่างง่าย
"""
import os
import sys
import logging
from pathlib import Path

# ตั้งค่า path
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("launcher")

def check_requirements():
    """ตรวจสอบการติดตั้ง package ที่จำเป็น"""
    required_packages = ["fastapi", "uvicorn", "python-multipart"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"⚠️ ไม่พบ package ที่จำเป็น: {', '.join(missing_packages)}")
        logger.info("🔄 กำลังติดตั้ง package ที่จำเป็น...")
        
        try:
            import subprocess
            cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
            subprocess.check_call(cmd)
            logger.info("✅ ติดตั้ง package เรียบร้อยแล้ว")
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดในการติดตั้ง package: {e}")
            logger.info("ลองติดตั้งเองด้วยคำสั่ง: pip install fastapi uvicorn python-multipart")
            return False
    
    return True

def launch_api(host="127.0.0.1", port=8000):
    """เริ่มระบบ API"""
    if not check_requirements():
        logger.error("❌ ไม่สามารถเริ่มระบบ API ได้เนื่องจากขาด package ที่จำเป็น")
        return
    
    try:
        # นำเข้าโมดูลที่จำเป็น
        import uvicorn
        from backend.api.production_api import app
        
        logger.info(f"🚀 กำลังเริ่มระบบ API ที่ http://{host}:{port}...")
        logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
        uvicorn.run(app, host=host, port=port)
        
    except ImportError as e:
        logger.error(f"❌ ไม่สามารถนำเข้าโมดูล API: {e}")
        logger.info("โปรดตรวจสอบว่าโครงสร้างไฟล์ถูกต้อง")
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการเริ่มระบบ API: {e}")

if __name__ == "__main__":
    # ตรวจสอบ command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Amulet-AI API Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()
    
    launch_api(host=args.host, port=args.port)
