#!/usr/bin/env python3
"""
Amulet-AI System Launcher
เริ่มระบบ Amulet-AI แบบอัตโนมัติ
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AmuletAILauncher:
    """ตัวเริ่มระบบ Amulet-AI"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.frontend_port = 8501
        self.backend_port = 8001
        
    def check_python_environment(self):
        """ตรวจสอบ Python environment"""
        logger.info("🔍 ตรวจสอบ Python environment...")
        
        # ตรวจสอบ Python version
        if sys.version_info < (3, 8):
            logger.error("❌ ต้องการ Python 3.8 หรือสูงกว่า")
            return False
        
        logger.info(f"✅ Python {sys.version}")
        return True
    
    def install_requirements(self):
        """ติดตั้ง dependencies"""
        logger.info("📦 ติดตั้ง dependencies...")
        
        requirements_files = [
            "requirements.txt",
            "ai_models/configs/requirements_advanced.txt"
        ]
        
        for req_file in requirements_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                try:
                    subprocess.run([
                        sys.executable, '-m', 'pip', 'install', '-r', str(req_path)
                    ], check=True, capture_output=True)
                    logger.info(f"✅ ติดตั้ง {req_file} สำเร็จ")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"⚠️ ไม่สามารถติดตั้ง {req_file}: {e}")
        
        return True
    
    def setup_directories(self):
        """สร้างโฟลเดอร์ที่จำเป็น"""
        logger.info("📁 สร้างโฟลเดอร์ที่จำเป็น...")
        
        directories = [
            "logs",
            "uploads", 
            "frontend/assets/css",
            "frontend/assets/js",
            "backend/logs",
            "ai_models/saved_models"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 {directory}")
        
        return True
    
    def check_backend_available(self):
        """ตรวจสอบว่า backend พร้อมใช้งาน"""
        backend_script = self.project_root / "backend" / "api_with_real_model.py"
        return backend_script.exists()
    
    def start_backend(self):
        """เริ่ม backend server"""
        if not self.check_backend_available():
            logger.warning("⚠️ Backend ไม่พร้อมใช้งาน จะใช้ mock data")
            return True
        
        logger.info("🚀 เริ่ม backend server...")
        
        try:
            # เริ่ม backend แบบ detached process
            cmd = [
                sys.executable, '-m', 'uvicorn',
                'backend.api_with_real_model:app',
                '--host', '0.0.0.0',
                '--port', str(self.backend_port),
                '--reload'
            ]
            
            subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # รอให้ backend เริ่มทำงาน
            for i in range(10):
                try:
                    response = requests.get(f"http://localhost:{self.backend_port}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info(f"✅ Backend เริ่มทำงานที่ port {self.backend_port}")
                        return True
                except:
                    time.sleep(2)
            
            logger.warning("⚠️ Backend อาจยังไม่พร้อม แต่จะดำเนินการต่อ")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ ไม่สามารถเริ่ม backend: {e}")
            return True  # ดำเนินการต่อแม้ไม่มี backend
    
    def start_frontend(self):
        """เริ่ม frontend Streamlit"""
        logger.info("🎨 เริ่ม frontend...")
        
        # ลองใช้แอปตามลำดับความสำคัญ (สไตล์เก่าก่อน)
        frontend_options = [
            self.project_root / "frontend" / "app_old_style.py",  # สไตล์เก่าที่สวยงาม (ใหม่)
            self.project_root / "frontend" / "app_streamlit.py",  # สไตล์เก่าต้นฉบับ
            self.project_root / "frontend" / "app_simple.py",    # สไตล์ใหม่แบบง่าย
            self.project_root / "frontend" / "app_modern.py",    # สำรอง
        ]
        
        frontend_script = None
        for option in frontend_options:
            if option.exists():
                frontend_script = option
                logger.info(f"📱 ใช้ frontend: {option.name}")
                break
        
        if not frontend_script:
            logger.error("❌ ไม่พบไฟล์ frontend")
            return False
        
        try:
            cmd = [
                sys.executable, '-m', 'streamlit', 'run',
                str(frontend_script),
                '--server.port', str(self.frontend_port),
                '--server.address', '0.0.0.0',
                '--browser.gatherUsageStats', 'false'
            ]
            
            logger.info(f"🌐 เปิดเบราว์เซอร์ไปที่: http://localhost:{self.frontend_port}")
            subprocess.run(cmd, cwd=self.project_root)
            
        except KeyboardInterrupt:
            logger.info("🛑 ระบบถูกหยุดโดยผู้ใช้")
        except Exception as e:
            logger.error(f"❌ ไม่สามารถเริ่ม frontend: {e}")
            return False
        
        return True
    
    def launch(self):
        """เริ่มระบบทั้งหมด"""
        logger.info("🔮 เริ่มระบบ Amulet-AI...")
        
        try:
            # ตรวจสอบ environment
            if not self.check_python_environment():
                return False
            
            # สร้างโฟลเดอร์
            self.setup_directories()
            
            # ติดตั้ง dependencies
            self.install_requirements()
            
            # เริ่ม backend
            self.start_backend()
            
            # เริ่ม frontend
            self.start_frontend()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("🛑 การเริ่มระบบถูกยกเลิก")
            return False
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาด: {e}")
            return False


def main():
    """ฟังก์ชันหลัก"""
    print("""
    🔮 Amulet-AI System Launcher
    ============================
    ระบบวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์
    
    """)
    
    launcher = AmuletAILauncher()
    success = launcher.launch()
    
    if success:
        print("\n✅ ระบบเริ่มทำงานสำเร็จ!")
    else:
        print("\n❌ ไม่สามารถเริ่มระบบได้")
        sys.exit(1)


if __name__ == "__main__":
    main()
