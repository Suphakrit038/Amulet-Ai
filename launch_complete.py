#!/usr/bin/env python3
"""
🔮 Amulet-AI Complete System Launcher
ตัวเปิดระบบครบครัน - รวมทุกฟีเจอร์ในไฟล์เดียว

คุณสมบัติ:
- ✅ ติดตั้ง dependencies อัตโนมัติ
- ✅ ตรวจสอบระบบและแก้ไขปัญหา  
- ✅ เริ่มระบบแบบอัตโนมัติ
- ✅ ตรวจสอบสุขภาพระบบ
- ✅ รองรับทั้ง Windows และ Linux/Mac
- ✅ UI สวยงามพร้อมภาษาไทย

การใช้งาน:
  python launch_complete.py          # เริ่มระบบปกติ
  python launch_complete.py --test   # ทดสอบระบบ
  python launch_complete.py --help   # ดูวิธีใช้
"""
import sys
import os
import subprocess
import time
import requests
import logging
import argparse
import webbrowser
import json
import importlib
from pathlib import Path
from datetime import datetime
import platform

# Setup logging with beautiful formatting
class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green  
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if platform.system() == 'Windows':
            # Skip colors on Windows for compatibility
            return super().format(record)
        
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Apply colored formatter if not on Windows
if platform.system() != 'Windows':
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)

class AmuletAIComplete:
    """ตัวเปิดระบบ Amulet-AI แบบครบครัน"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.frontend_port = 8501
        self.backend_port = 8001
        self.processes = {}
        
        # Print beautiful header
        self.print_header()
        
    def print_header(self):
        """แสดงหัวข้อสวยงาม"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                  🔮 Amulet-AI System                         ║
║              ระบบวิเคราะห์พระเครื่องด้วย AI                   ║
║                                                              ║
║  ✨ พร้อมใช้งาน Production                                    ║
║  🎨 UI ทันสมัย + Responsive                                  ║
║  🇹🇭 รองรับภาษาไทยเต็มรูปแบบ                                 ║
║  🤖 AI Model ขั้นสูง                                          ║
╚══════════════════════════════════════════════════════════════╝
        """)

    def check_python_version(self):
        """ตรวจสอบ Python version"""
        logger.info("🔍 ตรวจสอบ Python version...")
        
        if sys.version_info < (3, 8):
            logger.error("❌ ต้องการ Python 3.8 หรือสูงกว่า")
            logger.error(f"   ปัจจุบันใช้: Python {sys.version}")
            return False
        
        logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True

    def test_imports(self):
        """ทดสอบการ import โมดูลสำคัญ"""
        logger.info("📦 ทดสอบการ import โมดูล...")
        
        modules = [
            ('streamlit', 'Web framework'),
            ('requests', 'HTTP client'),
            ('PIL', 'Image processing'),
            ('numpy', 'Scientific computing')
        ]
        
        missing = []
        for module, desc in modules:
            try:
                mod = importlib.import_module(module)
                version = getattr(mod, '__version__', 'unknown')
                logger.info(f"✅ {module} ({desc}) - v{version}")
            except ImportError:
                logger.warning(f"⚠️ {module} - ไม่พบ")
                missing.append(module)
        
        return len(missing) == 0, missing

    def install_dependencies(self, missing_modules=None):
        """ติดตั้ง dependencies"""
        logger.info("📥 ติดตั้ง dependencies...")
        
        # Base requirements
        packages = ['streamlit>=1.28.0', 'requests', 'pillow', 'numpy']
        
        if missing_modules:
            # Add specific missing modules
            for module in missing_modules:
                if module == 'PIL':
                    packages.append('pillow')
                else:
                    packages.append(module)
        
        # Install from requirements.txt if exists
        req_file = self.project_root / 'requirements.txt'
        if req_file.exists():
            logger.info("📋 ใช้ requirements.txt")
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-r', str(req_file)
                ], check=True, capture_output=True, text=True)
                logger.info("✅ ติดตั้งจาก requirements.txt สำเร็จ")
                return True
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ ติดตั้งจาก requirements.txt ไม่สำเร็จ: {e}")
        
        # Install individual packages
        for package in packages:
            try:
                logger.info(f"📦 ติดตั้ง {package}...")
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], check=True, capture_output=True, text=True)
                logger.info(f"✅ ติดตั้ง {package} สำเร็จ")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ ไม่สามารถติดตั้ง {package}: {e}")
                return False
        
        return True

    def setup_directories(self):
        """สร้างโฟลเดอร์จำเป็น"""
        logger.info("📁 สร้างโฟลเดอร์จำเป็น...")
        
        directories = [
            'logs',
            'uploads',
            'frontend/assets/css',
            'backend/logs',
            'ai_models/saved_models'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ {directory}")
        
        return True

    def find_frontend_app(self):
        """ค้นหาแอป frontend ที่เหมาะสม"""
        logger.info("🔍 ค้นหาแอป frontend...")
        
        # ลำดับความสำคัญ (สไตล์เก่าก่อน)
        frontend_options = [
            ('frontend/app_old_style.py', 'Old Style UI (สไตล์เก่าสวยงาม)'),
            ('frontend/main_app.py', 'Main UI (แอปหลัก)'),
            ('frontend/app_simple.py', 'Simple UI (เรียบง่าย)'),
            ('frontend/modern_ui.py', 'Modern UI (ทันสมัย)'),
            ('frontend/unified_interface.py', 'Unified Interface (รวมทุกอย่าง)')
        ]
        
        for app_path, description in frontend_options:
            full_path = self.project_root / app_path
            if full_path.exists():
                logger.info(f"✅ พบ: {description}")
                return str(full_path), description
        
        logger.error("❌ ไม่พบไฟล์ frontend")
        return None, None

    def check_ports(self):
        """ตรวจสอบ port ที่ใช้งาน"""
        logger.info("🔌 ตรวจสอบ ports...")
        
        ports_to_check = [self.frontend_port, self.backend_port]
        
        for port in ports_to_check:
            try:
                response = requests.get(f'http://localhost:{port}', timeout=2)
                logger.warning(f"⚠️ Port {port} กำลังใช้งาน")
            except requests.exceptions.RequestException:
                logger.info(f"✅ Port {port} ว่าง")
        
        return True

    def start_frontend(self, app_path):
        """เริ่ม frontend application"""
        logger.info("🎨 เริ่ม frontend application...")
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update({
                'STREAMLIT_SERVER_PORT': str(self.frontend_port),
                'STREAMLIT_SERVER_ADDRESS': '0.0.0.0',
                'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
                'STREAMLIT_SERVER_ENABLE_CORS': 'true',
                'PYTHONPATH': str(self.project_root)
            })
            
            # Start streamlit
            logger.info(f"🚀 เริ่ม Streamlit: {app_path}")
            logger.info(f"🌐 URL: http://localhost:{self.frontend_port}")
            
            # Open browser after a short delay
            def open_browser():
                time.sleep(3)
                try:
                    webbrowser.open(f'http://localhost:{self.frontend_port}')
                    logger.info("🌐 เปิดเบราว์เซอร์แล้ว")
                except:
                    logger.info("💡 กรุณาเปิดเบราว์เซอร์ไปที่: http://localhost:8501")
            
            import threading
            threading.Thread(target=open_browser, daemon=True).start()
            
            # Run streamlit
            subprocess.run([
                sys.executable, '-m', 'streamlit', 'run', app_path,
                '--server.port', str(self.frontend_port),
                '--server.address', '0.0.0.0',
                '--browser.gatherUsageStats', 'false',
                '--server.enableCORS', 'true'
            ], env=env)
            
        except KeyboardInterrupt:
            logger.info("🛑 หยุดระบบโดยผู้ใช้")
        except Exception as e:
            logger.error(f"❌ ไม่สามารถเริ่ม frontend: {e}")
            return False
        
        return True

    def run_tests(self):
        """รันการทดสอบระบบ"""
        logger.info("🧪 ทดสอบระบบ...")
        
        tests = [
            ("Python Version", self.check_python_version),
            ("Import Modules", lambda: self.test_imports()[0]),
            ("Directory Setup", self.setup_directories),
            ("Port Check", self.check_ports),
            ("Frontend App", lambda: self.find_frontend_app()[0] is not None)
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"🔍 {test_name}...")
            try:
                result = test_func()
                if result:
                    logger.info(f"✅ {test_name}: ผ่าน")
                    results.append(True)
                else:
                    logger.error(f"❌ {test_name}: ไม่ผ่าน")
                    results.append(False)
            except Exception as e:
                logger.error(f"❌ {test_name}: ข้อผิดพลาด - {e}")
                results.append(False)
        
        passed = sum(results)
        total = len(results)
        
        print(f"\n{'='*60}")
        print(f"📊 ผลการทดสอบ: {passed}/{total} ผ่าน")
        
        if all(results):
            print("🎉 ระบบพร้อมใช้งาน!")
            return True
        else:
            print("⚠️ พบปัญหา กรุณาแก้ไขก่อนใช้งาน")
            return False

    def create_launch_summary(self):
        """สร้างสรุปการใช้งาน"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.system(),
            "frontend_port": self.frontend_port,
            "backend_port": self.backend_port,
            "project_root": str(self.project_root)
        }
        
        # Save summary
        summary_file = self.project_root / 'logs' / 'launch_summary.json'
        summary_file.parent.mkdir(exist_ok=True)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 บันทึกสรุป: {summary_file}")

    def launch(self, test_only=False):
        """เริ่มระบบหลัก"""
        logger.info("🚀 เริ่มต้นระบบ Amulet-AI...")
        
        try:
            # 1. Check Python
            if not self.check_python_version():
                return False
            
            # 2. Test imports and install if needed
            import_ok, missing = self.test_imports()
            if not import_ok:
                logger.info("📥 ติดตั้ง dependencies ที่ขาดหายไป...")
                if not self.install_dependencies(missing):
                    return False
                
                # Test again after installation
                import_ok, _ = self.test_imports()
                if not import_ok:
                    logger.error("❌ ยังคงมีปัญหากับ dependencies")
                    return False
            
            # 3. Setup directories
            self.setup_directories()
            
            # 4. Find frontend app
            app_path, app_desc = self.find_frontend_app()
            if not app_path:
                return False
            
            logger.info(f"🎯 ใช้: {app_desc}")
            
            # 5. Check ports
            self.check_ports()
            
            # 6. Create launch summary
            self.create_launch_summary()
            
            # 7. If test only, stop here
            if test_only:
                logger.info("🧪 โหมดทดสอบเสร็จสิ้น")
                return self.run_tests()
            
            # 8. Start the system
            logger.info("🎨 เริ่มระบบ...")
            print(f"""
{'='*60}
🎉 ระบบ Amulet-AI พร้อมแล้ว!

🌐 เข้าใช้งาน: http://localhost:{self.frontend_port}
🛑 หยุดระบบ: กด Ctrl+C
💡 ปิดหน้าต่างนี้เพื่อหยุดระบบ

แอปที่ใช้: {app_desc}
{'='*60}
            """)
            
            return self.start_frontend(app_path)
            
        except KeyboardInterrupt:
            logger.info("🛑 หยุดระบบโดยผู้ใช้")
            return True
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาด: {e}")
            return False

def main():
    """ฟังก์ชันหลัก"""
    parser = argparse.ArgumentParser(
        description="🔮 Amulet-AI Complete System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ตัวอย่างการใช้งาน:
  python launch_complete.py           # เริ่มระบบปกติ
  python launch_complete.py --test    # ทดสอบระบบ
  python launch_complete.py --help    # ดูคำแนะนำ

🌐 หลังเริ่มระบบ เปิดเบราว์เซอร์ไปที่: http://localhost:8501
        """
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='รันโหมดทดสอบระบบ (ไม่เริ่มเว็บแอป)'
    )
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = AmuletAIComplete()
    
    # Run
    success = launcher.launch(test_only=args.test)
    
    if success:
        if args.test:
            print("\n✅ การทดสอบสำเร็จ!")
        else:
            print("\n👋 ขอบคุณที่ใช้ Amulet-AI!")
    else:
        print("\n❌ เกิดข้อผิดพลาด")
        print("\n🔧 แก้ไขปัญหา:")
        print("1. ตรวจสอบ Python version >= 3.8")
        print("2. รัน: pip install streamlit requests pillow numpy")
        print("3. ตรวจสอบไฟล์ frontend ใน folder frontend/")
        sys.exit(1)

if __name__ == "__main__":
    main()