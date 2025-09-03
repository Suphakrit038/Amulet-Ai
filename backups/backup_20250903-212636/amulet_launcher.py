#!/usr/bin/env python3
"""
🏺 Amulet-AI Unified Launcher
เปิดระบบ AI ทั้งหมด - Backend + Frontend + โมเดล AI
รวมความสามารถของทุกไฟล์ launcher เข้าด้วยกัน
"""

import subprocess
import time
import os
import sys
import threading
import socket
import signal
import logging
from pathlib import Path
import webbrowser
from datetime import datetime

# คอนฟิกการบันทึกล็อก
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('amulet-launcher')

class AmuletSystemLauncher:
    """
    ตัวเปิดระบบ Amulet-AI ครบวงจร
    รวมความสามารถของ launcher ทั้งหมดเข้าด้วยกัน
    """
    
    def __init__(self):
        self.processes = []
        self.base_dir = Path(__file__).parent
        self.running = True
        self.config = {
            'api_host': '127.0.0.1',
            'api_port': 8000,
            'frontend_host': '127.0.0.1',
            'frontend_port': 8501,
            'use_real_model': False,
            'open_browser': True
        }
        
    def check_dependencies(self):
        """ตรวจสอบการติดตั้ง dependencies"""
        required_packages = [
            ('fastapi', 'fastapi'), 
            ('uvicorn', 'uvicorn'), 
            ('streamlit', 'streamlit'), 
            ('requests', 'requests'), 
            ('PIL', 'pillow'), 
            ('numpy', 'numpy')
        ]
        
        missing_packages = []
        
        for import_name, package_name in required_packages:
            try:
                __import__(import_name)
                logger.info(f"✅ {package_name} - OK")
            except ImportError:
                missing_packages.append(package_name)
                logger.error(f"❌ {package_name} - Missing")
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Install with: pip install " + " ".join(missing_packages))
            
            # ถามผู้ใช้ว่าต้องการติดตั้งแพ็กเกจที่ขาดหรือไม่
            if input("ติดตั้งแพ็กเกจที่ขาดหรือไม่? (y/n): ").strip().lower() == 'y':
                try:
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install', *missing_packages
                    ])
                    logger.info("✅ ติดตั้งแพ็กเกจเรียบร้อยแล้ว")
                    return True
                except subprocess.CalledProcessError:
                    logger.error("❌ ติดตั้งแพ็กเกจไม่สำเร็จ")
                    return False
            return False
        
        return True
    
    def check_ai_model(self):
        """ตรวจสอบว่ามีโมเดล AI หรือไม่"""
        # ตรวจสอบโมเดลจำลอง
        mock_model_files = [
            "ai_models/amulet_model.h5",
            "ai_models/labels.json"
        ]
        
        mock_model_exists = True
        for model_file in mock_model_files:
            full_path = self.base_dir / model_file
            if full_path.exists():
                logger.info(f"✅ {model_file} - Found")
            else:
                logger.warning(f"⚠️ {model_file} - Not found")
                mock_model_exists = False
        
        # ตรวจสอบโมเดลจริง
        real_model_files = [
            "ai_models/training_output/step5_final_model.pth",
            "ai_models/training_output/PRODUCTION_MODEL_INFO.json"
        ]
        
        real_model_exists = True
        for model_file in real_model_files:
            full_path = self.base_dir / model_file
            if full_path.exists():
                logger.info(f"✅ {model_file} - Found")
            else:
                logger.warning(f"⚠️ {model_file} - Not found")
                real_model_exists = False
        
        # ตั้งค่าการใช้โมเดลจริงหรือโมเดลจำลอง
        if real_model_exists:
            logger.info("🧠 Using real AI model")
            self.config['use_real_model'] = True
        else:
            if mock_model_exists:
                logger.info("🧠 Using mock AI model")
                self.config['use_real_model'] = False
            else:
                logger.warning("⚠️ No AI models found, using mock data")
                self.config['use_real_model'] = False
        
        return mock_model_exists or real_model_exists
    
    def wait_for_service(self, host, port, name, max_attempts=30):
        """รอให้บริการพร้อมใช้งาน"""
        logger.info(f"⏳ Waiting for {name} on {host}:{port}...")
        
        for attempt in range(max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex((host, int(port)))
                    if result == 0:
                        logger.info(f"✅ {name} is ready!")
                        return True
            except:
                pass
            
            time.sleep(1)
            logger.info(f"⏳ Attempt {attempt + 1}/{max_attempts}...")
        
        logger.error(f"❌ {name} failed to start within timeout")
        return False
    
    def get_python_executable(self):
        """รับค่า Python executable ที่ถูกต้อง"""
        # ตรวจสอบ virtual environment ก่อน
        if sys.prefix != sys.base_prefix:
            # กำลังอยู่ใน virtual environment
            return sys.executable
        
        # ตรวจสอบ virtual environment ในโฟลเดอร์โปรเจค
        venv_paths = [
            Path(".venv/Scripts/python.exe"),  # Windows
            Path(".venv/bin/python"),          # Linux/Mac
            Path("venv/Scripts/python.exe"),   # Windows alternative
            Path("venv/bin/python")            # Linux/Mac alternative
        ]
        
        for venv_path in venv_paths:
            if venv_path.exists():
                return str(venv_path)
        
        # ถ้าไม่มี venv ใช้ python ปกติ
        return sys.executable
    
    def start_backend_api(self):
        """เริ่ม Backend API"""
        logger.info("🚀 Starting Backend API...")
        
        python_exe = self.get_python_executable()
        
        # เลือกโมดูล API ตามประเภทของโมเดล
        if self.config['use_real_model']:
            api_module = "backend.api_with_real_model:app"
            port = "8001"
        else:
            api_module = "backend.api:app"
            port = "8000"
        
        self.config['api_port'] = port
        
        cmd = [
            python_exe, "-m", "uvicorn", 
            api_module,
            "--host", self.config['api_host'],
            "--port", port,
            "--reload"
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append(('Backend API', process))
            logger.info(f"✅ Backend API process started on port {port}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start Backend API: {e}")
            return False
    
    def start_frontend(self):
        """เริ่ม Frontend"""
        logger.info("🎨 Starting Frontend...")
        
        python_exe = self.get_python_executable()
        
        # ตรวจสอบว่ามีไฟล์ frontend
        frontend_files = [
            "frontend/app_streamlit.py",
            "frontend/app_straemlit.py"  # ตรวจสอบกรณีที่มีการสะกดผิด
        ]
        
        frontend_file = None
        for file in frontend_files:
            if Path(file).exists():
                frontend_file = file
                break
        
        if not frontend_file:
            logger.error("❌ Frontend file not found")
            return False
        
        cmd = [
            python_exe, "-m", "streamlit", "run",
            frontend_file,
            "--server.port", str(self.config['frontend_port']),
            "--server.address", self.config['frontend_host']
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append(('Frontend', process))
            logger.info("✅ Frontend process started")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start Frontend: {e}")
            return False
    
    def open_web_browser(self):
        """เปิดเว็บเบราว์เซอร์"""
        if not self.config['open_browser']:
            return
        
        frontend_url = f"http://{self.config['frontend_host']}:{self.config['frontend_port']}"
        
        logger.info(f"🌐 Opening web browser: {frontend_url}")
        try:
            time.sleep(3)  # รอให้ frontend พร้อมใช้งาน
            webbrowser.open(frontend_url)
        except Exception as e:
            logger.error(f"❌ Failed to open web browser: {e}")
    
    def test_api_connection(self):
        """ทดสอบการเชื่อมต่อกับ API"""
        logger.info("🧪 Testing API connection...")
        
        api_url = f"http://{self.config['api_host']}:{self.config['api_port']}"
        
        try:
            # ทดสอบ health check
            import requests
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ API Health: {data}")
                return True
            else:
                logger.error(f"❌ API Health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ API connection test failed: {e}")
            return False
    
    def monitor_processes(self):
        """ตรวจสอบการทำงานของโปรเซส"""
        while self.running:
            try:
                for name, process in self.processes:
                    if process.poll() is not None:
                        logger.error(f"❌ {name} process has stopped!")
                        return False
                
                time.sleep(5)  # ตรวจสอบทุก 5 วินาที
                
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal")
                break
        
        return True
    
    def stop_all_processes(self):
        """หยุดการทำงานของทุกโปรเซส"""
        logger.info("🛑 Stopping all processes...")
        
        self.running = False
        
        for name, process in self.processes:
            try:
                logger.info(f"Stopping {name}...")
                process.terminate()
                
                # รอให้โปรเซสหยุดทำงาน
                try:
                    process.wait(timeout=5)
                    logger.info(f"✅ {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ Force killing {name}")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
        
        logger.info("✅ All processes stopped")
    
    def print_system_info(self):
        """แสดงข้อมูลระบบ"""
        print("\n" + "="*60)
        print("🏺 AMULET-AI SYSTEM")
        print("="*60)
        print(f"📅 Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Base Directory: {self.base_dir}")
        print(f"🐍 Python: {sys.version.split()[0]}")
        print(f"🤖 AI Mode: {'Real AI Model' if self.config['use_real_model'] else 'Mock Data'}")
        print("="*60)
        print("🌐 URLs:")
        print(f"  Backend API:  http://{self.config['api_host']}:{self.config['api_port']}")
        print(f"  API Docs:     http://{self.config['api_host']}:{self.config['api_port']}/docs")
        print(f"  Frontend:     http://{self.config['frontend_host']}:{self.config['frontend_port']}")
        print("="*60)
        print("⌨️ Controls:")
        print("  Ctrl+C:       Stop all services")
        print("="*60)
    
    def run(self):
        """เริ่มระบบทั้งหมด"""
        try:
            # แสดงข้อมูลระบบ
            self.print_system_info()
            
            # ตรวจสอบ dependencies
            logger.info("🔍 Checking dependencies...")
            if not self.check_dependencies():
                logger.error("❌ Missing dependencies. Please install required packages.")
                return False
            
            # ตรวจสอบโมเดล AI
            logger.info("🧠 Checking AI model...")
            self.check_ai_model()
            
            # เริ่ม backend API
            if not self.start_backend_api():
                logger.error("❌ Failed to start backend")
                return False
            
            # รอให้ API พร้อมใช้งาน
            if not self.wait_for_service(self.config['api_host'], self.config['api_port'], "Backend API"):
                logger.error("❌ API not responding")
                self.stop_all_processes()
                return False
            
            # ทดสอบการเชื่อมต่อกับ API
            if not self.test_api_connection():
                logger.warning("⚠️ API connection test failed, but continuing...")
            
            # เริ่ม frontend
            if not self.start_frontend():
                logger.error("❌ Failed to start frontend")
                self.stop_all_processes()
                return False
            
            # รอให้ frontend พร้อมใช้งาน
            if not self.wait_for_service(self.config['frontend_host'], self.config['frontend_port'], "Frontend"):
                logger.error("❌ Frontend not responding")
                self.stop_all_processes()
                return False
            
            # เปิดเว็บเบราว์เซอร์ (ถ้าต้องการ)
            browser_thread = threading.Thread(target=self.open_web_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            # แสดงข้อความว่าระบบพร้อมใช้งาน
            print("\n🎉 SYSTEM READY!")
            print(f"✅ Backend API: http://{self.config['api_host']}:{self.config['api_port']}")
            print(f"✅ Frontend:    http://{self.config['frontend_host']}:{self.config['frontend_port']}")
            print(f"📖 API Docs:   http://{self.config['api_host']}:{self.config['api_port']}/docs")
            print("\n💡 The system is now running. Access the frontend to use Amulet-AI!")
            print("🛑 Press Ctrl+C to stop all services\n")
            
            # ตรวจสอบการทำงานของโปรเซส
            self.monitor_processes()
            
        except KeyboardInterrupt:
            logger.info("🛑 Received shutdown signal")
        except Exception as e:
            logger.error(f"❌ System error: {e}")
        finally:
            self.stop_all_processes()


def signal_handler(sig, frame):
    """จัดการสัญญาณการปิดโปรแกรม"""
    print("\n🛑 Received interrupt signal. Shutting down...")
    sys.exit(0)


def main():
    """ฟังก์ชันหลัก"""
    # จัดการ Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    launcher = AmuletSystemLauncher()
    
    # ตรวจสอบพารามิเตอร์คำสั่ง
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            # โหมดทดสอบ - เพียงตรวจสอบว่าทุกอย่างทำงานปกติ
            if launcher.check_dependencies():
                launcher.check_ai_model()
                launcher.print_system_info()
                logger.info("✅ System check complete")
            return
        elif sys.argv[1] == '--api-only':
            # เริ่มเฉพาะ API
            launcher.start_backend_api()
            launcher.wait_for_service(launcher.config['api_host'], launcher.config['api_port'], "Backend API")
            launcher.test_api_connection()
            launcher.monitor_processes()
            return
        elif sys.argv[1] == '--real-model':
            # ใช้โมเดลจริง
            launcher.config['use_real_model'] = True
            logger.info("🧠 Using real AI model")
        elif sys.argv[1] == '--no-browser':
            # ไม่เปิดเว็บเบราว์เซอร์อัตโนมัติ
            launcher.config['open_browser'] = False
            logger.info("🌐 Automatic browser opening disabled")
    
    # เริ่มระบบทั้งหมด
    launcher.run()


if __name__ == "__main__":
    main()
