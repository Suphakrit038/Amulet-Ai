#!/usr/bin/env python3
"""
Amulet-AI System Launcher
สคริปต์สำหรับเริ่มระบบ Amulet-AI อย่างง่าย
"""

import os
import sys
import subprocess
import time
import threading
import signal
import webbrowser
from pathlib import Path

class AmuletSystemLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.root_dir = Path(__file__).parent
        
    def check_dependencies(self):
        """ตรวจสอบการติดตั้ง dependencies"""
        print("🔍 ตรวจสอบการติดตั้ง...")
        
        required_packages = [
            'streamlit',
            'fastapi',
            'uvicorn',
            'requests',
            'Pillow'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ ขาด packages: {', '.join(missing_packages)}")
            print("📦 กำลังติดตั้งแพ็กเกจที่ขาด...")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', 
                    *missing_packages
                ])
                print("✅ ติดตั้งแพ็กเกจสำเร็จ")
            except subprocess.CalledProcessError:
                print("❌ ติดตั้งแพ็กเกจไม่สำเร็จ")
                return False
        else:
            print("✅ แพ็กเกจครบถ้วน")
        
        return True
    
    def start_backend(self):
        """เริ่ม Backend API"""
        print("🚀 กำลังเริ่ม Backend API...")
        
        backend_dir = self.root_dir / "backend"
        if not backend_dir.exists():
            print("❌ ไม่พบโฟลเดอร์ backend")
            return False
        
        try:
            self.backend_process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn',
                'api:app',
                '--host', '0.0.0.0',
                '--port', '8000',
                '--reload'
            ], cwd=backend_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            
            # รอ backend เริ่มทำงาน
            time.sleep(3)
            
            # ตรวจสอบว่า backend ทำงานอยู่
            if self.backend_process.poll() is None:
                print("✅ Backend API เริ่มทำงานที่ http://localhost:8000")
                return True
            else:
                print("❌ Backend API เริ่มไม่สำเร็จ")
                return False
                
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการเริ่ม Backend: {e}")
            return False
    
    def start_frontend(self):
        """เริ่ม Frontend UI"""
        print("🌐 กำลังเริ่ม Frontend UI...")
        
        frontend_dir = self.root_dir / "frontend"
        if not frontend_dir.exists():
            print("❌ ไม่พบโฟลเดอร์ frontend")
            return False
        
        app_file = frontend_dir / "app_straemlit.py"
        if not app_file.exists():
            print("❌ ไม่พบไฟล์ app_straemlit.py")
            return False
        
        try:
            self.frontend_process = subprocess.Popen([
                sys.executable, '-m', 'streamlit', 'run',
                'app_straemlit.py',
                '--server.port', '8501',
                '--server.headless', 'true'
            ], cwd=frontend_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            
            # รอ frontend เริ่มทำงาน
            time.sleep(5)
            
            if self.frontend_process.poll() is None:
                print("✅ Frontend UI เริ่มทำงานที่ http://localhost:8501")
                return True
            else:
                print("❌ Frontend UI เริ่มไม่สำเร็จ")
                return False
                
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการเริ่ม Frontend: {e}")
            return False
    
    def open_browser(self):
        """เปิดเว็บเบราว์เซอร์"""
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:8501')
            print("🌐 เปิดเว็บเบราว์เซอร์แล้ว")
        except:
            print("⚠️ ไม่สามารถเปิดเว็บเบราว์เซอร์อัตโนมัติ")
            print("กรุณาเปิด http://localhost:8501 ด้วยตนเอง")
    
    def cleanup(self):
        """ปิดระบบ"""
        print("\n🔄 กำลังปิดระบบ...")
        
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                print("✅ ปิด Frontend สำเร็จ")
            except:
                self.frontend_process.kill()
        
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=5)
                print("✅ ปิด Backend สำเร็จ")
            except:
                self.backend_process.kill()
        
        print("👋 ปิดระบบเรียบร้อยแล้ว")
    
    def signal_handler(self, signum, frame):
        """จัดการสัญญาณปิดระบบ"""
        self.cleanup()
        sys.exit(0)
    
    def run(self):
        """เริ่มระบบทั้งหมด"""
        print("=" * 50)
        print("🏺 Amulet-AI System Launcher")
        print("ระบบปัญญาประดิษฐ์ระบุพระเครื่อง")
        print("=" * 50)
        
        # ตั้งค่า signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # ตรวจสอบ dependencies
            if not self.check_dependencies():
                return
            
            # เริ่ม Backend
            if not self.start_backend():
                return
            
            # เริ่ม Frontend
            if not self.start_frontend():
                self.cleanup()
                return
            
            # เปิดเบราว์เซอร์
            browser_thread = threading.Thread(target=self.open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            print("\n" + "=" * 50)
            print("🎉 ระบบเริ่มทำงานเรียบร้อยแล้ว!")
            print("🌐 Frontend: http://localhost:8501")
            print("🚀 Backend API: http://localhost:8000")
            print("📚 API Docs: http://localhost:8000/docs")
            print("\nกด Ctrl+C เพื่อปิดระบบ")
            print("=" * 50)
            
            # รอจนกว่าจะได้รับสัญญาณปิด
            while True:
                if self.backend_process and self.backend_process.poll() is not None:
                    print("❌ Backend หยุดทำงาน")
                    break
                
                if self.frontend_process and self.frontend_process.poll() is not None:
                    print("❌ Frontend หยุดทำงาน")
                    break
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")
        finally:
            self.cleanup()

def main():
    """Main function"""
    launcher = AmuletSystemLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
