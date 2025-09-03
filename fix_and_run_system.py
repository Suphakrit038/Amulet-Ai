"""
แก้ปัญหาและเริ่มใช้งานระบบ Amulet-AI ทั้งหมด
"""
import os
import sys
import time
import subprocess
import webbrowser
import signal
import requests
from pathlib import Path

def kill_processes(process_names):
    """ยุติการทำงานของกระบวนการที่อาจจะค้าง"""
    print("🛑 กำลังหยุดโปรเซสเก่า...")
    
    import psutil
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if any(name.lower() in proc.name().lower() for name in process_names):
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                    print(f"    ยุติ {proc.name()} (PID: {proc.pid})")
                except Exception as e:
                    print(f"    ไม่สามารถยุติ {proc.name()} (PID: {proc.pid}): {e}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    time.sleep(1)  # รอให้โปรเซสหยุดทำงาน

def find_executable(name):
    """ค้นหา executable ในสภาพแวดล้อมเสมือน"""
    base_dir = Path(__file__).parent
    venv_dir = base_dir / '.venv'
    
    if sys.platform == 'win32':
        exec_path = venv_dir / 'Scripts' / f"{name}.exe"
        python_path = venv_dir / 'Scripts' / 'python.exe'
    else:
        exec_path = venv_dir / 'bin' / name
        python_path = venv_dir / 'bin' / 'python'
    
    if exec_path.exists():
        return str(exec_path)
    elif python_path.exists():
        return f"{python_path} -m {name}"
    
    return name  # กลับไปใช้ system-wide

def start_backend():
    """เริ่มใช้งาน backend API"""
    print("\n🚀 กำลังเริ่มต้น Backend API...")
    
    backend_script = Path(__file__).parent / 'backend' / 'mock_api.py'
    python_exe = find_executable('python').split(' ')[0]  # เอาเฉพาะส่วน path
    
    if not backend_script.exists():
        print(f"❌ ไม่พบไฟล์ {backend_script}")
        sys.exit(1)
    
    command = [python_exe, str(backend_script)]
    try:
        process = subprocess.Popen(command, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0)
        
        # รอให้ API เริ่มทำงาน
        print("⏳ รอให้ API เริ่มทำงาน...")
        start_time = time.time()
        while time.time() - start_time < 10:  # รอไม่เกิน 10 วินาที
            try:
                response = requests.get("http://127.0.0.1:8000/health", timeout=1)
                if response.status_code == 200:
                    print("✅ Backend API พร้อมใช้งานแล้ว!")
                    return True
            except requests.exceptions.RequestException:
                time.sleep(0.5)
        
        print("⚠️ ไม่สามารถตรวจสอบสถานะ API ได้ แต่จะดำเนินการต่อ")
        return True
        
    except Exception as e:
        print(f"❌ ไม่สามารถเริ่ม backend API: {e}")
        return False

def start_streamlit():
    """เริ่มใช้งาน Streamlit frontend"""
    print("\n🚀 กำลังเริ่มต้น Streamlit frontend...")
    
    streamlit_app = Path(__file__).parent / 'frontend' / 'app_streamlit.py'
    
    if not streamlit_app.exists():
        print(f"❌ ไม่พบไฟล์แอป Streamlit: {streamlit_app}")
        sys.exit(1)
    
    streamlit_command = find_executable('streamlit')
    if ' ' in streamlit_command:  # กรณีที่เป็น "python -m streamlit"
        parts = streamlit_command.split(' ', 2)
        command = [
            parts[0],
            "-m",
            "streamlit",
            "run",
            str(streamlit_app),
            "--server.port=8501",
            "--server.address=127.0.0.1",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
            "--browser.serverAddress=127.0.0.1",
            "--logger.level=info"
        ]
    else:
        command = [
            streamlit_command,
            "run",
            str(streamlit_app),
            "--server.port=8501",
            "--server.address=127.0.0.1",
            "--server.enableCORS=false", 
            "--server.enableXsrfProtection=false",
            "--browser.serverAddress=127.0.0.1",
            "--logger.level=info"
        ]
    
    try:
        process = subprocess.Popen(command, 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0)
        
        # รอให้ Streamlit เริ่มทำงาน
        print("⏳ รอให้ Streamlit เริ่มทำงาน...")
        time.sleep(5)
        
        # ทดลองเปิดเบราวเซอร์
        print("🔗 กำลังเปิดเบราวเซอร์...")
        webbrowser.open("http://127.0.0.1:8501")
        return True
        
    except Exception as e:
        print(f"❌ ไม่สามารถเริ่ม Streamlit: {e}")
        return False

def main():
    """ฟังก์ชันหลัก"""
    print("=" * 50)
    print("      Amulet-AI: เริ่มต้นระบบทั้งหมด      ")
    print("=" * 50)
    
    try:
        # ติดตั้ง psutil หากยังไม่มี
        try:
            import psutil
        except ImportError:
            print("📦 กำลังติดตั้ง psutil...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
            import psutil
        
        # ยุติโปรเซสเก่า
        kill_processes(['python', 'streamlit'])
        
        # เริ่ม backend
        if not start_backend():
            print("❌ ไม่สามารถเริ่ม backend ได้")
            return 1
        
        # เริ่ม frontend
        if not start_streamlit():
            print("❌ ไม่สามารถเริ่ม frontend ได้")
            return 1
        
        print("\n✅ ระบบทั้งหมดเริ่มทำงานเรียบร้อยแล้ว")
        print("📡 Backend API: http://127.0.0.1:8000")
        print("🖥️ Frontend: http://127.0.0.1:8501")
        print("\n👉 กรุณารอสักครู่เพื่อให้ระบบทำงานสมบูรณ์")
        print("⚠️ หากไม่เปิดเบราวเซอร์อัตโนมัติ ให้เปิด http://127.0.0.1:8501 เอง")
        
        # รอให้ผู้ใช้กด Ctrl+C
        print("\n👋 กด Ctrl+C เพื่อปิดระบบทั้งหมด")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 กำลังปิดระบบ...")
        kill_processes(['python', 'streamlit'])
        print("👋 ปิดระบบเรียบร้อยแล้ว")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
