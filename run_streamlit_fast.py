#!/usr/bin/env python
"""
สคริปต์รันแอพพลิเคชัน Streamlit แบบเร็ว
มีการกำหนดค่าที่เหมาะสมเพื่อให้แอพรันได้เร็วขึ้น
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def find_streamlit_executable():
    """ค้นหา executable ของ streamlit ใน virtual environment"""
    base_dir = Path(__file__).parent
    venv_dir = base_dir / '.venv'
    
    if sys.platform == 'win32':
        streamlit_path = venv_dir / 'Scripts' / 'streamlit.exe'
        python_path = venv_dir / 'Scripts' / 'python.exe'
    else:
        streamlit_path = venv_dir / 'bin' / 'streamlit'
        python_path = venv_dir / 'bin' / 'python'
    
    if streamlit_path.exists():
        return str(streamlit_path)
    elif python_path.exists():
        return f"{python_path} -m streamlit"
    else:
        # หาก virtual environment ไม่มี streamlit ให้ใช้จาก system
        return "streamlit"

def main():
    """ฟังก์ชันหลักสำหรับรัน Streamlit แบบเร็ว"""
    # ตั้งค่า
    app_path = "frontend/app_streamlit.py"
    port = 8501
    
    # ค้นหา streamlit executable
    streamlit_exe = find_streamlit_executable()
    
    # สร้าง command
    if ' ' in streamlit_exe:
        # กรณีที่ต้องใช้ python -m streamlit
        cmd_parts = streamlit_exe.split(' ')
        cmd = cmd_parts + [
            "run", app_path,
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--browser.serverAddress", "localhost",
            "--server.maxUploadSize", "10"
        ]
    else:
        # กรณีใช้ streamlit โดยตรง
        cmd = [
            streamlit_exe, "run", app_path,
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--browser.serverAddress", "localhost",
            "--server.maxUploadSize", "10"
        ]
    
    # แสดงข้อความเริ่มต้น
    print("=" * 50)
    print("    รันแอพพลิเคชัน Streamlit แบบเร็ว")
    print("=" * 50)
    print(f"กำลังเริ่มต้น Streamlit ที่พอร์ต {port}...")
    
    try:
        # รัน streamlit
        process = subprocess.Popen(cmd)
        
        # รอให้เซิร์ฟเวอร์เริ่มต้น
        time.sleep(2)
        
        # เปิดเบราว์เซอร์
        webbrowser.open(f"http://localhost:{port}")
        
        print(f"เปิด Streamlit แล้วที่ http://localhost:{port}")
        print("กด Ctrl+C เพื่อหยุดการทำงาน")
        
        # รอให้โปรเซสทำงานจนเสร็จ
        process.wait()
        
    except KeyboardInterrupt:
        print("\nได้รับคำสั่งให้หยุดทำงาน กำลังปิด Streamlit...")
        process.terminate()
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
