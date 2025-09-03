@echo off
echo =======================================
echo     รันแอพพลิเคชัน Streamlit แบบเร็ว
echo =======================================

:: ตั้งค่าตัวแปรสำหรับแอปพลิเคชัน
set APP_PATH=frontend/app_streamlit.py
set PORT=8501

:: ตรวจสอบว่ามี virtual environment หรือไม่
if exist .venv\Scripts\activate.bat (
    echo เปิดใช้งาน virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo ไม่พบ virtual environment
    echo กำลังใช้ Python จากระบบ
)

:: รัน Streamlit แบบเร็ว
echo กำลังเริ่มต้น Streamlit ที่พอร์ต %PORT%...
streamlit run %APP_PATH% --server.port=%PORT% --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --server.maxUploadSize=10

:: หากมีข้อผิดพลาด
if %ERRORLEVEL% NEQ 0 (
    echo เกิดข้อผิดพลาดในการรัน Streamlit
    echo ลองใช้คำสั่ง: .\.venv\Scripts\streamlit.exe run %APP_PATH% --server.port=%PORT%
    pause
)
