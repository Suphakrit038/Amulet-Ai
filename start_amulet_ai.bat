@echo off
echo =====================================
echo    🔮 Amulet-AI System Launcher
echo =====================================
echo ระบบวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์
echo.

REM ตรวจสอบ Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python ไม่ถูกติดตั้งหรือไม่อยู่ใน PATH
    echo กรุณาติดตั้ง Python 3.8+ จาก https://python.org
    pause
    exit /b 1
)

echo ✅ พบ Python
python --version

REM ตรวจสอบ virtual environment
if exist ".venv\Scripts\activate" (
    echo ✅ พบ virtual environment
    call .venv\Scripts\activate
) else (
    echo ⚠️ ไม่พบ virtual environment
    echo สร้าง virtual environment ใหม่...
    python -m venv .venv
    call .venv\Scripts\activate
)

REM ติดตั้ง dependencies พื้นฐาน
echo 📦 ติดตั้ง dependencies...
pip install streamlit requests pillow numpy

REM ตรวจสอบไฟล์ frontend (สไตล์เก่าก่อน)
if exist "frontend\app_old_style.py" (
    set "FRONTEND_APP=frontend\app_old_style.py"
    echo 🎨 ใช้ Old Style UI (สไตล์เก่าที่สวยงาม)
) else if exist "frontend\app_streamlit.py" (
    set "FRONTEND_APP=frontend\app_streamlit.py"
    echo 🎨 ใช้ Standard UI (สไตล์เก่าต้นฉบับ)
) else if exist "frontend\app_simple.py" (
    set "FRONTEND_APP=frontend\app_simple.py"
    echo 🎨 ใช้ Simple UI (สไตล์ใหม่)
) else if exist "frontend\app_modern.py" (
    set "FRONTEND_APP=frontend\app_modern.py"
    echo 🎨 ใช้ Modern UI
) else (
    echo ❌ ไม่พบไฟล์ frontend
    pause
    exit /b 1
)

REM สร้างโฟลเดอร์จำเป็น
if not exist "logs" mkdir logs
if not exist "uploads" mkdir uploads
if not exist "backend\logs" mkdir backend\logs

echo.
echo 🚀 เริ่มระบบ Amulet-AI...
echo 🌐 เบราว์เซอร์จะเปิดอัตโนมัติ
echo 🛑 กด Ctrl+C เพื่อหยุดระบบ
echo.

REM เริ่ม Streamlit
streamlit run "%FRONTEND_APP%" --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false

pause
