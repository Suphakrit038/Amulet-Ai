@echo off
echo ===============================
echo  เริ่มระบบ Amulet AI ทั้งหมด
echo ===============================
echo.

echo [1] กำลังหยุดโปรเซสเก่าที่อาจมีปัญหา...
taskkill /F /IM python.exe 2>NUL
taskkill /F /IM streamlit.exe 2>NUL
timeout /t 2 > NUL
echo.

echo [2] กำลังเริ่ม Backend API...
start "Amulet-AI Backend" cmd /c ".\.venv\Scripts\python.exe backend\mock_api.py"
timeout /t 5 > NUL

echo [3] กำลังทดสอบการเชื่อมต่อ API...
.\.venv\Scripts\python.exe -c "import sys, requests; print('API พร้อมใช้งาน!' if requests.get('http://127.0.0.1:8000/health', timeout=5).status_code == 200 else 'ไม่สามารถเชื่อมต่อ API ได้'); sys.exit(0 if requests.get('http://127.0.0.1:8000/health', timeout=5).status_code == 200 else 1)"
if %ERRORLEVEL% NEQ 0 (
    echo [!] ไม่สามารถเชื่อมต่อกับ API ได้ กรุณาตรวจสอบ backend
    pause
    exit /b 1
)
echo.

echo [4] กำลังเริ่ม Frontend (Streamlit)...
start "Amulet-AI Frontend" cmd /c ".\.venv\Scripts\streamlit.exe run frontend\app_streamlit.py --server.port=8501 --server.address=127.0.0.1 --server.enableCORS=false --browser.serverAddress=127.0.0.1"
timeout /t 5 > NUL

echo [5] กำลังเปิดเว็บเบราว์เซอร์...
start http://127.0.0.1:8501
echo.

echo =============================
echo ระบบทั้งหมดเริ่มทำงานเรียบร้อยแล้ว
echo =============================
echo.
echo Backend API: http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:8501
echo.
echo กรุณาอย่าปิดหน้าต่างนี้ขณะใช้งานระบบ
echo หากต้องการปิดระบบ ให้กด Ctrl+C หรือปิดหน้าต่างนี้
echo.
pause
