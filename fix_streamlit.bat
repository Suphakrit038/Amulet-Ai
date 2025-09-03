@echo off
echo ============================
echo  แก้ปัญหา Streamlit ไม่ทำงาน
echo ============================

echo [1] กำลังยุติโปรเซสที่อาจมีปัญหา...
taskkill /F /IM streamlit.exe 2>NUL
taskkill /F /IM python.exe 2>NUL
timeout /t 2 > NUL

echo [2] กำลังรัน Streamlit แบบแก้ปัญหา...
echo.

:: เรียกใช้ streamlit โดยตรง พร้อมค่าที่เหมาะสม
.\.venv\Scripts\streamlit.exe run frontend/app_streamlit.py --server.port=8501 --server.address=127.0.0.1 --server.enableCORS=false --server.enableXsrfProtection=false --browser.serverAddress=127.0.0.1 --server.runOnSave=true --client.showErrorDetails=true --logger.level=info

:: หากไม่สำเร็จ จะไปที่นี่
echo.
echo [!] มีปัญหาในการรัน Streamlit กำลังลองวิธีที่ 2...
echo.

:: วิธีที่ 2 ใช้ python เรียก streamlit module
.\.venv\Scripts\python.exe -m streamlit run frontend/app_streamlit.py --server.port=8501 --server.address=127.0.0.1

:: หากยังไม่สำเร็จอีก
echo.
echo [!] ยังไม่สำเร็จ กรุณาลองทำขั้นตอนต่อไปนี้ด้วยตนเอง:
echo 1. รันคำสั่ง: python -m pip install --upgrade streamlit
echo 2. ทดสอบรันโดยตรงด้วย: streamlit hello
echo.
pause
