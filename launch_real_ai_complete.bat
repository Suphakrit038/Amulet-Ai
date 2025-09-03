@echo off
chcp 65001 >nul

echo ========================================
echo 🏺 เริ่มระบบ Amulet-AI แบบ Complete 
echo    (Real AI Backend + Streamlit Frontend)
echo ========================================
echo.

echo 🔄 ตรวจสอบ Python environment...
python --version

echo.
echo 📦 ติดตั้ง dependencies...
pip install -q -r requirements.txt

echo.
echo ========================================
echo 🚀 เริ่มต้น Real AI Complete System
echo ========================================
echo.

echo 🔥 เริ่ม Real AI Backend (Port 8001) ในพื้นหลัง...
cd backend
start "Amulet-AI Real Backend" python api_with_real_model.py

echo ⏳ รอ backend โหลดเสร็จ...
timeout /t 10 /nobreak >nul

cd ..
echo.
echo 🎨 เริ่ม Streamlit Frontend (จะเปิดในเบราว์เซอร์)...
streamlit run frontend/app_streamlit.py --server.port 8501

pause
