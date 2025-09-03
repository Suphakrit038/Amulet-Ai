@echo off
chcp 65001 >nul

echo ========================================
echo 🏺 เริ่มระบบ Amulet-AI แบบ Real Model
echo ========================================
echo.

echo 🔄 ตรวจสอบ Python environment...
python --version

echo.
echo 📦 ติดตั้ง dependencies ถ้าจำเป็น...
pip install -q -r requirements.txt

echo.
echo ========================================
echo 🚀 เริ่มต้น Real AI Backend (Port 8001)
echo ========================================
echo.

cd backend
echo ✅ เปลี่ยนไดเรกทอรี่ไป backend/
echo 🔥 เริ่ม Real AI Model API Server...

python api_with_real_model.py

pause
