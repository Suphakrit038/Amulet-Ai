@echo off
echo.
echo ==========================================
echo    📱 Amulet-AI Frontend Interface
echo ==========================================
echo.

echo 🌟 Features Available:
echo    - Advanced Image Analysis
echo    - Smart Amulet Classification  
echo    - Real-time Price Estimation
echo    - Market Insights
echo    - Beautiful User Interface
echo.

echo 🚀 Starting Frontend Application...
echo    URL: http://localhost:8501
echo.
echo ⏳ Please wait for browser to open...

cd /d "%~dp0" 
python -m streamlit run frontend/app_streamlit.py

pause
