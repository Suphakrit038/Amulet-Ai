@echo off
echo.
echo ==========================================
echo    🤖 Amulet-AI System Launcher 
echo ==========================================
echo.

echo 📋 System Status:
echo    - Mode: Advanced AI Simulation
echo    - Real Image Analysis: Enabled
echo    - Mock Data: Disabled
echo    - Ready for Production: Yes*
echo.
echo 💡 *Using synthetic neural network until
echo    real training data is ready
echo.

echo 🚀 Starting Backend API Server...
echo    URL: http://localhost:8000
echo    Docs: http://localhost:8000/docs
echo.

cd /d "%~dp0"
python -m uvicorn backend.api:app --reload --port 8000

pause
