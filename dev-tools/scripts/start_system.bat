@echo off
echo.
echo ================================================
echo    🚀 Amulet-AI Complete System Launcher
echo ================================================
echo.

echo 🎯 System Configuration:
echo    Backend API:  http://localhost:8000
echo    Frontend UI:  http://localhost:8501
echo    AI Mode:      Advanced Simulation
echo    Status:       Production Ready*
echo.
echo 💡 *Using advanced AI simulation until
echo    training data is ready
echo.

echo 🔄 Starting Backend Server...
start "🤖 Amulet-AI Backend" cmd /k "python -m uvicorn backend.api:app --reload --port 8000"

echo ⏳ Waiting 5 seconds for backend to initialize...
timeout /t 5 /nobreak > nul

echo 🔄 Starting Frontend Interface...  
start "📱 Amulet-AI Frontend" cmd /k "python -m streamlit run frontend/app_streamlit.py"

echo.
echo ✅ System launched successfully!
echo.
echo 📚 Quick Start Guide:
echo    1. Wait for both windows to finish loading
echo    2. Open browser to http://localhost:8501
echo    3. Upload an amulet image to test
echo    4. Check API docs at http://localhost:8000/docs
echo.
echo 🔧 Troubleshooting:
echo    - If backend fails: Check port 8000 is free
echo    - If frontend fails: Check port 8501 is free
echo    - System status: http://localhost:8000/system-status
echo.

echo 🎉 Ready to analyze amulets with AI!
pause
