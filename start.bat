@echo off
echo.
echo 🔮 ================================ 🔮
echo       AMULET-AI - Thai Buddhist
echo        Amulet Recognition System
echo 🔮 ================================ 🔮
echo.

cd /d "%~dp0"

echo 🚀 Starting Amulet-AI System...
echo.

echo 📊 Checking dataset...
if exist "dataset" (
    echo    ✅ Dataset found (3 classes)
) else (
    echo    ❌ Dataset not found
    pause
    exit /b 1
)

echo.
echo 🤖 Checking AI model...
if exist "trained_model" (
    echo    ✅ Trained model found
) else (
    echo    ❌ Trained model not found
    pause
    exit /b 1
)

echo.
echo 🌐 Starting API Server...
echo    📝 API URL: http://localhost:8000
echo    📚 API Docs: http://localhost:8000/docs
echo.

start "Amulet-AI API" cmd /k "python backend\api\main_api.py"

timeout /t 5 /nobreak >nul

echo 🎨 Starting Frontend...
echo    📝 Frontend URL: http://localhost:8501
echo.

start "Amulet-AI Frontend" cmd /k "python -m streamlit run frontend\production_app.py --server.port 8501"

echo.
echo ✅ System started successfully!
echo.
echo 🌐 Access points:
echo    - Frontend: http://localhost:8501
echo    - API: http://localhost:8000
echo.
echo ⏹️  Close the opened windows to stop the system
pause
