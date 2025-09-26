@echo off
echo.
echo 🔮 ================================ 🔮
echo       AMULET-AI v2.0 SYSTEM
echo     Production-Ready Launcher
echo 🔮 ================================ 🔮
echo.

cd /d "%~dp0"

echo 🚀 Starting Amulet-AI System...
echo.

echo 📊 Checking dataset...
if exist "dataset_optimized" (
    echo    ✅ Optimized dataset found
) else (
    echo    🔄 Creating optimized dataset...
    .venv\Scripts\python.exe tools\optimize_dataset.py
)

echo.
echo 🤖 Checking AI model...
if exist "trained_model_optimized" (
    echo    ✅ Trained model found
) else (
    echo    🔄 Training AI model...
    .venv\Scripts\python.exe ai_models\optimized_model.py
)

echo.
echo 🌐 Starting API Server...
echo    📝 API URL: http://localhost:8000
echo    📚 API Docs: http://localhost:8000/docs
echo.

start "Amulet-AI API" cmd /k ".venv\Scripts\python.exe backend\api\production_ready_api.py"

timeout /t 5 /nobreak >nul

echo 🎨 Starting Frontend...
echo    📝 Frontend URL: http://localhost:8501
echo.

start "Amulet-AI Frontend" cmd /k ".venv\Scripts\python.exe -m streamlit run frontend\production_app.py --server.port 8501"

echo.
echo ✅ System started successfully!
echo.
echo 🌐 Access points:
echo    - Frontend: http://localhost:8501
echo    - API: http://localhost:8000
echo    - API Docs: http://localhost:8000/docs
echo.
echo ⏹️  Close the opened windows to stop the system
pause