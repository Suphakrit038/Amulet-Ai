@echo off
echo ===============================================
echo     🏺 Amulet-AI Quick Setup & Launch System
echo ===============================================
echo.
echo [STEP 1] Installing required packages...
echo.

REM Install core dependencies
echo Installing FastAPI, Streamlit and Uvicorn...
python -m pip install --upgrade pip
python -m pip install fastapi==0.115.0 uvicorn[standard]==0.32.0 streamlit==1.40.0 python-multipart==0.0.20

REM Install core image processing
echo Installing image processing libraries...
python -m pip install pillow==10.4.0 opencv-python-headless==4.10.0.84 numpy==1.26.4

REM Install data processing  
echo Installing data processing libraries...
python -m pip install pandas==2.2.3 requests==2.32.3 pydantic==2.10.0

REM Install visualization
echo Installing visualization libraries...
python -m pip install matplotlib==3.9.3 plotly==5.24.1

echo.
echo [STEP 2] Testing installation...
python -c "import fastapi, uvicorn, streamlit, PIL, numpy, pandas, requests; print('✅ All core modules installed successfully!')"

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Installation failed. Please check the errors above.
    pause
    exit /b 1
)

echo.
echo [STEP 3] Starting Amulet-AI System...
echo.

REM Kill any existing processes
echo Stopping any running processes...
taskkill /F /IM python.exe 2>NUL
taskkill /F /IM streamlit.exe 2>NUL
timeout /t 2 >NUL

REM Start backend API
echo [Backend] Starting API server...
start "Amulet-AI Backend" cmd /c "python backend\mock_api.py"
timeout /t 5 >NUL

REM Test API connection
echo [Backend] Testing API connection...
python -c "import requests; r=requests.get('http://127.0.0.1:8000/health', timeout=10); print('✅ Backend ready!' if r.status_code==200 else '❌ Backend failed')" 2>NUL

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Backend API failed to start. Trying alternative...
    start "Amulet-AI Backend" cmd /c "python -m uvicorn backend.mock_api:app --host 127.0.0.1 --port 8000"
    timeout /t 8 >NUL
)

REM Start frontend
echo [Frontend] Starting Streamlit web app...
start "Amulet-AI Frontend" cmd /c "streamlit run frontend\app_streamlit.py --server.port 8501 --server.address 127.0.0.1"
timeout /t 8 >NUL

REM Open browser
echo [Browser] Opening web interface...
timeout /t 3 >NUL
start http://127.0.0.1:8501

echo.
echo ===============================================
echo     🎉 Amulet-AI System is Ready!
echo ===============================================
echo.
echo 🔗 Web Interface: http://127.0.0.1:8501
echo 🔗 API Backend:   http://127.0.0.1:8000
echo 📚 API Docs:      http://127.0.0.1:8000/docs
echo.
echo ⚠️  Keep this window open while using the system
echo    Press any key to close all services
echo.
pause

REM Cleanup
echo Stopping all services...
taskkill /F /IM python.exe 2>NUL
taskkill /F /IM streamlit.exe 2>NUL
echo ✅ System stopped successfully
