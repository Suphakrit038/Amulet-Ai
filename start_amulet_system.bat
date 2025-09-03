@echo off
echo Starting Amulet-AI System...
echo.

:: Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo Warning: Virtual environment not found
)

:: Run the starter script
python start_amulet_system.py

:: Pause at the end if there's an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred. Check the output above.
    pause
)
