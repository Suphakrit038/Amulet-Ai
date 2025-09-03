@echo off
echo ===============================================================
echo  Amulet-AI - เตรียมข้อมูลฝึกสอน
echo ===============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or later.
    goto end
)

:menu
echo Choose an option:
echo.
echo  1. Check dataset structure
echo  2. Reorganize dataset (will keep original files)
echo  3. Reorganize and remove empty Thai folders
echo  4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Checking dataset structure...
    python tools\prepare_dataset.py --check-only
    echo.
    goto menu
)

if "%choice%"=="2" (
    echo.
    echo Reorganizing dataset...
    python tools\prepare_dataset.py
    echo.
    goto menu
)

if "%choice%"=="3" (
    echo.
    echo Reorganizing dataset and removing empty Thai folders...
    python tools\prepare_dataset.py --remove-empty
    echo.
    goto menu
)

if "%choice%"=="4" (
    goto end
) else (
    echo.
    echo Invalid choice. Please try again.
    echo.
    goto menu
)

:end
echo.
echo ===============================================================
echo Thank you for using Amulet-AI Dataset Preparation Tool
echo ===============================================================
pause
