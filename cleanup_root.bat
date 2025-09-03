@echo off
echo ===============================================================
echo  Amulet-AI - Clean up root directory
echo ===============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.6 or newer.
    goto :end
)

REM Check if the cleanup script exists
if not exist tools\cleanup_root.py (
    echo [ERROR] Cleanup script not found: tools\cleanup_root.py
    goto :end
)

echo Starting root directory cleanup...

REM Do a dry run first to show what will happen
echo.
echo ===== Checking changes that will be made (dry run) =====
python tools\cleanup_root.py --dry-run

echo.
set /p confirmation=Do you want to proceed? (y/n): 

if /i "%confirmation%" neq "y" goto :canceled

echo.
echo ===== Cleaning up root directory =====
python tools\cleanup_root.py

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Error occurred during cleanup.
) else (
    echo.
    echo [SUCCESS] Root directory cleanup completed successfully.
)

goto :end

:canceled
echo.
echo Root directory cleanup canceled.

:end
echo.
echo ===============================================================
pause
