@echo off
echo ===============================================================
echo  Amulet-AI - จัดระเบียบข้อมูลฝึกสอน
echo ===============================================================
echo.

REM ตรวจสอบว่า Python มีหรือไม่
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ผิดพลาด] ไม่พบ Python กรุณาติดตั้ง Python 3.6 หรือใหม่กว่า
    goto :end
)

REM ตรวจสอบว่าสคริปต์ reorganize_dataset.py มีหรือไม่
if not exist tools\reorganize_dataset.py (
    echo [ผิดพลาด] ไม่พบไฟล์ tools\reorganize_dataset.py
    goto :end
)

echo เริ่มจัดระเบียบข้อมูลฝึกสอน...

REM ทำ dry run ก่อนเพื่อแสดงการเปลี่ยนแปลงที่จะเกิดขึ้น
echo.
echo ===== ตรวจสอบการเปลี่ยนแปลงที่จะเกิดขึ้น (dry run) =====
python tools\reorganize_dataset.py --dry-run

echo.
set /p confirmation=ต้องการดำเนินการต่อหรือไม่? (y/n): 

if /i "%confirmation%" neq "y" goto :canceled

echo.
echo ===== กำลังจัดระเบียบข้อมูลฝึกสอน =====
python tools\reorganize_dataset.py

if %ERRORLEVEL% neq 0 (
    echo [ผิดพลาด] เกิดข้อผิดพลาดในการจัดระเบียบข้อมูลฝึกสอน
) else (
    echo.
    echo [สำเร็จ] จัดระเบียบข้อมูลฝึกสอนเรียบร้อยแล้ว
)

goto :end

:canceled
echo.
echo ยกเลิกการจัดระเบียบข้อมูลฝึกสอน

:end
echo.
echo ===============================================================
pause
