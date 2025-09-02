@echo off
echo 🗑️ Advanced Cleanup Script
echo ========================

REM ลบไฟล์ที่ไม่จำเป็น
echo Deleting unnecessary files...

del /F /Q analyze_dataset.py 2>nul
del /F /Q app.py 2>nul
del /F /Q check_data_models.py 2>nul
del /F /Q cleanup_files.bat 2>nul
del /F /Q complete_organizer.py 2>nul
del /F /Q config.json 2>nul
del /F /Q dataset_inspector.py 2>nul
del /F /Q dataset_organizer.py 2>nul
del /F /Q debug_copy.py 2>nul
del /F /Q organize_dataset.bat 2>nul
del /F /Q organize_dataset.ps1 2>nul
del /F /Q organize_dataset.py 2>nul
del /F /Q organize_karaoke.bat 2>nul
del /F /Q organize_step1.py 2>nul
del /F /Q quick_dataset_stats.py 2>nul
del /F /Q rename_dataset_files.py 2>nul
del /F /Q requirements.txt 2>nul
del /F /Q simple_copy.py 2>nul
del /F /Q simple_organizer.py 2>nul
del /F /Q test_copy.bat 2>nul
del /F /Q test_copy.py 2>nul

REM ลบไฟล์ .md ที่ไม่จำเป็น
del /F /Q BUGFIXES_SUMMARY.md 2>nul
del /F /Q CLEANUP_REPORT.md 2>nul
del /F /Q COMPLETE_DATASET_INSPECTION.md 2>nul
del /F /Q DATASET_INSPECTION_REPORT.md 2>nul
del /F /Q DATASET_ORGANIZATION_GUIDE.md 2>nul
del /F /Q DATASET_ORGANIZATION_STATUS.md 2>nul
del /F /Q DATA_MODEL_ANALYSIS_REPORT.md 2>nul
del /F /Q KARAOKE_DATASET_ORGANIZATION.md 2>nul
del /F /Q PROJECT_STRUCTURE.md 2>nul

REM ลบโฟลเดอร์ที่ไม่จำเป็น
rmdir /S /Q data-processing 2>nul
rmdir /S /Q dev-tools 2>nul
rmdir /S /Q logs 2>nul
rmdir /S /Q .pytest_cache 2>nul

REM ลบ __pycache__ directories
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul

REM ลบไฟล์ใน backend ที่ไม่จำเป็น
del /F /Q backend\test_api.py 2>nul
del /F /Q backend\api_simple.py 2>nul
del /F /Q backend\minimal_api.py 2>nul
del /F /Q backend\optimized_*.py 2>nul

REM ลบไฟล์ใน frontend ที่ไม่จำเป็น
del /F /Q frontend\app_straemlit.py 2>nul
del /F /Q frontend\master_setup.py 2>nul
del /F /Q frontend\master_setup_*.log 2>nul
rmdir /S /Q "frontend\not use" 2>nul

REM ลบไฟล์ในโฟลเดอร์อื่นๆ
del /F /Q dataset_organized\*.json 2>nul

echo ✅ Cleanup completed!
echo.
echo 📁 Remaining structure:
echo   ├── ai_models/ (Advanced AI System)
echo   ├── backend/ (Cleaned API)
echo   ├── frontend/ (Cleaned UI)
echo   ├── dataset/ (Your images)
echo   ├── docs/ (Documentation)
echo   ├── tests/ (Test files)
echo   ├── utils/ (Utilities)
echo   └── README.md
echo.
echo 🚀 Ready to use Advanced Amulet AI system!
pause
