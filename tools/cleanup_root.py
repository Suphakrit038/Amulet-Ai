#!/usr/bin/env python3
"""
Amulet-AI Root Directory Cleanup Script
จัดระเบียบและทำความสะอาดไฟล์ในไดเรกทอรีหลัก
"""
import os
import shutil
from pathlib import Path
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cleanup_root')

# Define the project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Define directories
TOOLS_DIR = PROJECT_ROOT / "tools"
TESTS_DIR = PROJECT_ROOT / "tests"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIG_DIR = PROJECT_ROOT / "config"
ARCHIVE_DIR = PROJECT_ROOT / "archive" / "root_cleanup"

# Files to move to tools directory
TOOLS_FILES = [
    "maintenance.py",
    "repair_system.py",
    "verify_system.py",
    "organize_files.py",
    "file_access_test.py",
    "comprehensive_file_test.py",
    "cleanup.py",
    "cleanup_files_phase2.py",
]

# Files to move to tests directory
TESTS_FILES = [
    "test_config_manager.py",
    "test_file_operations.py",
    "test_write.txt",
]

# Files to move to scripts directory
SCRIPTS_FILES = [
    "setup_models.py",
    "setup_complete_system.py",
    "amulet_launcher.py",
]

# Files to move to config directory
CONFIG_FILES = [
    # Leave config.json in root for now as it may be referenced directly
]

# Batch files to move to scripts directory
BATCH_FILES = [
    "start.bat",
    "amulet_launcher.bat",
    "organize.bat",
    "organize_folders.bat",
    "initialize_structure.bat",
]

# Files to be archived (not immediately used but kept for reference)
ARCHIVE_FILES = [
    "README_updated.md",
]

def ensure_directory(dir_path):
    """Create directory if it doesn't exist"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {dir_path}")

def move_file(src, dest):
    """Move a file from source to destination"""
    try:
        if not Path(src).exists():
            logger.warning(f"Source file does not exist: {src}")
            return False
        
        # Create parent directories if they don't exist
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if destination file already exists
        if Path(dest).exists():
            logger.warning(f"Destination file already exists: {dest}")
            backup_dest = str(dest) + ".bak"
            logger.info(f"Creating backup: {backup_dest}")
            shutil.copy2(dest, backup_dest)
        
        # Move the file
        shutil.move(src, dest)
        logger.info(f"Moved: {src} -> {dest}")
        return True
    except Exception as e:
        logger.error(f"Error moving file {src} to {dest}: {e}")
        return False

def clean_root_directory(dry_run=False):
    """Clean up the root directory"""
    # Ensure all required directories exist
    ensure_directory(TOOLS_DIR)
    ensure_directory(TESTS_DIR)
    ensure_directory(SCRIPTS_DIR)
    ensure_directory(CONFIG_DIR)
    ensure_directory(ARCHIVE_DIR)
    
    moved_files = 0
    failed_moves = 0
    
    # Move tool files
    for file in TOOLS_FILES:
        src = PROJECT_ROOT / file
        dest = TOOLS_DIR / file
        if dry_run:
            logger.info(f"Would move: {src} -> {dest}")
            moved_files += 1
        else:
            if move_file(src, dest):
                moved_files += 1
            else:
                failed_moves += 1
    
    # Move test files
    for file in TESTS_FILES:
        src = PROJECT_ROOT / file
        dest = TESTS_DIR / file
        if dry_run:
            logger.info(f"Would move: {src} -> {dest}")
            moved_files += 1
        else:
            if move_file(src, dest):
                moved_files += 1
            else:
                failed_moves += 1
    
    # Move script files
    for file in SCRIPTS_FILES:
        src = PROJECT_ROOT / file
        dest = SCRIPTS_DIR / file
        if dry_run:
            logger.info(f"Would move: {src} -> {dest}")
            moved_files += 1
        else:
            if move_file(src, dest):
                moved_files += 1
            else:
                failed_moves += 1
    
    # Move config files
    for file in CONFIG_FILES:
        src = PROJECT_ROOT / file
        dest = CONFIG_DIR / file
        if dry_run:
            logger.info(f"Would move: {src} -> {dest}")
            moved_files += 1
        else:
            if move_file(src, dest):
                moved_files += 1
            else:
                failed_moves += 1
    
    # Move batch files
    for file in BATCH_FILES:
        src = PROJECT_ROOT / file
        dest = SCRIPTS_DIR / file
        if dry_run:
            logger.info(f"Would move: {src} -> {dest}")
            moved_files += 1
        else:
            if move_file(src, dest):
                moved_files += 1
            else:
                failed_moves += 1
    
    # Archive files
    for file in ARCHIVE_FILES:
        src = PROJECT_ROOT / file
        dest = ARCHIVE_DIR / file
        if dry_run:
            logger.info(f"Would archive: {src} -> {dest}")
            moved_files += 1
        else:
            if move_file(src, dest):
                moved_files += 1
            else:
                failed_moves += 1
    
    return moved_files, failed_moves

def main():
    """Main function"""
    print("=" * 60)
    print("🧹 Amulet-AI Root Directory Cleanup")
    print("=" * 60)
    
    # Check for dry run flag
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("Running in dry-run mode (no files will be moved)")
    
    # Clean up the root directory
    moved_files, failed_moves = clean_root_directory(dry_run)
    
    # Print results
    if dry_run:
        print(f"\n✅ Would move {moved_files} files")
    else:
        print(f"\n✅ Moved {moved_files} files")
        if failed_moves > 0:
            print(f"❌ Failed to move {failed_moves} files")
    
    # Create a batch file for easy execution
    batch_file_path = PROJECT_ROOT / "cleanup_root.bat"
    batch_content = """@echo off
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
if not exist tools\\cleanup_root.py (
    echo [ERROR] Cleanup script not found: tools\\cleanup_root.py
    goto :end
)

echo Starting root directory cleanup...

REM Do a dry run first to show what will happen
echo.
echo ===== Checking changes that will be made (dry run) =====
python tools\\cleanup_root.py --dry-run

echo.
set /p confirmation=Do you want to proceed? (y/n): 

if /i "%confirmation%" neq "y" goto :canceled

echo.
echo ===== Cleaning up root directory =====
python tools\\cleanup_root.py

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
"""
    
    if not dry_run:
        with open(batch_file_path, "w") as f:
            f.write(batch_content)
        print(f"\n✅ Created batch file: {batch_file_path}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
