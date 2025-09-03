#!/usr/bin/env python3
"""
Amulet-AI File Organizer Script
จัดระเบียบไฟล์ในระบบให้เป็นระเบียบมากขึ้น
"""

import os
import shutil
import sys
from pathlib import Path


def ensure_directory(dir_path):
    """สร้างโฟลเดอร์ถ้ายังไม่มี"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✅ ตรวจสอบโฟลเดอร์ {dir_path}")


def move_file(src, dest):
    """ย้ายไฟล์จาก src ไป dest"""
    try:
        if not os.path.exists(src):
            print(f"❌ ไม่พบไฟล์ {src}")
            return False
        
        if os.path.exists(dest):
            print(f"⚠️ ไฟล์ {dest} มีอยู่แล้ว - ข้าม")
            return False
        
        shutil.move(src, dest)
        print(f"✅ ย้าย {src} ไปยัง {dest}")
        return True
    except Exception as e:
        print(f"❌ ไม่สามารถย้าย {src}: {e}")
        return False


def main():
    """ฟังก์ชันหลัก"""
    print("=" * 60)
    print("🧹 เริ่มจัดระเบียบไฟล์ในระบบ Amulet-AI")
    print("=" * 60)
    
    base_dir = Path.cwd()
    print(f"📂 โฟลเดอร์หลัก: {base_dir}")
    
    # สร้างโฟลเดอร์ที่จำเป็น
    archive_dir = base_dir / "archive"
    ensure_directory(archive_dir)
    ensure_directory(archive_dir / "launchers")
    ensure_directory(archive_dir / "tests")
    ensure_directory(archive_dir / "configs")
    ensure_directory(archive_dir / "scripts")
    ensure_directory(archive_dir / "batches")
    
    # รายการไฟล์ที่จะย้ายไปยังโฟลเดอร์ launchers
    launcher_files = [
        "launch_system.py", 
        "launch_complete_system.py", 
        "launch_real_ai_system.py", 
        "simple_launcher.py", 
        "start_amulet_system.py", 
        "quick_start.py",
        "start.py",
        "system_launcher.py"
    ]
    
    # รายการไฟล์ที่จะย้ายไปยังโฟลเดอร์ tests
    test_files = [
        "test_api_connection.py", 
        "test_api_import.py", 
        "test_real_ai_system.py", 
        "test_simple.py", 
        "test_system.py", 
        "final_test.py", 
        "quick_test.py"
    ]
    
    # รายการไฟล์ที่จะย้ายไปยังโฟลเดอร์ scripts
    script_files = [
        "cleanup_temp.py", 
        "fix_and_run_system.py", 
        "run_streamlit_fast.py", 
        "setup_complete_system.py", 
        "emergency_fix.py"
    ]
    
    # รายการไฟล์ที่จะย้ายไปยังโฟลเดอร์ configs
    config_files = [
        "requirements_fixed.txt", 
        "README_NEW.md"
    ]
    
    # รายการไฟล์ batch ที่จะย้ายไปยังโฟลเดอร์ batches
    batch_files = [
        "launch.bat", 
        "launch_real_ai_backend.bat", 
        "launch_real_ai_complete.bat", 
        "fix_streamlit.bat", 
        "run_streamlit_fast.bat", 
        "start_system.bat",
        "start_amulet_system.bat"
    ]
    
    # ย้ายไฟล์ launcher
    print("\n🚀 กำลังย้ายไฟล์ launcher...")
    for file in launcher_files:
        src = base_dir / file
        dest = archive_dir / "launchers" / file
        move_file(src, dest)
    
    # ย้ายไฟล์ test
    print("\n🧪 กำลังย้ายไฟล์ test...")
    for file in test_files:
        src = base_dir / file
        dest = archive_dir / "tests" / file
        move_file(src, dest)
    
    # ย้ายไฟล์ script
    print("\n📜 กำลังย้ายไฟล์ script...")
    for file in script_files:
        src = base_dir / file
        dest = archive_dir / "scripts" / file
        move_file(src, dest)
    
    # ย้ายไฟล์ config
    print("\n⚙️ กำลังย้ายไฟล์ config...")
    for file in config_files:
        src = base_dir / file
        dest = archive_dir / "configs" / file
        move_file(src, dest)
    
    # ย้ายไฟล์ batch
    print("\n💻 กำลังย้ายไฟล์ batch...")
    for file in batch_files:
        src = base_dir / file
        dest = archive_dir / "batches" / file
        move_file(src, dest)
    
    # ตรวจสอบไฟล์ที่เหลือในรูท
    print("\n📋 ไฟล์ที่เหลือในรูทหลังจากการจัดระเบียบ:")
    root_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    for file in sorted(root_files):
        print(f"  - {file}")
    
    print("\n✅ เสร็จสิ้นการจัดระเบียบไฟล์")
    print("=" * 60)


if __name__ == "__main__":
    main()
