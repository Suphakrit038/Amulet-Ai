#!/usr/bin/env python3
"""
Amulet-AI Cleanup Script
ลบไฟล์ที่ไม่จำเป็นและรวมไฟล์ซ้ำซ้อน
"""

import os
import shutil
import sys
from pathlib import Path

def main():
    """ฟังก์ชันหลัก"""
    print("=" * 60)
    print("🧹 Amulet-AI Cleanup Script")
    print("=" * 60)
    
    # ไฟล์ที่จำเป็นและควรเก็บไว้
    essential_files = [
        "amulet_launcher.py",
        "amulet_launcher.bat",
        "setup_models.py",
        "README.md",
        "requirements.txt",
        "config.json",
        "cleanup.py"
    ]
    
    # โฟลเดอร์ที่จำเป็นและควรเก็บไว้
    essential_folders = [
        ".git",
        ".venv",
        "ai_models",
        "backend",
        "frontend",
        "utils",
        "docs",
        "logs",
        "tests",
        "archive",
        "dataset",
        "dataset_organized",
        "dataset_split",
        "training_output"
    ]
    
    base_dir = Path.cwd()
    archive_dir = base_dir / "archive" / "old_files"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # รวบรวมไฟล์ที่ไม่จำเป็น
    unnecessary_files = []
    for item in base_dir.iterdir():
        if item.is_file() and item.name not in essential_files:
            unnecessary_files.append(item)
    
    if not unnecessary_files:
        print("✅ ไม่มีไฟล์ที่ไม่จำเป็นในโฟลเดอร์หลัก")
        return
    
    print(f"พบไฟล์ที่ไม่จำเป็น {len(unnecessary_files)} ไฟล์:")
    for file in unnecessary_files:
        print(f"  - {file.name}")
    
    choice = input("\nคุณต้องการจัดการไฟล์เหล่านี้อย่างไร?\n"
                  "1) ย้ายไปที่ archive/old_files\n"
                  "2) ลบทิ้ง\n"
                  "3) ยกเลิก\n"
                  "เลือก (1/2/3): ")
    
    if choice == "1":
        # ย้ายไฟล์ไปที่ archive
        for file in unnecessary_files:
            try:
                dest = archive_dir / file.name
                # ถ้ามีไฟล์ชื่อซ้ำใน archive
                if dest.exists():
                    dest = archive_dir / f"{file.stem}_old{file.suffix}"
                
                shutil.move(str(file), str(dest))
                print(f"✅ ย้าย {file.name} ไปที่ {dest.relative_to(base_dir)}")
            except Exception as e:
                print(f"❌ ไม่สามารถย้าย {file.name}: {e}")
        
        print("\n✅ ย้ายไฟล์ทั้งหมดเรียบร้อยแล้ว")
    
    elif choice == "2":
        # ลบไฟล์
        confirm = input("⚠️ คุณแน่ใจหรือไม่ที่จะลบไฟล์เหล่านี้? (y/n): ")
        if confirm.lower() == "y":
            for file in unnecessary_files:
                try:
                    file.unlink()
                    print(f"✅ ลบ {file.name}")
                except Exception as e:
                    print(f"❌ ไม่สามารถลบ {file.name}: {e}")
            
            print("\n✅ ลบไฟล์ทั้งหมดเรียบร้อยแล้ว")
        else:
            print("ยกเลิกการลบไฟล์")
    
    else:
        print("ยกเลิกการดำเนินการ")


if __name__ == "__main__":
    main()
