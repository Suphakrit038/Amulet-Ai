#!/usr/bin/env python
"""
ทำความสะอาดโปรเจค Amulet-AI โดยการย้ายไฟล์ที่ไม่จำเป็นไปยังโฟลเดอร์ backup
"""

import os
import shutil
from pathlib import Path

def main():
    """ย้ายไฟล์ที่ไม่จำเป็นไปยังโฟลเดอร์ backup"""
    # รายการไฟล์ที่ไม่จำเป็น
    files_to_backup = [
        # สคริปต์ที่ไม่ได้ใช้งาน
        'advanced_multi_class_trainer.py',
        'advanced_training.log',
        'clean_storage.py',
        'evaluate_and_connect.py',
        'image_comparison.log',
        'image_comparison.py',
        'inspect_dataset.py',
        'run_app.bat',
        'run_comparison_app.bat',
        'run_streamlit_app.py',
        'run_amulet_system.py',
        'cleanup.py',
        'cleanup.bat',
        
        # README ที่ซ้ำซ้อน
        'README_UPDATED.md',
        'README_IMPROVED.md',
        'README_CLEANUP.md',
        
        # ไฟล์ Config ที่ซ้ำซ้อน
        'config_simplified.json',
    ]
    
    # สร้างโฟลเดอร์ backup ถ้ายังไม่มี
    backup_dir = os.path.join(os.getcwd(), 'backup')
    os.makedirs(backup_dir, exist_ok=True)
    
    # ย้ายไฟล์
    for filename in files_to_backup:
        src_path = os.path.join(os.getcwd(), filename)
        dst_path = os.path.join(backup_dir, filename)
        
        if os.path.exists(src_path):
            try:
                shutil.move(src_path, dst_path)
                print(f"✓ ย้าย {filename} ไปยัง backup/ สำเร็จ")
            except Exception as e:
                print(f"✗ ไม่สามารถย้าย {filename}: {e}")
        else:
            print(f"! ไม่พบไฟล์ {filename}")
    
    # ทำความสะอาด __pycache__
    for root, dirs, files in os.walk(os.getcwd()):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(pycache_path)
                    print(f"✓ ลบ {pycache_path} สำเร็จ")
                except Exception as e:
                    print(f"✗ ไม่สามารถลบ {pycache_path}: {e}")
    
    print("\n✅ ทำความสะอาดโปรเจคเรียบร้อยแล้ว")
    print("📋 ไฟล์หลักที่ควรเก็บไว้:")
    print("   - run_system.py")
    print("   - config.json")
    print("   - requirements.txt")
    print("   - README.md")
    print("   - โฟลเดอร์: frontend/, backend/, ai_models/, data_base/")

if __name__ == "__main__":
    print("เริ่มการทำความสะอาดโปรเจค...")
    main()
    print("\nสิ้นสุดการทำงาน")
