#!/usr/bin/env python3
"""
Amulet-AI - Setup Models Script
สคริปต์สำหรับดาวน์โหลดและตรวจสอบโมเดล AI
"""

import os
import sys
import requests
import hashlib
import json
from pathlib import Path
from tqdm import tqdm


def check_existing_models():
    """ตรวจสอบโมเดลที่มีอยู่แล้ว"""
    models_dir = Path("ai_models")
    model_files = [
        models_dir / "amulet_model.h5",
        models_dir / "amulet_model.tflite",
        models_dir / "labels.json"
    ]
    
    existing_files = []
    missing_files = []
    
    for model_file in model_files:
        if model_file.exists():
            existing_files.append(model_file)
        else:
            missing_files.append(model_file)
    
    return existing_files, missing_files


def download_file(url, destination, file_description=None):
    """ดาวน์โหลดไฟล์จาก URL พร้อมแสดง progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size from headers
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        desc = file_description if file_description else os.path.basename(destination)
        
        with open(destination, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def verify_file_hash(file_path, expected_hash):
    """ตรวจสอบความถูกต้องของไฟล์โดยใช้ SHA-256"""
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read and update hash in chunks for large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    file_hash = sha256_hash.hexdigest()
    return file_hash == expected_hash


def setup_models():
    """ดาวน์โหลดและตรวจสอบโมเดล AI"""
    print("🔍 ตรวจสอบโมเดล AI ในระบบ...")
    
    existing_files, missing_files = check_existing_models()
    
    if existing_files:
        print(f"✅ พบโมเดลที่มีอยู่แล้ว ({len(existing_files)}):")
        for file in existing_files:
            print(f"  - {file}")
    
    if not missing_files:
        print("✅ มีโมเดลครบถ้วนแล้ว ไม่จำเป็นต้องดาวน์โหลดเพิ่ม")
        return True
    
    print(f"⚠️ ต้องดาวน์โหลดโมเดลเพิ่ม ({len(missing_files)}):")
    for file in missing_files:
        print(f"  - {file}")
    
    # URL สำหรับดาวน์โหลดโมเดล (ตัวอย่าง - ต้องแก้ไขให้เป็น URL จริง)
    model_urls = {
        "amulet_model.h5": "https://example.com/models/amulet_model.h5",
        "amulet_model.tflite": "https://example.com/models/amulet_model.tflite",
        "labels.json": "https://example.com/models/labels.json"
    }
    
    model_hashes = {
        "amulet_model.h5": "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "amulet_model.tflite": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "labels.json": "90abcdef1234567890abcdef1234567890abcdef1234567890abcdef12345678"
    }
    
    # สร้างโฟลเดอร์ ai_models ถ้ายังไม่มี
    models_dir = Path("ai_models")
    models_dir.mkdir(exist_ok=True)
    
    # ดาวน์โหลดโมเดลที่ขาด
    success = True
    for file in missing_files:
        file_name = file.name
        if file_name in model_urls:
            print(f"⬇️ กำลังดาวน์โหลด {file_name}...")
            
            if download_file(model_urls[file_name], file, file_name):
                print(f"✅ ดาวน์โหลด {file_name} สำเร็จ")
                
                # ตรวจสอบ hash
                if verify_file_hash(file, model_hashes[file_name]):
                    print(f"✅ ตรวจสอบ hash ของ {file_name} สำเร็จ")
                else:
                    print(f"❌ hash ของ {file_name} ไม่ตรงกับที่คาดหวัง")
                    success = False
            else:
                print(f"❌ ดาวน์โหลด {file_name} ไม่สำเร็จ")
                success = False
        else:
            print(f"❌ ไม่พบ URL สำหรับดาวน์โหลด {file_name}")
            success = False
    
    if success:
        print("✅ ดาวน์โหลดและตรวจสอบโมเดลทั้งหมดสำเร็จ")
    else:
        print("⚠️ มีบางโมเดลที่ดาวน์โหลดหรือตรวจสอบไม่สำเร็จ")
    
    return success


def create_dummy_models():
    """สร้างโมเดลตัวอย่างสำหรับการทดสอบ"""
    print("🛠️ กำลังสร้างโมเดลตัวอย่างสำหรับการทดสอบ...")
    
    models_dir = Path("ai_models")
    models_dir.mkdir(exist_ok=True)
    
    # สร้างไฟล์ labels.json
    labels = {
        "0": "โพธิ์ฐานบัว",
        "1": "สีวลี",
        "2": "สมเด็จ",
        "3": "หลวงพ่อกวย"
    }
    
    with open(models_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)
    
    # สร้างไฟล์ตัวอย่างสำหรับ amulet_model.h5
    with open(models_dir / "amulet_model.h5", "wb") as f:
        f.write(b"This is a dummy model file for testing purposes only.")
    
    # สร้างไฟล์ตัวอย่างสำหรับ amulet_model.tflite
    with open(models_dir / "amulet_model.tflite", "wb") as f:
        f.write(b"This is a dummy TFLite model file for testing purposes only.")
    
    print("✅ สร้างโมเดลตัวอย่างสำเร็จ")
    return True


def main():
    """ฟังก์ชันหลัก"""
    print("=" * 50)
    print("🏺 Amulet-AI Model Setup")
    print("=" * 50)
    
    # ตรวจสอบพารามิเตอร์
    if len(sys.argv) > 1 and sys.argv[1] == "--dummy":
        create_dummy_models()
        return
    
    # ดาวน์โหลดและตรวจสอบโมเดล
    try:
        setup_models()
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        print("⚠️ หากต้องการสร้างโมเดลตัวอย่างสำหรับทดสอบ ให้ใช้คำสั่ง: python setup_models.py --dummy")
        return
    
    print("\n💡 ทิป: หากต้องการสร้างโมเดลตัวอย่างสำหรับทดสอบ ให้ใช้คำสั่ง: python setup_models.py --dummy")


if __name__ == "__main__":
    main()
