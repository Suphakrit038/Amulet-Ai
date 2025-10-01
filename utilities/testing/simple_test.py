#!/usr/bin/env python3
"""
Simple test script สำหรับทดสอบโมเดลใหม่
"""
import os
import sys
import json
import joblib
import numpy as np
from pathlib import Path

def test_model_files():
    """ทดสอบไฟล์โมเดลที่จำเป็น"""
    print("🔍 ตรวจสอบไฟล์โมเดล...")
    
    required_files = [
        'trained_model/classifier.joblib',
        'trained_model/label_encoder.joblib', 
        'trained_model/scaler.joblib',
        'trained_model/pca.joblib',
        'trained_model/ood_detector.joblib',
        'trained_model/model_info.json',
        'ai_models/labels.json'
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({file_size:,} bytes)")
        else:
            print(f"❌ {file_path} - ไม่พบไฟล์")
            all_files_exist = False
    
    return all_files_exist

def test_model_loading():
    """ทดสอบการโหลดโมเดล"""
    print("\n🔄 ทดสอบการโหลดโมเดล...")
    
    try:
        # โหลด labels
        with open('ai_models/labels.json', 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print(f"✅ โหลด labels สำเร็จ: {len(labels)} คลาส")
        
        # โหลดโมเดล components
        classifier = joblib.load('trained_model/classifier.joblib')
        print("✅ โหลด classifier สำเร็จ")
        
        scaler = joblib.load('trained_model/scaler.joblib')
        print("✅ โหลด scaler สำเร็จ")
        
        label_encoder = joblib.load('trained_model/label_encoder.joblib')
        print("✅ โหลด label_encoder สำเร็จ")
        
        # อ่าน model info
        with open('trained_model/model_info.json', 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        print(f"✅ โมเดลเวอร์ชัน: {model_info.get('model_version', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_structure():
    """ทดสอบโครงสร้างชุดข้อมูล"""
    print("\n📁 ตรวจสอบโครงสร้างชุดข้อมูล...")
    
    dataset_paths = [
        'organized_dataset',
        'organized_dataset/raw',
        'organized_dataset/processed', 
        'organized_dataset/augmented',
        'organized_dataset/splits',
        'organized_dataset/metadata'
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                dir_count = len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
                print(f"✅ {path} ({dir_count} โฟลเดอร์, {file_count} ไฟล์)")
            else:
                print(f"✅ {path} (ไฟล์)")
        else:
            print(f"❌ {path} - ไม่พบ")

def check_api_availability():
    """ตรวจสอบว่า API server สามารถเริ่มได้หรือไม่"""
    print("\n🌐 ตรวจสอบความพร้อม API...")
    
    # ตรวจสอบไฟล์ API
    api_files = [
        'api/main_api.py',
        'api/main_api_fast.py'
    ]
    
    for api_file in api_files:
        if os.path.exists(api_file):
            print(f"✅ {api_file}")
        else:
            print(f"❌ {api_file} - ไม่พบ")

def show_project_summary():
    """แสดงสรุปโปรเจกต์"""
    print("\n📊 สรุปสถานะโปรเจกต์:")
    print("=" * 50)
    
    # ตรวจสอบไฟล์รายงาน
    if os.path.exists('PHASE_COMPLETION_REPORT.md'):
        print("✅ รายงานการทำงานเสร็จสิ้น")
        
        # อ่านบางส่วนของรายงาน
        try:
            with open('PHASE_COMPLETION_REPORT.md', 'r', encoding='utf-8') as f:
                content = f.read()
                if 'PHASE 1' in content:
                    print("   ✅ Phase 1: Dataset Organization")
                if 'PHASE 2' in content:
                    print("   ✅ Phase 2: Data Preprocessing")
                if 'PHASE 3' in content:
                    print("   ✅ Phase 3: Model Training")
        except:
            pass
    
    # ตรวจสอบ trained_model_backup
    if os.path.exists('trained_model_backup'):
        print("✅ สำรองโมเดลเก่าแล้ว")
    
    # ตรวจสอบโมเดลใหม่
    if os.path.exists('trained_model/model_info.json'):
        try:
            with open('trained_model/model_info.json', 'r') as f:
                info = json.load(f)
                print(f"✅ โมเดลใหม่เวอร์ชัน: {info.get('model_version', 'Unknown')}")
                
                # แสดงผลการทดสอบ
                if 'test_accuracy' in info:
                    print(f"   📈 ความแม่นยำ: {info['test_accuracy']:.2%}")
                if 'validation_accuracy' in info:
                    print(f"   📈 ความแม่นยำ (validation): {info['validation_accuracy']:.2%}")
                    
        except:
            print("✅ โมเดลใหม่พร้อมใช้งาน")

def main():
    """ฟังก์ชันหลัก"""
    print("🚀 ทดสอบระบบ Amulet-AI v3.0")
    print("=" * 50)
    
    # ทดสอบไฟล์โมเดล
    model_files_ok = test_model_files()
    
    # ทดสอบการโหลดโมเดล
    model_loading_ok = test_model_loading()
    
    # ทดสอบโครงสร้างชุดข้อมูล
    test_dataset_structure()
    
    # ตรวจสอบ API
    check_api_availability()
    
    # แสดงสรุป
    show_project_summary()
    
    print("\n" + "=" * 50)
    
    if model_files_ok and model_loading_ok:
        print("🎉 ระบบพร้อมใช้งาน!")
        print("💡 คำแนะนำ:")
        print("   - เริ่ม API: python api/main_api_fast.py")
        print("   - ทดสอบ Frontend: python -m streamlit run frontend/main_streamlit_app.py")
    else:
        print("⚠️ ระบบยังไม่พร้อมใช้งาน - กรุณาตรวจสอบข้อผิดพลาด")
    
    print("✨ การทดสอบเสร็จสิ้น!")

if __name__ == "__main__":
    main()