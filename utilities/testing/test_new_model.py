#!/usr/bin/env python3
"""
Test script สำหรับทดสอบโมเดลใหม่ที่เพิ่งเทรนเสร็จ
"""
import os
import sys
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# เพิ่ม path ของโปรเจกต์
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import โมเดลของเรา
try:
    from ai_models.compatibility_loader import load_production_model
    print("✅ สามารถ import โมเดลได้สำเร็จ")
except Exception as e:
    print(f"❌ Error importing model: {e}")
    sys.exit(1)

def test_model_loading():
    """ทดสอบการโหลดโมเดล"""
    print("\n🔄 กำลังทดสอบการโหลดโมเดล...")
    
    try:
        # โหลดโมเดล
        model = load_production_model('trained_model')
        print("✅ โหลดโมเดลสำเร็จ")
        
        # ตรวจสอบ labels
        with open('ai_models/labels.json', 'r', encoding='utf-8') as f:
            labels = json.load(f)
        
        print(f"📊 จำนวนคลาสในโมเดล: {len(labels)}")
        print("🏷️ คลาสที่มี:")
        for i, (key, label) in enumerate(labels.items()):
            print(f"   {i+1}. {key}: {label}")
        
        return model, labels
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_with_sample_images():
    """ทดสอบกับรูปตัวอย่างจากชุดข้อมูล"""
    print("\n🖼️ กำลังทดสอบกับรูปตัวอย่าง...")
    
    # โหลดโมเดล
    model, labels = test_model_loading()
    if model is None:
        return
    
    # หารูปตัวอย่างจากชุดข้อมูลที่จัดระเบียบแล้ว
    test_folder = "organized_dataset/splits/test"
    
    if not os.path.exists(test_folder):
        print(f"❌ ไม่พบโฟลเดอร์ทดสอบ: {test_folder}")
        return
    
    # ค้นหารูปตัวอย่างในแต่ละคลาส
    test_results = []
    
    for class_folder in os.listdir(test_folder):
        class_path = os.path.join(test_folder, class_folder)
        if os.path.isdir(class_path):
            # หารูปแรกในคลาสนี้
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_image = os.path.join(class_path, images[0])
                print(f"\n🧪 ทดสอบคลาส: {class_folder}")
                print(f"📂 รูปทดสอบ: {images[0]}")
                
                try:
                    # ทดสอบการทำนาย
                    result = model.predict_with_ood_detection(test_image)
                    
                    print(f"🎯 ผลการทำนาย:")
                    print(f"   คลาส: {result.get('predicted_class', 'Unknown')}")
                    print(f"   ความมั่นใจ: {result.get('confidence', 0):.2%}")
                    print(f"   OOD Score: {result.get('ood_score', 0):.4f}")
                    print(f"   สถานะ: {result.get('status', 'Unknown')}")
                    
                    # เก็บผลลัพธ์
                    test_results.append({
                        'actual_class': class_folder,
                        'predicted_class': result.get('predicted_class', ''),
                        'confidence': result.get('confidence', 0),
                        'correct': class_folder == result.get('predicted_class', '')
                    })
                    
                except Exception as e:
                    print(f"❌ Error testing {test_image}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # สรุปผลการทดสอบ
    if test_results:
        print(f"\n📈 สรุปผลการทดสอบ:")
        correct_predictions = sum(1 for r in test_results if r['correct'])
        total_predictions = len(test_results)
        accuracy = correct_predictions / total_predictions
        
        print(f"   ความแม่นยำ: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
        print(f"   ความมั่นใจเฉลี่ย: {np.mean([r['confidence'] for r in test_results]):.2%}")

def check_model_files():
    """ตรวจสอบไฟล์โมเดลที่จำเป็น"""
    print("\n📁 ตรวจสอบไฟล์โมเดล...")
    
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

def main():
    """ฟังก์ชันหลัก"""
    print("🚀 เริ่มการทดสอบระบบ Amulet-AI v3.0")
    print("=" * 50)
    
    # ตรวจสอบไฟล์โมเดล
    if not check_model_files():
        print("❌ ไฟล์โมเดลไม่ครบถ้วน")
        return
    
    # ทดสอบการโหลดโมเดล
    model, labels = test_model_loading()
    if model is None:
        print("❌ ไม่สามารถโหลดโมเดลได้")
        return
    
    # ทดสอบกับรูปตัวอย่าง
    test_with_sample_images()
    
    print("\n✨ การทดสอบเสร็จสิ้น!")
    print("🎉 ระบบ Amulet-AI v3.0 พร้อมใช้งาน!")

if __name__ == "__main__":
    main()