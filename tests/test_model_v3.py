#!/usr/bin/env python3
"""
ทดสอบโมเดล v3.0 ที่เทรนเสร็จแล้ว
"""
import os
import sys
import json
import joblib
import numpy as np
import cv2
from pathlib import Path

def load_model_info():
    """โหลดข้อมูลโมเดล"""
    try:
        with open('trained_model/model_info.json', 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        with open('ai_models/labels.json', 'r', encoding='utf-8') as f:
            labels = json.load(f)
            
        return model_info, labels
    except Exception as e:
        print(f"❌ Error loading model info: {e}")
        return None, None

def test_basic_prediction():
    """ทดสอบการทำนายพื้นฐาน"""
    print("\n🧪 ทดสอบการทำนายพื้นฐาน...")
    
    try:
        # โหลดโมเดล components
        classifier = joblib.load('trained_model/classifier.joblib')
        scaler = joblib.load('trained_model/scaler.joblib')
        label_encoder = joblib.load('trained_model/label_encoder.joblib')
        
        # หารูปทดสอบ
        test_folder = "organized_dataset/splits/test"
        if not os.path.exists(test_folder):
            print("❌ ไม่พบโฟลเดอร์ทดสอบ")
            return
        
        test_results = []
        
        # ทดสอบกับรูปในแต่ละคลาส
        for class_folder in os.listdir(test_folder):
            class_path = os.path.join(test_folder, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            # ตรวจสอบว่ามี front/back หรือรูปตรงๆ
            front_path = os.path.join(class_path, "front")
            back_path = os.path.join(class_path, "back")
            
            # หารูปทดสอบ
            test_images = []
            if os.path.exists(front_path):
                front_images = [f for f in os.listdir(front_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if front_images:
                    test_images.append(os.path.join(front_path, front_images[0]))
            
            if os.path.exists(back_path):
                back_images = [f for f in os.listdir(back_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if back_images:
                    test_images.append(os.path.join(back_path, back_images[0]))
            
            # ถ้าไม่มี front/back ให้หารูปตรงๆ
            if not test_images:
                direct_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if direct_images:
                    test_images.append(os.path.join(class_path, direct_images[0]))
            
            if not test_images:
                print(f"\n⚠️ ไม่พบรูปทดสอบในคลาส: {class_folder}")
                continue
                
            print(f"\n🖼️ ทดสอบคลาส: {class_folder}")
            print(f"   จำนวนรูปทดสอบ: {len(test_images)}")
            
            # ทดสอบกับรูปแรก
            test_image_path = test_images[0]
            image_name = os.path.basename(test_image_path)
            print(f"   รูป: {image_name}")
            
            try:
                # โหลดและประมวลผลรูป
                image = cv2.imread(test_image_path)
                if image is None:
                    print(f"   ❌ ไม่สามารถโหลดรูปได้")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extract features แบบง่าย (ใช้วิธีเดียวกับตอนเทรน)
                features = extract_simple_features(image)
                
                # Scale features
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # ทำนาย
                prediction = classifier.predict(features_scaled)[0]
                probabilities = classifier.predict_proba(features_scaled)[0]
                
                # แปลงกลับเป็นชื่อคลาส
                predicted_class = label_encoder.inverse_transform([prediction])[0]
                confidence = float(probabilities[prediction])
                
                print(f"   🎯 ผลการทำนาย: {predicted_class}")
                print(f"   📊 ความมั่นใจ: {confidence:.2%}")
                
                # ตรวจสอบความถูกต้อง (ต้องจัดการกับ portrait_back)
                actual_class = class_folder.replace('_back', '').replace('_front', '')
                predicted_class_clean = predicted_class.replace('_back', '').replace('_front', '')
                is_correct = predicted_class_clean == actual_class
                print(f"   ✅ ถูกต้อง: {'YES' if is_correct else 'NO'}")
                
                # แสดงความน่าจะเป็นทั้งหมด
                print(f"   📋 ความน่าจะเป็นทั้งหมด:")
                all_classes = label_encoder.classes_
                for i, prob in enumerate(probabilities):
                    class_name = all_classes[i]
                    print(f"      {class_name}: {prob:.2%}")
                
                test_results.append({
                    'actual': actual_class,
                    'predicted': predicted_class_clean,
                    'confidence': confidence,
                    'correct': is_correct
                })
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                
        # สรุปผลการทดสอบ
        if test_results:
            print(f"\n📈 สรุปผลการทดสอบ:")
            correct_count = sum(1 for r in test_results if r['correct'])
            total_count = len(test_results)
            accuracy = correct_count / total_count if total_count > 0 else 0
            avg_confidence = np.mean([r['confidence'] for r in test_results])
            
            print(f"   ความแม่นยำ: {accuracy:.2%} ({correct_count}/{total_count})")
            print(f"   ความมั่นใจเฉลี่ย: {avg_confidence:.2%}")
            
            print(f"\n📋 รายละเอียด:")
            for r in test_results:
                status = "✅" if r['correct'] else "❌"
                print(f"   {status} {r['actual']} -> {r['predicted']} ({r['confidence']:.2%})")
                
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()

def extract_simple_features(image):
    """Extract features แบบเดียวกับตอนเทรน (raw pixels)"""
    # Resize to standard size (224x224 เหมือนตอนเทรน)
    image_resized = cv2.resize(image, (224, 224))
    
    # Convert to float and normalize (0-1)
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Flatten to 1D array (224 * 224 * 3 = 150,528 features)
    features = image_normalized.flatten()
    
    return features

def check_model_performance():
    """ตรวจสอบประสิทธิภาพโมเดล"""
    print("\n📊 ประสิทธิภาพโมเดล:")
    
    model_info, labels = load_model_info()
    if not model_info:
        return
        
    print(f"   📈 ความแม่นยำ (Training): {model_info['training_results']['train_accuracy']:.2%}")
    print(f"   📈 ความแม่นยำ (Validation): {model_info['training_results']['val_accuracy']:.2%}")
    print(f"   📈 ความแม่นยำ (Test): {model_info['training_results']['test_accuracy']:.2%}")
    
    print(f"\n🏷️ คลาสที่รองรับ ({len(labels)} คลาส):")
    for key, value in labels.items():
        print(f"   • {key}: {value}")

def main():
    """ฟังก์ชันหลัก"""
    print("🚀 ทดสอบ Amulet-AI Model v3.0")
    print("=" * 60)
    
    # ตรวจสอบไฟล์โมเดล
    required_files = [
        'trained_model/classifier.joblib',
        'trained_model/scaler.joblib',
        'trained_model/label_encoder.joblib',
        'trained_model/model_info.json',
        'ai_models/labels.json'
    ]
    
    print("🔍 ตรวจสอบไฟล์โมเดล...")
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ ไฟล์ที่ขาดหาย: {len(missing_files)} ไฟล์")
        return
    
    # แสดงประสิทธิภาพโมเดล
    check_model_performance()
    
    # ทดสอบการทำนาย
    test_basic_prediction()
    
    print("\n" + "=" * 60)
    print("🎉 การทดสอบเสร็จสิ้น!")
    print("💡 โมเดล v3.0 พร้อมใช้งาน")
    print("🌐 เริ่ม API: python api/main_api_fast.py")
    print("🖥️ เริ่ม Frontend: python -m streamlit run frontend/main_streamlit_app.py")

if __name__ == "__main__":
    main()