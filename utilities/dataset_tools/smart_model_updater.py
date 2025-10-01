#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Model Updater - ใช้โมเดลที่มีอยู่แล้วและอัปเดตเฉพาะที่จำเป็น
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Configuration
TRAINED_MODEL_PATH = "trained_model"
LABELS_PATH = "ai_models/labels.json"
DATASET_PATH = "Data set"

def check_existing_models():
    """ตรวจสอบโมเดลที่มีอยู่แล้ว"""
    models = {}
    
    model_files = [
        "classifier.joblib",
        "scaler.joblib", 
        "label_encoder.joblib",
        "pca.joblib",
        "ood_detector.joblib"
    ]
    
    print("🔍 Scanning existing models...")
    
    for model_file in model_files:
        model_path = os.path.join(TRAINED_MODEL_PATH, model_file)
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                models[model_file] = {
                    "path": model_path,
                    "model": model,
                    "size": os.path.getsize(model_path),
                    "modified": datetime.fromtimestamp(os.path.getmtime(model_path))
                }
                print(f"✅ Found: {model_file} ({models[model_file]['size']} bytes)")
            except Exception as e:
                print(f"❌ Error loading {model_file}: {e}")
        else:
            print(f"❌ Missing: {model_file}")
    
    return models

def check_model_info():
    """ตรวจสอบข้อมูลโมเดล"""
    info_files = ["model_info.json", "deployment_info.json"]
    info = {}
    
    for info_file in info_files:
        info_path = os.path.join(TRAINED_MODEL_PATH, info_file)
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    info[info_file] = json.load(f)
                print(f"✅ Found: {info_file}")
            except Exception as e:
                print(f"❌ Error reading {info_file}: {e}")
        else:
            print(f"❌ Missing: {info_file}")
    
    return info

def check_dataset_structure():
    """ตรวจสอบโครงสร้าง dataset"""
    dataset_path = Path(DATASET_PATH)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {DATASET_PATH}")
        return {}
    
    print(f"🔍 Scanning dataset: {DATASET_PATH}")
    
    classes = {}
    for class_folder in dataset_path.iterdir():
        if class_folder.is_dir():
            # Count image files
            image_files = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
                image_files.extend(list(class_folder.rglob(ext)))
            
            classes[class_folder.name] = {
                "path": str(class_folder),
                "image_count": len(image_files),
                "sample_files": [str(f) for f in image_files[:3]]  # First 3 files as samples
            }
            
            print(f"📁 {class_folder.name}: {len(image_files)} images")
    
    return classes

def update_labels_mapping(dataset_classes):
    """อัปเดต labels mapping ให้ตรงกับ dataset"""
    if not dataset_classes:
        print("❌ No dataset classes found")
        return False
    
    # Load existing labels
    try:
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
    except:
        print("❌ Could not load labels.json")
        return False
    
    # Update dataset mapping
    labels_data["dataset_mapping"] = {}
    labels_data["current_classes"] = {}
    
    for idx, class_name in enumerate(sorted(dataset_classes.keys())):
        labels_data["dataset_mapping"][class_name] = idx
        labels_data["current_classes"][str(idx)] = class_name
    
    labels_data["total_dataset_classes"] = len(dataset_classes)
    labels_data["last_updated"] = datetime.now().isoformat()
    
    # Save updated labels
    try:
        with open(LABELS_PATH, 'w', encoding='utf-8') as f:
            json.dump(labels_data, f, ensure_ascii=False, indent=2)
        print("✅ Updated labels.json with current dataset")
        return True
    except Exception as e:
        print(f"❌ Error updating labels.json: {e}")
        return False

def create_label_encoder_for_dataset(dataset_classes):
    """สร้าง label encoder ใหม่สำหรับ dataset ปัจจุบัน"""
    if not dataset_classes:
        return None
    
    # Create new label encoder
    label_encoder = LabelEncoder()
    class_names = sorted(dataset_classes.keys())
    label_encoder.fit(class_names)
    
    # Save new label encoder
    encoder_path = os.path.join(TRAINED_MODEL_PATH, "dataset_label_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)
    
    print(f"✅ Created new label encoder for dataset: {encoder_path}")
    print(f"📝 Classes: {class_names}")
    
    return label_encoder

def main():
    """Main function - สแกนและใช้ของเดิม"""
    print("🚀 Smart Model Updater Starting...")
    print("=" * 50)
    
    # 1. Check existing models
    existing_models = check_existing_models()
    print()
    
    # 2. Check model info
    model_info = check_model_info()
    print()
    
    # 3. Check dataset
    dataset_classes = check_dataset_structure()
    print()
    
    # 4. Update labels if needed
    if dataset_classes:
        print("📝 Updating labels mapping...")
        update_labels_mapping(dataset_classes)
        
        # Create dataset-specific label encoder
        create_label_encoder_for_dataset(dataset_classes)
        print()
    
    # 5. Summary
    print("📊 SUMMARY")
    print("=" * 30)
    print(f"✅ Existing models: {len(existing_models)}")
    print(f"✅ Model info files: {len(model_info)}")
    print(f"✅ Dataset classes: {len(dataset_classes)}")
    
    if len(existing_models) >= 3:  # classifier, scaler, label_encoder
        print("\n🎉 READY TO USE!")
        print("✅ Sufficient models found - no need to retrain")
        print("✅ API can start with existing models")
        
        # Test model loading
        try:
            classifier = joblib.load(os.path.join(TRAINED_MODEL_PATH, "classifier.joblib"))
            scaler = joblib.load(os.path.join(TRAINED_MODEL_PATH, "scaler.joblib"))
            print("✅ Models loaded successfully")
            
            # Show model info
            if "model_info.json" in model_info:
                info = model_info["model_info.json"]
                print(f"📈 Model accuracy: {info.get('model_metrics', {}).get('overall_accuracy', 'N/A')}")
                print(f"📊 Training samples: {info.get('dataset_info', {}).get('samples', 'N/A')}")
        except Exception as e:
            print(f"⚠️ Warning: Error testing models: {e}")
    
    else:
        print("\n⚠️ INCOMPLETE MODELS")
        print("❌ Need to train new models")
    
    print("\n🏁 Scan Complete!")

if __name__ == "__main__":
    main()