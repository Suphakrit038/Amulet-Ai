#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Training & Integration - Phase 3
เทรนโมเดลใหม่และรวมเข้าระบบ
"""

import os
import numpy as np
from pathlib import Path
import cv2
import joblib
import json
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class ModelTrainer:
    def __init__(self, dataset_dir="organized_dataset", model_dir="trained_model_v2"):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.image_size = (224, 224)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.class_mapping = {}
        
    def load_images_from_folder(self, folder_path):
        """โหลดรูปภาพจากโฟลเดอร์"""
        images = []
        labels = []
        
        print(f"    📂 โหลดจาก: {folder_path}")
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(folder_path.glob(ext))
        
        for img_file in image_files:
            try:
                # Load and preprocess image
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                # Convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, self.image_size)
                
                # Normalize
                img = img.astype(np.float32) / 255.0
                
                # Flatten
                img_flattened = img.flatten()
                
                images.append(img_flattened)
                
                # Extract class from path (parent folder name)
                class_name = folder_path.parent.name
                labels.append(class_name)
                
            except Exception as e:
                print(f"      ❌ ข้อผิดพลาด {img_file.name}: {e}")
                continue
        
        print(f"      ✅ โหลด: {len(images)} ไฟล์")
        return np.array(images), np.array(labels)
    
    def load_dataset(self, split='train'):
        """โหลด dataset สำหรับ split ที่กำหนด"""
        print(f"\n📊 โหลด {split} dataset...")
        
        split_dir = self.dataset_dir / "splits" / split
        
        all_images = []
        all_labels = []
        class_stats = defaultdict(int)
        
        # Load each class
        for class_folder in split_dir.iterdir():
            if not class_folder.is_dir():
                continue
                
            class_name = class_folder.name
            print(f"  📁 {class_name}")
            
            # Load front and back images
            for side in ['front', 'back']:
                side_folder = class_folder / side
                if side_folder.exists():
                    images, labels = self.load_images_from_folder(side_folder)
                    
                    if len(images) > 0:
                        all_images.extend(images)
                        # Add side information to label
                        side_labels = [f"{label}_{side}" for label in labels]
                        all_labels.extend(side_labels)
                        class_stats[f"{class_name}_{side}"] += len(images)
        
        print(f"\n  📈 สถิติ {split}:")
        for class_side, count in class_stats.items():
            print(f"    {class_side}: {count} ไฟล์")
        
        return np.array(all_images), np.array(all_labels)
    
    def create_simple_class_mapping(self, labels):
        """สร้าง class mapping แบบง่าย (ไม่แยกหน้า-หลัง)"""
        # Extract base class names (remove _front, _back)
        base_classes = set()
        for label in labels:
            base_class = label.replace('_front', '').replace('_back', '')
            base_classes.add(base_class)
        
        # Create mapping
        self.class_mapping = {cls: idx for idx, cls in enumerate(sorted(base_classes))}
        
        # Convert labels to base classes
        simple_labels = []
        for label in labels:
            base_class = label.replace('_front', '').replace('_back', '')
            simple_labels.append(base_class)
        
        return np.array(simple_labels)
    
    def train_model(self):
        """เทรนโมเดล"""
        print("🚀 เริ่มเทรนโมเดล...")
        print("=" * 50)
        
        # Load training data
        X_train, y_train_detailed = self.load_dataset('train')
        X_val, y_val_detailed = self.load_dataset('validation')
        
        # Simplify labels (remove front/back distinction)
        y_train = self.create_simple_class_mapping(y_train_detailed)
        y_val = self.create_simple_class_mapping(y_val_detailed)
        
        print(f"\n📊 ข้อมูลเทรนนิ่ง:")
        print(f"  🔢 Training samples: {len(X_train)}")
        print(f"  🔢 Validation samples: {len(X_val)}")
        print(f"  🎯 Classes: {len(self.class_mapping)}")
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        # Scale features
        print(f"\n⚙️ Feature scaling...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        print(f"\n🎯 เทรนโมเดล Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # Evaluate
        print(f"\n📊 ประเมินผล...")
        
        # Training accuracy
        train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train_encoded, train_pred)
        
        # Validation accuracy
        val_pred = self.model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val_encoded, val_pred)
        
        print(f"  ✅ Training Accuracy: {train_accuracy:.4f}")
        print(f"  ✅ Validation Accuracy: {val_accuracy:.4f}")
        
        # Cross validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train_encoded, cv=5)
        print(f"  ✅ Cross-validation: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        class_names = self.label_encoder.classes_
        report = classification_report(y_val_encoded, val_pred, target_names=class_names)
        print(report)
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report
        }
    
    def save_model(self, training_results):
        """บันทึกโมเดล"""
        print(f"\n💾 บันทึกโมเดล...")
        
        # Save model components
        joblib.dump(self.model, self.model_dir / "classifier.joblib")
        joblib.dump(self.scaler, self.model_dir / "scaler.joblib") 
        joblib.dump(self.label_encoder, self.model_dir / "label_encoder.joblib")
        
        # Save class mapping
        with open(self.model_dir / "class_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(self.class_mapping, f, ensure_ascii=False, indent=2)
        
        # Save model info
        model_info = {
            "version": "3.0",
            "created_at": datetime.datetime.now().isoformat(),
            "architecture": "Random Forest Classifier",
            "image_size": self.image_size,
            "num_classes": len(self.class_mapping),
            "classes": list(self.class_mapping.keys()),
            "training_results": training_results,
            "model_params": self.model.get_params()
        }
        
        with open(self.model_dir / "model_info.json", 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        # Save deployment info
        deployment_info = {
            "model_path": "classifier.joblib",
            "scaler_path": "scaler.joblib",
            "label_encoder_path": "label_encoder.joblib",
            "class_mapping_path": "class_mapping.json",
            "model_version": "3.0",
            "deployment_date": datetime.datetime.now().isoformat(),
            "status": "ready",
            "accuracy": training_results['val_accuracy']
        }
        
        with open(self.model_dir / "deployment_info.json", 'w', encoding='utf-8') as f:
            json.dump(deployment_info, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ บันทึกแล้วใน: {self.model_dir}")
        
    def test_model(self):
        """ทดสอบโมเดลกับ test set"""
        print(f"\n🧪 ทดสอบโมเดลกับ test set...")
        
        # Load test data
        X_test, y_test_detailed = self.load_dataset('test')
        y_test = self.create_simple_class_mapping(y_test_detailed)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        test_pred = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test_encoded, test_pred)
        
        print(f"  ✅ Test Accuracy: {test_accuracy:.4f}")
        
        # Detailed results
        class_names = self.label_encoder.classes_
        test_report = classification_report(y_test_encoded, test_pred, target_names=class_names)
        print(f"\n📋 Test Classification Report:")
        print(test_report)
        
        return test_accuracy, test_report
    
    def run_phase3(self):
        """รัน Phase 3"""
        print("🚀 เริ่ม Phase 3: Model Training & Integration")
        print("=" * 60)
        
        # Train model
        training_results = self.train_model()
        
        # Test model
        test_accuracy, test_report = self.test_model()
        training_results['test_accuracy'] = test_accuracy
        training_results['test_report'] = test_report
        
        # Save model
        self.save_model(training_results)
        
        print(f"\n🎉 Phase 3 เสร็จสิ้น!")
        print(f"📊 สรุปผลการเทรน:")
        print(f"  🎯 Validation Accuracy: {training_results['val_accuracy']:.4f}")
        print(f"  🧪 Test Accuracy: {training_results['test_accuracy']:.4f}")
        print(f"  📁 โมเดลบันทึกใน: {self.model_dir}")
        
        return training_results

def main():
    trainer = ModelTrainer()
    results = trainer.run_phase3()
    
    print(f"\n🎯 ผลลัพธ์สุดท้าย:")
    print(f"  ✅ โมเดลใหม่ (v3.0) พร้อมใช้งาน")
    print(f"  📊 Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"  🎯 {len(trainer.class_mapping)} คลาส")

if __name__ == "__main__":
    main()