#!/usr/bin/env python3
"""
Optimized AI Model for Small Dataset - Amulet-AI
โมเดล AI ที่ปรับให้เหมาะสำหรับข้อมูลขนาดเล็ก (20 รูปต่อ class)
"""

import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptimizedAmuletModel:
    """โมเดล AI ที่ปรับให้เหมาะสำหรับข้อมูลขนาดเล็ก"""
    
    def __init__(self):
        self.feature_extractor = self._create_feature_extractor()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classes = ['phra_nang_phya', 'phra_rod', 'phra_somdej']
        
    def _create_feature_extractor(self):
        """สร้าง feature extractor ที่เหมาะสำหรับข้อมูลเล็ก"""
        return {
            'hog': cv2.HOGDescriptor(),
            'orb': cv2.ORB_create(nfeatures=50),  # ลดจำนวน features
            'lbp_radius': 2,
            'lbp_points': 16
        }
        
    def extract_features(self, image_path):
        """สกัด features จากรูปภาพ"""
        try:
            # โหลดรูปภาพ
            img = cv2.imread(str(image_path))
            if img is None:
                return None
                
            # ปรับขนาดให้เหมาะสม
            img = cv2.resize(img, (128, 128))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            features = []
            
            # 1. HOG Features (ลดขนาด)
            hog_features = self.feature_extractor['hog'].compute(gray)
            if hog_features is not None:
                features.extend(hog_features.flatten()[:200])  # เอาแค่ 200 features
            
            # 2. ORB Features (simplified)
            keypoints, descriptors = self.feature_extractor['orb'].detectAndCompute(gray, None)
            if descriptors is not None:
                # ใช้ mean ของ descriptors
                orb_mean = np.mean(descriptors, axis=0)
                features.extend(orb_mean)
            else:
                features.extend([0] * 32)  # ORB descriptor size
                
            # 3. Color Histogram (ย่อ)
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            features.extend(hist.flatten())
            
            # 4. Texture Features (LBP simplified)
            lbp = self._calculate_lbp(gray)
            lbp_hist, _ = np.histogram(lbp, bins=16, range=(0, 16))
            features.extend(lbp_hist)
            
            # 5. Basic Statistical Features
            features.extend([
                np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
                np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
            
    def _calculate_lbp(self, gray):
        """คำนวณ Local Binary Pattern (แบบง่าย)"""
        height, width = gray.shape
        lbp = np.zeros((height-2, width-2), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = gray[i, j]
                binary_string = ''
                
                # เช็ค 8 จุดรอบๆ
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                    
                lbp[i-1, j-1] = int(binary_string, 2) % 16  # ลดให้เหลือ 16 patterns
                
        return lbp
    
    def load_dataset(self, dataset_path):
        """โหลด dataset และสกัด features"""
        print("📁 Loading optimized dataset...")
        
        X = []
        y = []
        
        dataset_path = Path(dataset_path)
        train_path = dataset_path / 'train'
        
        for class_name in self.classes:
            class_path = train_path / class_name
            if not class_path.exists():
                print(f"⚠️ Class directory not found: {class_path}")
                continue
                
            images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            print(f"   {class_name}: {len(images)} images")
            
            for img_path in images:
                features = self.extract_features(img_path)
                if features is not None:
                    X.append(features)
                    y.append(class_name)
                    
        return np.array(X), np.array(y)
    
    def train(self, dataset_path):
        """เทรนโมเดลด้วยข้อมูลที่ปรับแล้ว"""
        print("🤖 Training optimized AI model...")
        
        # โหลดข้อมูล
        X, y = self.load_dataset(dataset_path)
        
        if len(X) == 0:
            print("❌ No training data found!")
            return False
            
        print(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # เตรียมข้อมูล
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        
        # สร้างโมเดลที่เหมาะสำหรับข้อมูลเล็ก
        self.model = RandomForestClassifier(
            n_estimators=50,      # ลดจำนวน trees
            max_depth=10,         # จำกัดความลึก
            min_samples_split=3,  # เพิ่มค่าต่ำสุดสำหรับการแบ่ง
            min_samples_leaf=2,   # เพิ่มค่าต่ำสุดของ leaf
            random_state=42,
            class_weight='balanced'  # จัดการ class imbalance
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y_encoded, 
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        print(f"📈 Cross-validation scores: {cv_scores}")
        print(f"📈 Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # เทรนโมเดลสุดท้าย
        self.model.fit(X_scaled, y_encoded)
        
        # ประเมินผล
        train_score = self.model.score(X_scaled, y_encoded)
        print(f"📈 Training accuracy: {train_score:.3f}")
        
        # แสดง feature importance
        feature_importance = self.model.feature_importances_
        print(f"🔍 Top 5 important features (indices): {np.argsort(feature_importance)[-5:][::-1]}")
        
        return True
    
    def predict(self, image_path):
        """ทำนายผลจากรูปภาพ"""
        if self.model is None:
            return None, None
            
        features = self.extract_features(image_path)
        if features is None:
            return None, None
            
        # ปรับ features ให้มีขนาดเท่าที่เทรน
        try:
            expected_features = self.scaler.n_features_in_
        except AttributeError:
            # ถ้าไม่มี n_features_in_ ใช้ shape จาก scaler
            expected_features = len(self.scaler.mean_) if hasattr(self.scaler, 'mean_') else len(features)
        
        if len(features) != expected_features:
            # Pad หรือ truncate ให้เท่ากัน
            if len(features) < expected_features:
                features = np.pad(features, (0, expected_features - len(features)))
            else:
                features = features[:expected_features]
        
        try:
            features_scaled = self.scaler.transform([features])
            
            # ทำนาย
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            class_name = self.label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities.max()
            
            return class_name, confidence
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None
    
    def save_model(self, save_dir):
        """บันทึกโมเดล"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # บันทึกโมเดลและ components
        joblib.dump(self.model, save_dir / 'optimized_model.joblib')
        joblib.dump(self.scaler, save_dir / 'scaler.joblib')
        joblib.dump(self.label_encoder, save_dir / 'label_encoder.joblib')
        
        # บันทึก config
        config = {
            'model_type': 'OptimizedRandomForest',
            'classes': self.classes,
            'n_features': self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None,
            'created_date': datetime.now().isoformat(),
            'description': 'Optimized model for small dataset (20 samples per class)'
        }
        
        with open(save_dir / 'model_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Model saved to: {save_dir}")
        
    def load_model(self, model_dir):
        """โหลดโมเดล"""
        model_dir = Path(model_dir)
        
        try:
            self.model = joblib.load(model_dir / 'optimized_model.joblib')
            self.scaler = joblib.load(model_dir / 'scaler.joblib')
            self.label_encoder = joblib.load(model_dir / 'label_encoder.joblib')
            
            with open(model_dir / 'model_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.classes = config['classes']
                
            print(f"📂 Model loaded from: {model_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """ฟังก์ชันหลัก - เทรนโมเดลใหม่"""
    dataset_path = "c:/Users/Admin/Documents/GitHub/Amulet-Ai/dataset_optimized"
    model_save_path = "c:/Users/Admin/Documents/GitHub/Amulet-Ai/trained_model_optimized"
    
    # สร้างและเทรนโมเดล
    model = OptimizedAmuletModel()
    
    if model.train(dataset_path):
        model.save_model(model_save_path)
        print("✅ Model training completed successfully!")
    else:
        print("❌ Model training failed!")

if __name__ == "__main__":
    main()