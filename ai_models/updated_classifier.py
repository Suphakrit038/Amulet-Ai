#!/usr/bin/env python3
"""
Updated Model Loader for trained_model directory
โหลดโมเดลใหม่จาก trained_model
"""

import joblib
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class UpdatedAmuletClassifier:
    """อัปเดตเวอร์ชัน classifier ที่ใช้โมเดลใหม่"""
    
    def __init__(self, model_path: str = "trained_model"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.class_mapping = None
        self.model_info = None
        self.image_size = (224, 224)
        
    def load_model(self, model_path: Optional[str] = None):
        """โหลดโมเดลและ artifacts ทั้งหมด"""
        if model_path:
            self.model_path = Path(model_path)
            
        print(f"Loading model from: {self.model_path}")
        
        # โหลด model files
        try:
            # โหลด classifier
            classifier_path = self.model_path / "classifier.joblib"
            if classifier_path.exists():
                self.model = joblib.load(classifier_path)
                print("✅ Loaded classifier.joblib")
            else:
                raise FileNotFoundError(f"Classifier not found: {classifier_path}")
            
            # โหลด scaler
            scaler_path = self.model_path / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print("✅ Loaded scaler.joblib")
            else:
                print("⚠️ Scaler not found, will use raw features")
            
            # โหลด label encoder
            encoder_path = self.model_path / "label_encoder.joblib"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                print("✅ Loaded label_encoder.joblib")
            else:
                print("⚠️ Label encoder not found")
            
            # โหลด model info
            info_path = self.model_path / "model_info.json"
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)
                self.class_mapping = self.model_info.get('classes', {})
                self.image_size = tuple(self.model_info.get('image_size', [224, 224]))
                print("✅ Loaded model_info.json")
            else:
                print("⚠️ Model info not found")
            
            print(f"🎯 Model loaded successfully!")
            print(f"📊 Accuracy: {self.model_info.get('accuracy', 'Unknown'):.4f}")
            print(f"📁 Classes: {len(self.class_mapping)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """เตรียมรูปภาพสำหรับการทำนาย"""
        try:
            # ตรวจสอบว่ารูปภาพเป็น numpy array
            if not isinstance(image, np.ndarray):
                return None
            
            # แปลงเป็น RGB ถ้าจำเป็น
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB ถ้าเป็น OpenCV format
                if image.dtype == np.uint8 and np.max(image) > 1:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # ปรับขนาดรูปภาพ
            image_resized = cv2.resize(image, self.image_size)
            
            # Normalize
            if image_resized.dtype == np.uint8:
                image_resized = image_resized.astype(np.float32) / 255.0
            
            # Flatten สำหรับ traditional ML model
            features = image_resized.flatten()
            
            return features.reshape(1, -1)  # Shape: (1, features)
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image: np.ndarray) -> Dict:
        """ทำนายผลจากรูปภาพ"""
        if self.model is None:
            return {
                "success": False,
                "error": "Model not loaded",
                "predicted_class": None,
                "confidence": 0.0,
                "probabilities": {}
            }
        
        try:
            # เตรียมรูปภาพ
            features = self.preprocess_image(image)
            if features is None:
                return {
                    "success": False,
                    "error": "Failed to preprocess image",
                    "predicted_class": None,
                    "confidence": 0.0,
                    "probabilities": {}
                }
            
            # ใช้ scaler ถ้ามี
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # ทำนาย
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # แปลงผลลัพธ์
            class_names = list(self.class_mapping.keys())
            confidence = float(np.max(probabilities))
            predicted_class = class_names[prediction]
            
            # สร้าง probability dictionary
            prob_dict = {
                class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_index": int(prediction),
                "probabilities": prob_dict,
                "model_info": {
                    "version": self.model_info.get('version', '2.0'),
                    "accuracy": self.model_info.get('accuracy', 0.0),
                    "training_date": self.model_info.get('training_date', 'Unknown')
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}",
                "predicted_class": None,
                "confidence": 0.0,
                "probabilities": {}
            }
    
    def predict_from_file(self, image_path: str) -> Dict:
        """ทำนายจากไฟล์รูปภาพ"""
        try:
            # โหลดรูปภาพ
            image = cv2.imread(image_path)
            if image is None:
                # ลองใช้ PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image.convert('RGB'))
            
            return self.predict(image)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load image: {str(e)}",
                "predicted_class": None,
                "confidence": 0.0,
                "probabilities": {}
            }
    
    def get_model_status(self) -> Dict:
        """ได้สถานะของโมเดล"""
        return {
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "label_encoder_loaded": self.label_encoder is not None,
            "model_path": str(self.model_path),
            "num_classes": len(self.class_mapping) if self.class_mapping else 0,
            "class_names": list(self.class_mapping.keys()) if self.class_mapping else [],
            "image_size": self.image_size,
            "model_info": self.model_info
        }

# Global classifier instance
updated_classifier = None

def get_updated_classifier() -> UpdatedAmuletClassifier:
    """ได้ classifier instance (singleton pattern)"""
    global updated_classifier
    if updated_classifier is None:
        updated_classifier = UpdatedAmuletClassifier()
        updated_classifier.load_model()
    return updated_classifier

def test_model():
    """ทดสอบโมเดล"""
    print("=== Testing Updated Amulet Classifier ===")
    
    classifier = UpdatedAmuletClassifier()
    success = classifier.load_model()
    
    if success:
        status = classifier.get_model_status()
        print(f"✅ Model Status: {status}")
        
        # สร้างรูปภาพทดสอบ
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = classifier.predict(test_image)
        print(f"🔮 Test Prediction: {result}")
    else:
        print("❌ Failed to load model")

if __name__ == "__main__":
    test_model()