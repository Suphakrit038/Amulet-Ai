#!/usr/bin/env python3
"""
Quick AI Training Script for Amulet-AI
เทรนโมเดลแบบด่วนจาก dataset
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2

# Configuration
DATASET_PATH = "Data set"
MODEL_OUTPUT_PATH = "trained_model"
IMAGE_SIZE = (224, 224)

# Classes mapping
CLASS_MAPPING = {
    "ปรกโพธิ์9ใบ": 0,
    "พระสมเด็จประธานพรเนื้อพุทธกวัก": 1,
    "พระสีวลี": 2,
    "วัดหนองอีดุก": 3,
    "หลังรูปเหมือน": 4,
    "แหวกม่าน": 5
}

def preprocess_image(image_path):
    """Image preprocessing with augmentation"""
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, IMAGE_SIZE)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Flatten for traditional ML
        return img.flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_dataset():
    """Load and preprocess dataset"""
    X, y = [], []
    dataset_path = Path(DATASET_PATH)
    
    print("Loading dataset...")
    
    for class_folder in dataset_path.iterdir():
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        if class_name not in CLASS_MAPPING:
            continue
        
        print(f"Processing class: {class_name}")
        class_label = CLASS_MAPPING[class_name]
        
        # Process all images in class folder
        for img_file in class_folder.rglob("*.png"):
            features = preprocess_image(img_file)
            if features is not None:
                X.append(features)
                y.append(class_label)
        
        # Process JPG files too
        for img_file in class_folder.rglob("*.jpg"):
            features = preprocess_image(img_file)
            if features is not None:
                X.append(features)
                y.append(class_label)
    
    return np.array(X), np.array(y)

def train_model():
    """Train the model"""
    print("Starting training...")
    
    # Load dataset
    X, y = load_dataset()
    print(f"Dataset loaded: {len(X)} samples, {len(np.unique(y))} classes")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    
    # Save classifier
    joblib.dump(model, f"{MODEL_OUTPUT_PATH}/classifier.joblib")
    
    # Save label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(list(CLASS_MAPPING.keys()))
    joblib.dump(label_encoder, f"{MODEL_OUTPUT_PATH}/label_encoder.joblib")
    
    # Save model info
    model_info = {
        "model_type": "RandomForestClassifier",
        "accuracy": float(accuracy),
        "classes": CLASS_MAPPING,
        "image_size": IMAGE_SIZE,
        "num_samples": len(X)
    }
    
    import json
    with open(f"{MODEL_OUTPUT_PATH}/model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print("Model saved successfully!")
    return model, label_encoder

if __name__ == "__main__":
    train_model()