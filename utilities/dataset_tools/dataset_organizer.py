#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Organizer and Training Pipeline
จัดระเบียบฐานข้อมูลและเทรนโมเดลใหม่
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import cv2
from datetime import datetime

# Paths
DATASET_PATH = "Data set"
MODEL_OUTPUT_PATH = "trained_model"
LABELS_PATH = "ai_models/labels.json"
IMAGE_SIZE = (224, 224)

class DatasetOrganizer:
    def __init__(self):
        self.dataset_path = Path(DATASET_PATH)
        self.model_path = Path(MODEL_OUTPUT_PATH)
        self.labels_path = Path(LABELS_PATH)
        
        # Load current labels
        self.load_labels()
        
    def load_labels(self):
        """Load existing labels configuration"""
        if self.labels_path.exists():
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                self.labels_config = json.load(f)
        else:
            self.labels_config = {}
        
        # Get current dataset mapping
        self.class_mapping = self.labels_config.get('dataset_mapping', {})
        print(f"Loaded {len(self.class_mapping)} classes from labels.json")
        
    def scan_dataset(self):
        """Scan dataset and organize files"""
        print("=== Scanning Dataset ===")
        
        dataset_info = {
            'total_classes': 0,
            'total_images': 0,
            'class_details': {},
            'problematic_files': []
        }
        
        for class_folder in self.dataset_path.iterdir():
            if not class_folder.is_dir():
                continue
                
            class_name = class_folder.name
            print(f"\nScanning class: {class_name}")
            
            # Count images
            image_files = []
            valid_files = []
            invalid_files = []
            
            # Scan for image files
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                image_files.extend(list(class_folder.rglob(ext)))
            
            # Validate each image
            for img_file in image_files:
                try:
                    # Try to open image
                    with Image.open(img_file) as img:
                        # Check if image is valid
                        img.verify()
                    valid_files.append(img_file)
                except Exception as e:
                    print(f"  Invalid image: {img_file.name} - {e}")
                    invalid_files.append(str(img_file))
            
            # Store class info
            dataset_info['class_details'][class_name] = {
                'total_files': len(image_files),
                'valid_files': len(valid_files),
                'invalid_files': len(invalid_files),
                'file_paths': [str(f) for f in valid_files[:5]]  # Sample paths
            }
            
            dataset_info['total_images'] += len(valid_files)
            dataset_info['problematic_files'].extend(invalid_files)
            
            print(f"  Total files: {len(image_files)}")
            print(f"  Valid images: {len(valid_files)}")
            print(f"  Invalid images: {len(invalid_files)}")
        
        dataset_info['total_classes'] = len(dataset_info['class_details'])
        
        # Save scan results
        with open('dataset_scan_results.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== Dataset Scan Complete ===")
        print(f"Total classes: {dataset_info['total_classes']}")
        print(f"Total valid images: {dataset_info['total_images']}")
        print(f"Problematic files: {len(dataset_info['problematic_files'])}")
        
        return dataset_info
    
    def preprocess_image(self, image_path):
        """Preprocess a single image"""
        try:
            # Use PIL for reliable image loading
            with Image.open(image_path) as img:
                # Convert to RGB
                img = img.convert('RGB')
                
                # Resize
                img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Normalize
                img_array = img_array.astype(np.float32) / 255.0
                
                # Flatten for traditional ML
                return img_array.flatten()
                
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def load_dataset_for_training(self):
        """Load and preprocess dataset for training"""
        print("\n=== Loading Dataset for Training ===")
        
        X, y, file_paths = [], [], []
        
        for class_name, class_index in self.class_mapping.items():
            class_folder = self.dataset_path / class_name
            if not class_folder.exists():
                print(f"Warning: Class folder not found: {class_name}")
                continue
            
            print(f"Loading class {class_index}: {class_name}")
            
            # Find all image files
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                image_files.extend(list(class_folder.rglob(ext)))
            
            class_count = 0
            for img_file in image_files:
                features = self.preprocess_image(img_file)
                if features is not None:
                    X.append(features)
                    y.append(class_index)
                    file_paths.append(str(img_file))
                    class_count += 1
            
            print(f"  Loaded {class_count} images")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nDataset loaded: {len(X)} samples, {len(np.unique(y))} classes")
        return X, y, file_paths
    
    def train_new_model(self):
        """Train a new model with current dataset"""
        print("\n=== Training New Model ===")
        
        # Load dataset
        X, y, file_paths = self.load_dataset_for_training()
        
        if len(X) == 0:
            print("Error: No images loaded for training!")
            return None
        
        # Split dataset
        test_size = min(0.2, 1.0 - 2.0/len(X))  # Ensure minimum samples
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Feature scaling
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("Training Random Forest classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        
        # Get class names for report
        class_names = [name for name, idx in sorted(self.class_mapping.items(), key=lambda x: x[1])]
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Save model
        self.save_model_artifacts(model, scaler, accuracy, len(X))
        
        return model, scaler, accuracy
    
    def save_model_artifacts(self, model, scaler, accuracy, total_samples):
        """Save all model artifacts"""
        print("\n=== Saving Model Artifacts ===")
        
        # Create output directory
        self.model_path.mkdir(exist_ok=True)
        
        # Save classifier
        classifier_path = self.model_path / "classifier.joblib"
        joblib.dump(model, classifier_path)
        print(f"Saved: {classifier_path}")
        
        # Save scaler
        scaler_path = self.model_path / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"Saved: {scaler_path}")
        
        # Create and save label encoder
        label_encoder = LabelEncoder()
        class_names = [name for name, idx in sorted(self.class_mapping.items(), key=lambda x: x[1])]
        label_encoder.fit(class_names)
        
        encoder_path = self.model_path / "label_encoder.joblib"
        joblib.dump(label_encoder, encoder_path)
        print(f"Saved: {encoder_path}")
        
        # Save model info
        model_info = {
            "model_type": "RandomForestClassifier",
            "accuracy": float(accuracy),
            "classes": self.class_mapping,
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "num_samples": total_samples,
            "features": IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "training_date": datetime.now().isoformat(),
            "version": "2.0"
        }
        
        model_info_path = self.model_path / "model_info.json"
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        print(f"Saved: {model_info_path}")
        
        # Update deployment info
        deployment_info = {
            "model_path": "classifier.joblib",
            "scaler_path": "scaler.joblib",
            "label_encoder_path": "label_encoder.joblib",
            "model_version": "2.0",
            "deployment_date": datetime.now().isoformat(),
            "status": "ready",
            "accuracy": float(accuracy),
            "total_classes": len(self.class_mapping)
        }
        
        deployment_path = self.model_path / "deployment_info.json"
        with open(deployment_path, 'w', encoding='utf-8') as f:
            json.dump(deployment_info, f, ensure_ascii=False, indent=2)
        print(f"Saved: {deployment_path}")
        
        return model_info
    
    def update_labels_config(self, model_info):
        """Update labels configuration with new training info"""
        print("\n=== Updating Labels Configuration ===")
        
        # Update labels.json with new training info
        self.labels_config['last_training'] = {
            'date': datetime.now().isoformat(),
            'accuracy': model_info['accuracy'],
            'samples': model_info['num_samples'],
            'version': model_info['version']
        }
        
        # Save updated labels
        with open(self.labels_path, 'w', encoding='utf-8') as f:
            json.dump(self.labels_config, f, ensure_ascii=False, indent=2)
        
        print(f"Updated: {self.labels_path}")

def main():
    """Main execution"""
    print("=== Amulet AI Dataset Organizer & Trainer ===")
    
    # Initialize organizer
    organizer = DatasetOrganizer()
    
    # Step 1: Scan and organize dataset
    dataset_info = organizer.scan_dataset()
    
    # Step 2: Train new model
    if dataset_info['total_images'] > 0:
        model, scaler, accuracy = organizer.train_new_model()
        
        if model is not None:
            # Step 3: Update configuration
            model_info = {
                "accuracy": accuracy,
                "num_samples": dataset_info['total_images'],
                "version": "2.0"
            }
            organizer.update_labels_config(model_info)
            
            print(f"\n=== Training Complete ===")
            print(f"✅ Model trained successfully!")
            print(f"📊 Accuracy: {accuracy:.4f}")
            print(f"📁 Model saved in: {MODEL_OUTPUT_PATH}")
            print(f"🔄 Ready for API integration")
        else:
            print("❌ Model training failed!")
    else:
        print("❌ No valid images found for training!")

if __name__ == "__main__":
    main()