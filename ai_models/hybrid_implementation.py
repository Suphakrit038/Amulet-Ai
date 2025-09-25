"""
🚀 Hybrid ML Pipeline Implementation Guide
Step-by-step guide to deploy the optimal amulet recognition system
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
from hybrid_ml_pipeline import HybridMLPipeline, HybridPipelineConfig

class AmuletRecognitionSystem:
    """Complete implementation of hybrid amulet recognition system"""
    
    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = Path(dataset_path)
        self.pipeline = None
        self.labels_mapping = {}
        self.config = None
        
    def setup_optimized_config(self) -> HybridPipelineConfig:
        """Create optimized configuration for amulet dataset"""
        
        # Load labels mapping
        labels_file = Path("data_base/labels.json")
        if labels_file.exists():
            with open(labels_file, 'r', encoding='utf-8') as f:
                self.labels_mapping = json.load(f)
        
        # Optimized config for small dataset (173 images, 10 classes)
        config = HybridPipelineConfig(
            # Dataset
            data_path=str(self.dataset_path),
            image_size=(224, 224),
            random_state=42,
            
            # CNN Features (PyTorch CPU)
            use_cnn_features=True,
            cnn_backbone="resnet50",  # Best balance performance/speed
            cnn_freeze_layers=True,
            
            # Classical Features (OpenCV)
            use_classical_features=True,
            use_hog=True,    # Shape patterns
            use_lbp=True,    # Texture patterns  
            use_color_hist=True,  # Color distribution
            use_edge_features=True,  # Edge information
            
            # Dimensionality Reduction (Critical for small dataset)
            use_pca=True,
            pca_components=200,  # Conservative for 173 samples
            normalize_features=True,
            
            # Ensemble Models (Diverse for robustness)
            ensemble_models=['random_forest', 'svm', 'gradient_boost'],
            ensemble_voting="soft",  # Probabilistic voting
            
            # Small Dataset Optimization
            use_augmentation=True,
            augmentation_factor=8,  # 173 → 1,384 images
            use_class_weights=True,  # Handle imbalanced classes
            use_stratified_kfold=True,
            cv_folds=5,  # Good for small dataset
            
            # Performance
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )
        
        self.config = config
        return config
    
    def load_dataset(self) -> tuple:
        """Load and organize dataset for training"""
        images = []
        labels = []
        
        print(f"📂 Loading dataset from {self.dataset_path}")
        
        # Check if dataset exists
        if not self.dataset_path.exists():
            print(f"❌ Dataset path not found: {self.dataset_path}")
            return [], []
        
        # Scan for images in class folders
        class_counts = {}
        
        for class_folder in self.dataset_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                class_images = []
                
                # Load images from class folder
                for img_file in class_folder.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        try:
                            # Load image
                            img = cv2.imread(str(img_file))
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                class_images.append(img_rgb)
                                
                        except Exception as e:
                            print(f"⚠️ Error loading {img_file}: {e}")
                
                # Add to dataset
                if class_images:
                    images.extend(class_images)
                    labels.extend([class_name] * len(class_images))
                    class_counts[class_name] = len(class_images)
        
        print(f"✅ Dataset loaded: {len(images)} images, {len(set(labels))} classes")
        print("📊 Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"   {class_name}: {count} images")
        
        return images, labels
    
    def train_system(self):
        """Train the complete hybrid system"""
        print("🚀 Starting Amulet Recognition System Training")
        print("=" * 60)
        
        # Setup configuration
        config = self.setup_optimized_config()
        
        # Create pipeline
        self.pipeline = HybridMLPipeline(config)
        
        # Load dataset
        images, labels = self.load_dataset()
        
        if not images:
            print("❌ No images found for training!")
            return False
        
        # Train pipeline
        try:
            self.pipeline.fit(images, labels)
            
            # Save trained system
            save_dir = "ai_models/saved_models/hybrid_pipeline"
            self.pipeline.save(save_dir)
            
            print("🎉 Training completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def load_trained_system(self, model_path: str = "ai_models/saved_models/hybrid_pipeline"):
        """Load pre-trained system"""
        try:
            # Setup configuration (needed for extractors)
            config = self.setup_optimized_config()
            self.pipeline = HybridMLPipeline(config)
            
            # Load trained components
            self.pipeline.load(model_path)
            
            print(f"✅ Trained system loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load system: {e}")
            return False
    
    def predict_image(self, image_path: str) -> dict:
        """Predict single image"""
        if not self.pipeline:
            print("❌ System not trained or loaded!")
            return {}
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not load image"}
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Predict
            predictions, probabilities = self.pipeline.predict([img_rgb])
            
            # Format results
            result = {
                "predicted_class": predictions[0],
                "confidence": float(np.max(probabilities[0])),
                "all_probabilities": {
                    label: float(prob) 
                    for label, prob in zip(self.pipeline.label_encoder.classes_, probabilities[0])
                }
            }
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_system(self, test_images: list, test_labels: list) -> dict:
        """Evaluate system performance"""
        if not self.pipeline:
            print("❌ System not trained or loaded!")
            return {}
        
        try:
            predictions, probabilities = self.pipeline.predict(test_images)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            accuracy = accuracy_score(test_labels, predictions)
            report = classification_report(test_labels, predictions, output_dict=True)
            
            results = {
                "accuracy": accuracy,
                "classification_report": report,
                "num_test_samples": len(test_images)
            }
            
            return results
            
        except Exception as e:
            return {"error": str(e)}

def demo_implementation():
    """Demonstration of the hybrid system"""
    print("🎯 Amulet Recognition Hybrid ML System Demo")
    print("=" * 50)
    
    # Initialize system
    system = AmuletRecognitionSystem(dataset_path="dataset")
    
    # Option 1: Train new system
    print("\n1️⃣ Training new hybrid system...")
    success = system.train_system()
    
    if success:
        print("\n2️⃣ Testing prediction...")
        
        # Test on a sample image (if available)
        test_image = Path("dataset").glob("**/*.jpg")
        test_image = next(test_image, None)
        
        if test_image:
            result = system.predict_image(str(test_image))
            print(f"📸 Test image: {test_image}")
            print(f"🎯 Prediction: {result}")
        
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo failed during training")

# Performance Analysis
def analyze_system_performance():
    """Analyze expected performance of hybrid system"""
    
    print("📊 HYBRID SYSTEM PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Dataset Analysis
    print("\n🗂️ DATASET CHARACTERISTICS:")
    print("   • Total Images: 173")
    print("   • Classes: 10")
    print("   • Class Imbalance: 16:1 ratio (2-36 images/class)")
    print("   • After Augmentation: 1,384 images (8x)")
    
    # Feature Analysis
    print("\n🔧 FEATURE EXTRACTION:")
    print("   • CNN Features (ResNet50): 2,048 dimensions")
    print("   • Classical Features: ~4,200 dimensions")
    print("   • Total Raw Features: ~6,200 dimensions")
    print("   • After PCA: 200 dimensions")
    
    # Model Analysis
    print("\n🏗️ MODEL ARCHITECTURE:")
    print("   • Ensemble: RandomForest + SVM + GradientBoost")
    print("   • Voting: Soft (probabilistic)")
    print("   • Class Balancing: Enabled")
    print("   • Cross-Validation: 5-fold stratified")
    
    # Expected Performance
    print("\n🎯 EXPECTED PERFORMANCE:")
    print("   • Training Accuracy: 95-99% (with augmentation)")
    print("   • Cross-Validation: 85-92%")
    print("   • Real-world Accuracy: 80-88%")
    print("   • Processing Speed: 3-5 images/second")
    print("   • Memory Usage: 1-2 GB RAM")
    
    # Advantages
    print("\n✅ SYSTEM ADVANTAGES:")
    print("   • Python 3.13 Compatible")
    print("   • No GPU Required")
    print("   • Robust to Small Dataset")
    print("   • Handles Class Imbalance")
    print("   • Combines Deep + Classical Features")
    print("   • Ensemble Robustness")
    
    # Challenges
    print("\n⚠️ POTENTIAL CHALLENGES:")
    print("   • Small dataset may overfit")
    print("   • Class imbalance remains challenging")
    print("   • CNN features limited by CPU-only PyTorch")
    print("   • Need careful validation")

if __name__ == "__main__":
    print("🧠 OPTIMAL HYBRID ML PIPELINE FOR AMULET RECOGNITION")
    print("Combining CNN (PyTorch) + Classical (OpenCV) + Ensemble (scikit-learn)")
    print("=" * 70)
    
    # Show performance analysis
    analyze_system_performance()
    
    print("\n" + "=" * 70)
    print("🚀 Ready to run demo? Uncomment the line below:")
    print("# demo_implementation()")