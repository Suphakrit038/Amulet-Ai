"""
🧪 Ultra-Simple Model Inference Test
ทดสอบโมเดลที่ฝึกแล้วด้วยระบบฉุกเฉิน
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraSimpleModel(nn.Module):
    """Ultra-simple CNN for emergency training"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Very simple architecture
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),  # 32x32 -> 8x8
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
            
            nn.AdaptiveAvgPool2d((2, 2))  # Force to 2x2
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(32 * 2 * 2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def load_and_test_model():
    """Load trained model and test it"""
    logger.info("🧪 Testing Ultra-Simple Emergency Model")
    
    # Load model checkpoint
    model_path = Path("training_output/ultra_simple_model.pth")
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        return
    
    logger.info(f"📥 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model info
    categories = checkpoint['categories']
    num_classes = len(categories)
    model_params = checkpoint['model_params']
    training_stats = checkpoint['training_stats']
    
    logger.info(f"🧠 Model parameters: {model_params:,}")
    logger.info(f"📊 Categories: {num_classes}")
    logger.info(f"🎯 Training success rate: {training_stats['successful_samples']}/{training_stats['successful_samples'] + training_stats['failed_samples']}")
    
    # Create and load model
    model = UltraSimpleModel(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("✅ Model loaded successfully")
    
    # Create reverse category mapping
    idx_to_category = {v: k for k, v in categories.items()}
    
    # Test with some images from test set
    test_path = Path("dataset_split/test")
    if not test_path.exists():
        logger.error(f"❌ Test dataset not found: {test_path}")
        return
    
    # Find test images
    test_images = []
    actual_labels = []
    
    for category_dir in test_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            if category_name in categories:
                # Get up to 2 images per category
                image_count = 0
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for img_path in category_dir.glob(ext):
                        if image_count < 2:
                            test_images.append(img_path)
                            actual_labels.append(category_name)
                            image_count += 1
    
    logger.info(f"🧪 Testing with {len(test_images)} test images")
    
    # Test predictions
    correct_predictions = 0
    total_predictions = 0
    
    for img_path, actual_label in zip(test_images, actual_labels):
        try:
            # Load and preprocess image
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
                
                predicted_label = idx_to_category[predicted_idx]
                
                is_correct = predicted_label == actual_label
                if is_correct:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Display result
                status = "✅" if is_correct else "❌"
                logger.info(f"{status} {img_path.name}")
                logger.info(f"   Actual: {actual_label}")
                logger.info(f"   Predicted: {predicted_label}")
                logger.info(f"   Confidence: {confidence:.2%}")
                
        except Exception as e:
            logger.error(f"❌ Failed to test {img_path}: {e}")
    
    # Final results
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    logger.info("📊 TESTING COMPLETED!")
    logger.info(f"🎯 Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
    
    # Save test results
    test_results = {
        'model_type': 'ultra_simple_emergency',
        'model_parameters': model_params,
        'total_test_images': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy_percent': accuracy,
        'categories_tested': len(set(actual_labels)),
        'training_stats': training_stats
    }
    
    results_path = Path("training_output/test_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Test results saved to: {results_path}")
    
    return test_results

if __name__ == "__main__":
    try:
        results = load_and_test_model()
        logger.info("✅ Model testing completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Model testing failed: {e}")
        import traceback
        traceback.print_exc()
