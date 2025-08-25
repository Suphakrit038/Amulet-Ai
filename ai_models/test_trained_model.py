"""
Test script for trained somdej-fatherguay AI model - PostgreSQL Version
ทดสอบโมเดล AI ที่เทรนแล้วสำหรับพระสมเด็จหลวงพ่อกวย (ใช้ Smart Image Processor)
"""
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from smart_image_processor import SmartImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmuletTester:
    """AI Model Tester for amulet recognition with Smart Image Processing"""
    
    def __init__(self, model_path="ai_models"):
        self.model_path = Path(model_path)
        self.model = None
        self.metadata = None
        self.img_size = (224, 224)
        
        # Initialize Smart Image Processor
        self.image_processor = SmartImageProcessor(
            target_size=self.img_size,
            padding_color=(128, 128, 128)
        )
        
        # Load model and metadata
        self.load_model()
        
    def load_model(self):
        """Load trained model and metadata"""
        logger.info("📥 Loading trained model...")
        
        # Try to load Keras format first (recommended)
        keras_path = self.model_path / "somdej-fatherguay_trained_model.keras"
        h5_path = self.model_path / "somdej-fatherguay_trained_model.h5"
        metadata_path = self.model_path / "somdej-fatherguay_trained_model_metadata.json"
        
        # Load model
        if keras_path.exists():
            self.model = tf.keras.models.load_model(keras_path)
            logger.info(f"✅ Loaded Keras model: {keras_path}")
        elif h5_path.exists():
            self.model = tf.keras.models.load_model(h5_path)
            logger.info(f"✅ Loaded H5 model: {h5_path}")
        else:
            raise FileNotFoundError("No trained model found!")
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"✅ Loaded metadata: {metadata_path}")
        else:
            logger.warning("⚠️ No metadata file found")
            
    def preprocess_image(self, image_path):
        """Preprocess image for model prediction using Smart Image Processor"""
        try:
            # Use Smart Image Processor for consistent processing
            img_array, success, metadata = self.image_processor.process_for_training(
                image_path, method='pad'  # Same method as training
            )
            
            if success:
                # Add batch dimension and ensure proper normalization
                if img_array.dtype != np.float32:
                    img_array = img_array.astype(np.float32) / 255.0
                
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                
                logger.info(f"✅ Preprocessed with Smart Processor: {metadata.get('resize_method', 'pad')}")
                return img_array, metadata
            else:
                logger.error(f"❌ Smart processing failed: {metadata.get('error')}")
                return None, None
                
        except Exception as e:
            logger.error(f"❌ Failed to preprocess image {image_path}: {e}")
            return None, None
    
    def predict_single_image(self, image_path):
        """Predict single image using Smart Image Processing"""
        logger.info(f"🔍 Predicting: {Path(image_path).name}")
        
        # Preprocess image with Smart Processor
        img_array, processing_metadata = self.preprocess_image(image_path)
        if img_array is None:
            return None
        
        # Make prediction
        try:
            prediction = self.model.predict(img_array, verbose=0)
            probability = float(prediction[0][0])  # Get probability
            is_somdej_fatherguay = probability > 0.5
            confidence = probability if is_somdej_fatherguay else (1 - probability)
            
            result = {
                'image_path': str(image_path),
                'image_name': Path(image_path).name,
                'prediction': 'somdej-fatherguay' if is_somdej_fatherguay else 'other',
                'probability': probability,
                'confidence': confidence,
                'is_target_class': is_somdej_fatherguay
            }
            
            logger.info(f"   Result: {result['prediction']} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return None
    
    def test_dataset_folder(self, folder_path):
        """Test all images in a folder"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            logger.error(f"❌ Folder not found: {folder_path}")
            return []
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in folder_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        logger.info(f"📂 Testing folder: {folder_path}")
        logger.info(f"📊 Found {len(image_files)} images")
        
        results = []
        correct_predictions = 0
        
        for image_file in image_files:
            result = self.predict_single_image(image_file)
            if result:
                results.append(result)
                
                # Check if prediction is correct (assuming folder name indicates class)
                folder_name = folder_path.name.lower()
                is_correct = (
                    (result['is_target_class'] and 'somdej-fatherguay' in folder_name) or
                    (not result['is_target_class'] and 'somdej-fatherguay' not in folder_name)
                )
                
                if is_correct:
                    correct_predictions += 1
                    
        accuracy = correct_predictions / len(results) if results else 0
        logger.info(f"🎯 Folder accuracy: {accuracy:.3f} ({correct_predictions}/{len(results)})")
        
        return results
    
    def run_comprehensive_test(self):
        """Run comprehensive test on all dataset folders"""
        logger.info("🧪 Running comprehensive model test...")
        logger.info("="*50)
        
        dataset_path = Path("dataset")
        if not dataset_path.exists():
            logger.error("❌ Dataset folder not found!")
            return
        
        all_results = {}
        total_correct = 0
        total_images = 0
        
        # Test each dataset folder
        dataset_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        for folder in dataset_folders:
            folder_results = self.test_dataset_folder(folder)
            all_results[folder.name] = folder_results
            
            # Calculate accuracy for this folder
            folder_name = folder.name.lower()
            correct_in_folder = 0
            
            for result in folder_results:
                is_correct = (
                    (result['is_target_class'] and 'somdej-fatherguay' in folder_name) or
                    (not result['is_target_class'] and 'somdej-fatherguay' not in folder_name)
                )
                if is_correct:
                    correct_in_folder += 1
                    total_correct += 1
                total_images += 1
            
            logger.info(f"📊 {folder.name}: {correct_in_folder}/{len(folder_results)} correct")
        
        # Overall accuracy
        overall_accuracy = total_correct / total_images if total_images > 0 else 0
        logger.info("="*50)
        logger.info(f"🏆 Overall Test Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_images})")
        
        # Show model info
        if self.metadata:
            logger.info("📋 Model Information:")
            logger.info(f"   Class: {self.metadata['class_name']}")
            logger.info(f"   Training Date: {self.metadata['training_date']}")
            logger.info(f"   Training Accuracy: {self.metadata['metrics']['accuracy']:.3f}")
            logger.info(f"   F1-Score: {self.metadata['metrics']['f1_score']:.3f}")
        
        return all_results
    
    def predict_sample_images(self):
        """Predict some sample images for demonstration"""
        logger.info("🖼️ Testing sample images...")
        
        # Test somdej-fatherguay images (should be positive)
        somdej_path = Path("dataset/somdej-fatherguay")
        if somdej_path.exists():
            somdej_images = list(somdej_path.glob("*.jpg"))[:3]  # First 3 images
            logger.info("✅ Somdej-FatherGuay samples (should predict as positive):")
            for img in somdej_images:
                result = self.predict_single_image(img)
                if result:
                    status = "✅ CORRECT" if result['is_target_class'] else "❌ WRONG"
                    logger.info(f"   {img.name}: {result['prediction']} ({result['confidence']:.3f}) {status}")
        
        # Test other class images (should be negative)
        other_folders = [d for d in Path("dataset").iterdir() 
                        if d.is_dir() and d.name != "somdej-fatherguay"]
        
        if other_folders:
            logger.info("🔄 Other class samples (should predict as negative):")
            for folder in other_folders[:2]:  # First 2 other classes
                other_images = list(folder.glob("*.jpg"))[:2]  # 2 images each
                for img in other_images:
                    result = self.predict_single_image(img)
                    if result:
                        status = "✅ CORRECT" if not result['is_target_class'] else "❌ WRONG"
                        logger.info(f"   {img.name}: {result['prediction']} ({result['confidence']:.3f}) {status}")

def main():
    """Main testing function"""
    print("🧪 AI Model Testing for Somdej-FatherGuay")
    print("="*45)
    
    try:
        # Initialize tester
        tester = AmuletTester()
        
        # Run sample predictions
        tester.predict_sample_images()
        
        print("\n" + "="*45)
        
        # Run comprehensive test
        all_results = tester.run_comprehensive_test()
        
        print("\n🎉 Testing completed successfully!")
        print("🔗 Model is ready for integration with the API system")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
