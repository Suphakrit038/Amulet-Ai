"""
AI Training Script for Single Class: somdej-fatherguay
เทรน AI โมเดลสำหรับพระสมเด็จหลวงพ่อกวยโฟลเดอร์เดียว
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from PIL import Image
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SingleClassTrainer:
    """AI Trainer for single amulet class"""
    
    def __init__(self, dataset_path="dataset", class_name="somdej-fatherguay"):
        self.dataset_path = Path(dataset_path)
        self.class_name = class_name
        self.class_path = self.dataset_path / class_name
        self.model_save_path = Path("ai_models")
        self.model_save_path.mkdir(exist_ok=True)
        
        # Model parameters
        self.img_size = (224, 224)
        self.batch_size = 16  # Small batch for limited data
        self.epochs = 50
        self.learning_rate = 0.0001
        
        logger.info(f"🤖 Initializing trainer for class: {class_name}")
        
    def verify_dataset(self):
        """Verify dataset exists and has images"""
        if not self.class_path.exists():
            raise ValueError(f"Dataset path not found: {self.class_path}")
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in self.class_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        logger.info(f"📊 Found {len(image_files)} images in {self.class_name}")
        
        if len(image_files) < 10:
            logger.warning(f"⚠️ Only {len(image_files)} images found. Recommend at least 20 for training.")
        
        return image_files
    
    def prepare_data(self, image_files):
        """Prepare training data for binary classification"""
        logger.info("📋 Preparing training data...")
        
        X = []  # Images
        y = []  # Labels (1 for somdej-fatherguay, 0 for others)
        
        # Load somdej-fatherguay images (positive class)
        for img_file in image_files:
            try:
                # Load and preprocess image
                img = load_img(img_file, target_size=self.img_size)
                img_array = img_to_array(img) / 255.0  # Normalize
                
                X.append(img_array)
                y.append(1)  # Positive class
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {img_file}: {e}")
        
        # Generate negative samples from other classes (if available)
        other_classes = [d for d in self.dataset_path.iterdir() 
                        if d.is_dir() and d.name != self.class_name 
                        and d.name != '__pycache__']
        
        negative_samples_needed = len(X)  # Same amount as positive
        negative_count = 0
        
        for other_class in other_classes:
            if negative_count >= negative_samples_needed:
                break
                
            other_images = [f for f in other_class.iterdir() 
                          if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
            
            for img_file in other_images[:max(1, negative_samples_needed // len(other_classes))]:
                try:
                    img = load_img(img_file, target_size=self.img_size)
                    img_array = img_to_array(img) / 255.0
                    
                    X.append(img_array)
                    y.append(0)  # Negative class
                    negative_count += 1
                    
                    if negative_count >= negative_samples_needed:
                        break
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load negative sample {img_file}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✅ Prepared dataset: {len(X)} samples")
        logger.info(f"   - Positive samples (somdej-fatherguay): {np.sum(y == 1)}")
        logger.info(f"   - Negative samples (others): {np.sum(y == 0)}")
        
        return X, y
    
    def create_model(self):
        """Create transfer learning model"""
        logger.info("🧠 Creating AI model...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("✅ Model created successfully")
        return model
    
    def train_model(self, X, y):
        """Train the model"""
        logger.info("🚀 Starting model training...")
        
        # Create model
        model = self.create_model()
        
        # Split data (simple random split)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Training split: {len(X_train)} train, {len(X_val)} validation")
        
        # Data augmentation for training
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                self.model_save_path / f'{self.class_name}_best.h5',
                monitor='val_accuracy', save_best_only=True, verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tune with unfrozen layers
        logger.info("🔧 Fine-tuning model...")
        model.layers[0].trainable = True  # Unfreeze base model
        
        # Lower learning rate for fine-tuning
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate / 10),
            loss='binary_crossentropy', 
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Continue training
        history_finetune = model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=20,  # Fewer epochs for fine-tuning
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history, history_finetune
    
    def evaluate_model(self, model, X, y):
        """Evaluate trained model"""
        logger.info("📊 Evaluating model performance...")
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info("🎯 Model Performance:")
        logger.info(f"   Accuracy:  {accuracy:.3f}")
        logger.info(f"   Precision: {precision:.3f}")
        logger.info(f"   Recall:    {recall:.3f}")
        logger.info(f"   F1-Score:  {f1:.3f}")
        logger.info(f"   Confusion Matrix:")
        logger.info(f"   {cm}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
    
    def save_model(self, model, metrics):
        """Save trained model and metadata"""
        logger.info("💾 Saving trained model...")
        
        # Save model in multiple formats
        model_name = f"{self.class_name}_trained_model"
        
        # Keras native format (recommended)
        keras_path = self.model_save_path / f"{model_name}.keras"
        model.save(keras_path)
        
        # H5 format (legacy compatibility)
        h5_path = self.model_save_path / f"{model_name}.h5"
        model.save(h5_path)
        
        # Export SavedModel format for TFLite/TFServing
        saved_model_path = self.model_save_path / f"{model_name}_savedmodel"
        model.export(str(saved_model_path))
        
        # TensorFlow Lite (mobile deployment)
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
            tflite_model = converter.convert()
            
            tflite_path = self.model_save_path / f"{model_name}.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            logger.info(f"✅ TensorFlow Lite model saved: {tflite_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create TFLite model: {e}")
        
        # Save metadata
        metadata = {
            'class_name': self.class_name,
            'model_type': 'binary_classification',
            'input_shape': [*self.img_size, 3],
            'training_date': datetime.now().isoformat(),
            'metrics': metrics,
            'description': f'Binary classifier for {self.class_name} amulet recognition'
        }
        
        metadata_path = self.model_save_path / f"{model_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Model saved successfully:")
        logger.info(f"   Keras format:   {keras_path}")
        logger.info(f"   H5 format:      {h5_path}")
        logger.info(f"   SavedModel:     {saved_model_path}")
        logger.info(f"   Metadata:       {metadata_path}")
        
        return keras_path, h5_path, saved_model_path, metadata_path
    
    def run_training(self):
        """Run complete training pipeline"""
        logger.info("🏺 Starting Amulet AI Training Pipeline")
        logger.info("="*50)
        
        try:
            # Step 1: Verify dataset
            image_files = self.verify_dataset()
            
            # Step 2: Prepare data
            X, y = self.prepare_data(image_files)
            
            if len(X) < 10:
                raise ValueError("Not enough training data. Need at least 10 samples.")
            
            # Step 3: Train model
            model, history, history_finetune = self.train_model(X, y)
            
            # Step 4: Evaluate model
            metrics = self.evaluate_model(model, X, y)
            
            # Step 5: Save model
            saved_paths = self.save_model(model, metrics)
            
            logger.info("🎉 Training completed successfully!")
            logger.info("="*50)
            
            return model, metrics, saved_paths
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

def main():
    """Main training function"""
    print("🤖 AI Training for Somdej-FatherGuay")
    print("="*40)
    
    # Check TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # Initialize trainer
    trainer = SingleClassTrainer(
        dataset_path="dataset",
        class_name="somdej-fatherguay"
    )
    
    # Run training
    try:
        model, metrics, saved_paths = trainer.run_training()
        
        print("\n🎯 Training Results:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        
        print("\n🚀 Next Steps:")
        print("1. Test the model with new images")
        print("2. Integrate with the API system")
        print("3. Add more training data if needed")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
