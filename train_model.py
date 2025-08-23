"""
TensorFlow model training script for amulet classification
Implements transfer learning with popular pre-trained models
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetV2B0, ResNet50V2, MobileNetV3Large
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"🎮 Found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        logger.error(f"❌ GPU configuration error: {e}")
else:
    logger.info("🖥️ Using CPU for training")

class AmuletClassifier:
    def __init__(self, 
                 model_name: str = "EfficientNetV2B0",
                 input_shape: tuple = (224, 224, 3),
                 num_classes: int = 4):
        """
        Initialize amulet classifier
        
        Args:
            model_name: Base model architecture
            input_shape: Input image shape
            num_classes: Number of amulet classes
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.labels = {
            0: "หลวงพ่อกวยแหวกม่าน",
            1: "โพธิ์ฐานบัว", 
            2: "ฐานสิงห์",
            3: "สีวลี"
        }
    
    def build_model(self, trainable_layers: int = 20):
        """
        Build transfer learning model
        
        Args:
            trainable_layers: Number of top layers to make trainable
        """
        logger.info(f"🏗️ Building {self.model_name} model")
        
        # Load pre-trained base model
        if self.model_name == "EfficientNetV2B0":
            base_model = EfficientNetV2B0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_name == "ResNet50V2":
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_name == "MobileNetV3Large":
            base_model = MobileNetV3Large(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build classification head
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu', name='feature_layer')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        logger.info(f"✅ Model built with {self.model.count_params():,} parameters")
        
        # Make top layers trainable for fine-tuning
        if trainable_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
            
            logger.info(f"🔧 Made top {trainable_layers} layers trainable")
    
    def prepare_data(self, dataset_path: str, validation_split: float = 0.2, batch_size: int = 32):
        """
        Prepare data generators for training
        
        Args:
            dataset_path: Path to dataset directory
            validation_split: Fraction of data for validation
            batch_size: Training batch size
            
        Returns:
            train_generator, validation_generator
        """
        logger.info(f"📁 Preparing data from {dataset_path}")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            dataset_path,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Update labels from generator
        class_indices = train_generator.class_indices
        self.labels = {v: k for k, v in class_indices.items()}
        
        logger.info(f"📊 Training samples: {train_generator.samples}")
        logger.info(f"📊 Validation samples: {validation_generator.samples}")
        logger.info(f"🏷️ Classes: {list(class_indices.keys())}")
        
        return train_generator, validation_generator
    
    def train(self, 
              train_generator, 
              validation_generator,
              epochs: int = 50,
              fine_tune_epochs: int = 20):
        """
        Train the model with transfer learning
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator  
            epochs: Initial training epochs
            fine_tune_epochs: Fine-tuning epochs
        """
        logger.info("🚀 Starting model training")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model_checkpoint.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Phase 1: Train with frozen base model
        logger.info("📚 Phase 1: Training with frozen base model")
        history1 = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning
        logger.info("🔧 Phase 2: Fine-tuning")
        
        # Unfreeze and use lower learning rate
        for layer in self.model.layers:
            layer.trainable = True
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        history2 = self.model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            initial_epoch=len(history1.history['loss']),
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
        }
        
        logger.info("✅ Training completed")
    
    def evaluate_model(self, test_generator):
        """Evaluate model performance"""
        logger.info("📊 Evaluating model")
        
        # Get predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # True labels
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())
        
        # Classification report
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_labels,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        logger.info("📋 Classification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=class_labels))
        
        return report, cm
    
    def save_model(self, save_path: str = "models/amulet_classifier"):
        """Save trained model"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save full model
        model_path = os.path.join(save_path, "model.h5")
        self.model.save(model_path)
        logger.info(f"💾 Saved model to {model_path}")
        
        # Save TFLite model
        tflite_path = os.path.join(save_path, "model.tflite")
        self.convert_to_tflite(tflite_path)
        
        # Save labels
        labels_path = os.path.join(save_path, "labels.json")
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved labels to {labels_path}")
        
        # Save training history
        if self.history:
            history_path = os.path.join(save_path, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"💾 Saved training history to {history_path}")
    
    def convert_to_tflite(self, save_path: str):
        """Convert model to TensorFlow Lite"""
        logger.info("📱 Converting to TensorFlow Lite")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Optional: Use representative dataset for better quantization
        converter.representative_dataset = self._representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"📱 TFLite model saved to {save_path}")
        logger.info(f"📏 Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    def _representative_dataset(self):
        """Generate representative dataset for quantization"""
        # This would use actual training data in production
        for _ in range(100):
            yield [np.random.uniform(0, 1, (1, *self.input_shape)).astype(np.float32)]
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        if not self.history:
            logger.warning("⚠️ No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history['loss'], label='Training Loss')
        ax2.plot(self.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"📈 Training plot saved to {save_path}")
        else:
            plt.show()

def main():
    """Main training function"""
    # Configuration
    MODEL_NAME = "EfficientNetV2B0"  # EfficientNetV2B0, ResNet50V2, MobileNetV3Large
    DATASET_PATH = "dataset"
    EPOCHS = 30
    FINE_TUNE_EPOCHS = 15
    BATCH_SIZE = 16
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        logger.error(f"❌ Dataset not found at {DATASET_PATH}")
        logger.info("💡 Please organize your dataset as:")
        logger.info("dataset/")
        logger.info("  ├── หลวงพ่อกวยแหวกม่าน/")
        logger.info("  ├── โพธิ์ฐานบัว/")
        logger.info("  ├── ฐานสิงห์/")
        logger.info("  └── สีวลี/")
        return
    
    # Initialize classifier
    classifier = AmuletClassifier(model_name=MODEL_NAME)
    
    # Build model
    classifier.build_model(trainable_layers=20)
    
    # Prepare data
    train_gen, val_gen = classifier.prepare_data(
        dataset_path=DATASET_PATH,
        batch_size=BATCH_SIZE
    )
    
    # Train model
    classifier.train(
        train_gen, 
        val_gen,
        epochs=EPOCHS,
        fine_tune_epochs=FINE_TUNE_EPOCHS
    )
    
    # Evaluate
    report, cm = classifier.evaluate_model(val_gen)
    
    # Save model and results
    classifier.save_model()
    classifier.plot_training_history("models/training_history.png")
    
    logger.info("🎉 Training completed successfully!")

if __name__ == "__main__":
    main()
