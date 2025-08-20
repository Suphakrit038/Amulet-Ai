import tensorflow as tf
import numpy as np
from PIL import Image

class ModelLoader:
    def __init__(self, model_path, labels_path):
        """
        Initialize model loader with paths to model and labels
        
        Args:
            model_path (str): Path to .h5 or .tflite model file
            labels_path (str): Path to labels.json file
        """
        self.model = self._load_model(model_path)
        self.labels = self._load_labels(labels_path)
        
    def _load_model(self, model_path):
        """Load TF/TFLite model from path"""
        if model_path.endswith('.tflite'):
            return tf.lite.Interpreter(model_path=model_path)
        return tf.keras.models.load_model(model_path)
    
    def _load_labels(self, labels_path):
        """Load label mapping from JSON file"""
        import json
        with open(labels_path) as f:
            return json.load(f)
            
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to image file
            target_size: Model input size tuple
            
        Returns:
            Preprocessed image array
        """
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        return np.expand_dims(img_array, axis=0)
        
    def predict(self, image_path):
        """
        Run inference on image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with predicted class and confidence
        """
        img = self.preprocess_image(image_path)
        
        if isinstance(self.model, tf.lite.Interpreter):
            self.model.allocate_tensors()
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            
            self.model.set_tensor(input_details[0]['index'], img)
            self.model.invoke()
            predictions = self.model.get_tensor(output_details[0]['index'])
        else:
            predictions = self.model.predict(img)
            
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return {
            'class': self.labels[str(predicted_class)],
            'confidence': confidence
        }
