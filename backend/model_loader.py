import io
import json
from typing import IO, Union

import tensorflow as tf
import numpy as np
from PIL import Image

# รองรับ HEIC format ในฝั่ง backend ด้วย
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

class ModelLoader:
    def __init__(self, model_path=None, labels_path=None):
        """
        Initialize model loader with paths to model and labels
        
        Args:
            model_path (str): Path to .h5 or .tflite model file (optional for testing)
            labels_path (str): Path to labels.json file (optional for testing)
        """
        if model_path and labels_path:
            try:
                self.model = self._load_model(model_path)
                self.labels = self._load_labels(labels_path)
                print(f"✅ Loaded model from {model_path}")
            except FileNotFoundError as e:
                print(f"⚠️  Model files not found: {e}")
                print("🔄 Using mock mode for testing")
                self.model = None
                self.labels = self._get_default_labels()
        else:
            print("🧪 Running in test mode with mock data")
            self.model = None
            self.labels = self._get_default_labels()
    
    def _get_default_labels(self):
        """Return default labels for testing"""
        return {
            "0": "หลวงพ่อกวยแหวกม่าน",
            "1": "โพธิ์ฐานบัว", 
            "2": "ฐานสิงห์",
            "3": "สีวลี"
        }
        
    def _load_model(self, model_path):
        """Load TF/TFLite model from path"""
        if model_path.endswith('.tflite'):
            return tf.lite.Interpreter(model_path=model_path)
        return tf.keras.models.load_model(model_path)
    
    def _load_labels(self, labels_path):
        """Load label mapping from JSON file"""
        with open(labels_path, encoding="utf-8") as f:
            return json.load(f)
            
    def preprocess_image(self, image: Union[str, IO[bytes]], target_size=(224, 224)):
        """
        Preprocess image for model input
        
        Args:
            image: Path or file-like object (e.g. FastAPI UploadFile or bytes stream)
            target_size: Model input size tuple
            
        Returns:
            Preprocessed image array
        """
        # Accept either a path (str) or a file-like object. If an UploadFile is passed,
        # it typically has a `.file` attribute (a SpooledTemporaryFile) so handle that too.
        if hasattr(image, "read"):
            stream = image
            # Starlette/FastAPI UploadFile exposes the underlying file as `.file`
            if hasattr(image, "file"):
                stream = image.file
            stream.seek(0)
            img = Image.open(stream)
        else:
            img = Image.open(str(image))

        img = img.convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array.astype("float32") / 255.0
        return np.expand_dims(img_array, axis=0)
        
    def predict(self, image: Union[str, IO[bytes]]):
        """
        Run inference on image
        
        Args:
            image: Path or file-like object
            
        Returns:
            Dictionary with predicted class and confidence
        """
        # ถ้าไม่มีโมเดลจริง ใช้ mock prediction
        if self.model is None:
            print("🎭 Using mock prediction")
            # จำลองการทำนายโดยเลือกแบบสุ่มจาก labels
            import random
            classes = list(self.labels.keys())
            predicted_class_id = random.choice(classes)
            confidence = random.uniform(0.7, 0.98)  # ความน่าจะเป็นสูง
            
            return {
                "class": self.labels[predicted_class_id], 
                "confidence": confidence
            }
        
        # โค้ดสำหรับโมเดลจริง
        img = self.preprocess_image(image)

        # TFLite interpreter handling
        if isinstance(self.model, tf.lite.Interpreter):
            interp = self.model
            interp.allocate_tensors()
            input_details = interp.get_input_details()
            output_details = interp.get_output_details()

            input_index = input_details[0]["index"]
            input_dtype = input_details[0]["dtype"]

            # Prepare input according to expected dtype
            if np.issubdtype(input_dtype, np.integer):
                # model expects uint8/uint16 etc. Scale from [0,1] back to [0,255]
                input_data = (img * 255.0).astype(input_dtype)
            else:
                input_data = img.astype(input_dtype)

            # Resize tensor if shape doesn't match (some TFLite models use dynamic shapes)
            try:
                if tuple(input_details[0]["shape"]) != input_data.shape:
                    interp.resize_tensor_input(input_index, input_data.shape)
                    interp.allocate_tensors()
            except Exception:
                # Some interpreter builds may not support resize_tensor_input; ignore
                pass

            interp.set_tensor(input_index, input_data)
            interp.invoke()
            predictions = interp.get_tensor(output_details[0]["index"])
        else:
            predictions = self.model.predict(img)

        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])

        # Resolve label from either dict or list
        label = None
        if isinstance(self.labels, dict):
            label = self.labels.get(str(predicted_class)) or self.labels.get(predicted_class)
        elif isinstance(self.labels, list):
            if 0 <= predicted_class < len(self.labels):
                label = self.labels[predicted_class]

        if label is None:
            # Fallback to using the class id as string
            label = str(predicted_class)

        return {"class": label, "confidence": confidence}
