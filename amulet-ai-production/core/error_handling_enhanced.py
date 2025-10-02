#!/usr/bin/env python3
"""
🚨 Comprehensive Error Handler
จัดการข้อผิดพลาดแบบครอบคลุมสำหรับทุกส่วนของระบบ
"""
import logging
import traceback
from functools import wraps
from typing import Any, Callable, Dict, Optional
import os
from datetime import datetime

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"amulet_ai_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AmuletAIError(Exception):
    """Base exception class for Amulet-AI"""
    pass

class ModelLoadError(AmuletAIError):
    """Model loading related errors"""
    pass

class ImageProcessingError(AmuletAIError):
    """Image processing related errors"""
    pass

class PredictionError(AmuletAIError):
    """Prediction related errors"""
    pass

class DataValidationError(AmuletAIError):
    """Data validation related errors"""
    pass

def error_handler(error_type: str = "general"):
    """Decorator for comprehensive error handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except AmuletAIError as e:
                logger.error(f"[{error_type.upper()}] Amulet-AI Error in {func.__name__}: {str(e)}")
                return {
                    "status": "error",
                    "error_type": error_type,
                    "message": str(e),
                    "function": func.__name__
                }
            except FileNotFoundError as e:
                logger.error(f"[{error_type.upper()}] File not found in {func.__name__}: {str(e)}")
                return {
                    "status": "error",
                    "error_type": "file_not_found",
                    "message": f"Required file not found: {str(e)}",
                    "function": func.__name__
                }
            except ValueError as e:
                logger.error(f"[{error_type.upper()}] Value error in {func.__name__}: {str(e)}")
                return {
                    "status": "error",
                    "error_type": "value_error",
                    "message": f"Invalid value: {str(e)}",
                    "function": func.__name__
                }
            except Exception as e:
                logger.error(f"[{error_type.upper()}] Unexpected error in {func.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
                return {
                    "status": "error",
                    "error_type": "unexpected",
                    "message": "An unexpected error occurred",
                    "function": func.__name__,
                    "details": str(e)
                }
        return wrapper
    return decorator

def validate_image_file(file_path: str) -> bool:
    """Validate if file is a valid image"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in valid_extensions:
            raise DataValidationError(f"Invalid image format: {file_ext}")
        
        # Check file size (max 50MB)
        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:
            raise DataValidationError(f"Image file too large: {file_size / 1024 / 1024:.1f}MB")
        
        # Try to open with cv2
        import cv2
        img = cv2.imread(file_path)
        if img is None:
            raise ImageProcessingError(f"Cannot read image file: {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise

def validate_model_components(model_dir: str) -> Dict[str, bool]:
    """Validate all required model components"""
    required_files = [
        "classifier.joblib",
        "scaler.joblib", 
        "label_encoder.joblib",
        "model_info.json"
    ]
    
    validation_results = {}
    missing_files = []
    
    for file_name in required_files:
        file_path = os.path.join(model_dir, file_name)
        exists = os.path.exists(file_path)
        validation_results[file_name] = exists
        
        if not exists:
            missing_files.append(file_name)
    
    if missing_files:
        raise ModelLoadError(f"Missing model components: {missing_files}")
    
    return validation_results

@error_handler("prediction")
def safe_prediction(image_path: str, model_components: Dict) -> Dict:
    """Safe prediction with comprehensive error handling"""
    
    # Validate input
    validate_image_file(image_path)
    
    # Load and process image
    import cv2
    import numpy as np
    
    image = cv2.imread(image_path)
    if image is None:
        raise ImageProcessingError(f"Failed to load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized.astype(np.float32) / 255.0
    features = image_normalized.flatten()
    
    # Make prediction
    try:
        features_scaled = model_components["scaler"].transform(features.reshape(1, -1))
        prediction = model_components["classifier"].predict(features_scaled)[0]
        probabilities = model_components["classifier"].predict_proba(features_scaled)[0]
        predicted_class = model_components["label_encoder"].inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        return {
            "status": "success",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities.tolist()
        }
        
    except Exception as e:
        raise PredictionError(f"Prediction failed: {str(e)}")

def log_system_health():
    """Log current system health"""
    try:
        import psutil
        
        health_info = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
        
        logger.info(f"System Health: {health_info}")
        return health_info
        
    except Exception as e:
        logger.error(f"Failed to log system health: {str(e)}")
        return None
