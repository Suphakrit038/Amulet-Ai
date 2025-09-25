"""
Enhanced API with Reference Images for Amulet-AI
- Uses improved model for better accuracy
- Includes reference images in the response
- Provides detailed prediction information
"""

import os
import sys
import logging
import time
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks

# Try different import approaches for TensorFlow/Keras
try:
    import tensorflow as tf
    keras = tf.keras
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    USING_TF_KERAS = True
except ImportError:
    try:
        import keras
        from keras.models import load_model
        from keras.preprocessing.image import img_to_array
        USING_TF_KERAS = False
    except ImportError:
        logger.error("Could not import TensorFlow or Keras. Please install them.")
        raise ImportError("Could not import TensorFlow or Keras. Please install them.")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.path.join(PROJECT_ROOT, 'training_output', 'improved_model')
REFERENCE_IMAGES_PATH = os.path.join(PROJECT_ROOT, 'unified_dataset', 'reference_images')
IMAGE_SIZE = (224, 224)
TOP_K = 5

# Create FastAPI app
app = FastAPI(
    title="Amulet-AI Enhanced API",
    description="API for amulet classification with reference images",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and class indices
model = None
class_indices = {}
indices_to_class = {}

def load_tf_model():
    """Load the TensorFlow model and class indices"""
    global model, class_indices, indices_to_class
    
    # Find the latest model if MODEL_PATH is a directory
    if os.path.isdir(MODEL_PATH):
        # Get the latest model directory
        model_dirs = sorted([d for d in os.listdir(MODEL_PATH) if os.path.isdir(os.path.join(MODEL_PATH, d))], reverse=True)
        if not model_dirs:
            raise FileNotFoundError(f"No model directories found in {MODEL_PATH}")
        
        latest_dir = os.path.join(MODEL_PATH, model_dirs[0])
        model_file = os.path.join(latest_dir, 'models', 'final_model.h5')
        class_indices_file = os.path.join(latest_dir, 'class_indices.json')
    else:
        # Assume MODEL_PATH is the direct path to the model file
        model_file = MODEL_PATH
        class_indices_file = os.path.join(os.path.dirname(model_file), 'class_indices.json')
    
    # Load the model
    if not os.path.exists(model_file):
        # Try to find any .h5 model in the training_output directory
        alternative_models = []
        for root, dirs, files in os.walk(os.path.join(PROJECT_ROOT, 'training_output')):
            for file in files:
                if file.endswith('.h5'):
                    alternative_models.append(os.path.join(root, file))
        
        if alternative_models:
            model_file = alternative_models[0]
            logger.warning(f"Using alternative model: {model_file}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_file} and no alternatives found")
    
    logger.info(f"Loading model from {model_file}")
    
    # Try different methods for loading the model based on what's available
    try:
        logger.info("Attempting to load model with TensorFlow/Keras...")
        try:
            # Try using custom loader from tools if available
            sys.path.append(os.path.join(PROJECT_ROOT, 'tools'))
            try:
                from keras_compatible_loader import load_model_compatible
                model = load_model_compatible(model_file)
                logger.info("Loaded model using compatible loader")
            except ImportError:
                # Fall back to standard loaders
                model = load_model(model_file)
                logger.info("Loaded model using standard loader")
        except Exception as e:
            logger.warning(f"Standard load_model failed: {e}")
            # Try alternate loading method
            try:
                import tensorflow as tf
                model = tf.saved_model.load(model_file)
                logger.info("Loaded model using tf.saved_model.load")
            except Exception as e2:
                logger.error(f"Could not load model with tf.saved_model.load: {e2}")
                raise e
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Load class indices
    if os.path.exists(class_indices_file):
        with open(class_indices_file, 'r') as f:
            class_info = json.load(f)
        class_indices = class_info['class_indices']
        indices_to_class = class_info['indices_to_class']
    else:
        # Try to find class_indices.json anywhere in the project
        alternative_indices = []
        for root, dirs, files in os.walk(PROJECT_ROOT):
            for file in files:
                if file == 'class_indices.json':
                    alternative_indices.append(os.path.join(root, file))
        
        if alternative_indices:
            with open(alternative_indices[0], 'r') as f:
                class_info = json.load(f)
            class_indices = class_info['class_indices']
            indices_to_class = class_info['indices_to_class']
            logger.warning(f"Using alternative class indices: {alternative_indices[0]}")
        else:
            # If no class indices file is found, try to use labels.json
            labels_file = os.path.join(PROJECT_ROOT, 'unified_dataset', 'labels.json')
            if os.path.exists(labels_file):
                with open(labels_file, 'r') as f:
                    labels = json.load(f)
                class_indices = {label: i for i, label in enumerate(labels)}
                indices_to_class = {str(i): label for i, label in enumerate(labels)}
                logger.warning(f"Using labels.json for class indices")
            else:
                logger.warning("No class indices file found, using empty dictionary")
    
    logger.info(f"Loaded model with {len(class_indices)} classes")
    return model

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess an image for model prediction"""
    img = img.resize(IMAGE_SIZE)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

def get_reference_images(class_name: str, max_images: int = 3) -> Dict[str, Dict[str, str]]:
    """Get reference images for a class"""
    reference_images = {}
    
    # Check if reference images directory exists
    ref_dir = os.path.join(REFERENCE_IMAGES_PATH, class_name)
    if not os.path.exists(ref_dir):
        logger.warning(f"No reference images found for class {class_name}")
        return reference_images
    
    # Get image files
    try:
        image_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Prioritize front and back views
        front_images = [f for f in image_files if f.startswith('front_')]
        back_images = [f for f in image_files if f.startswith('back_')]
        other_images = [f for f in image_files if not f.startswith('front_') and not f.startswith('back_')]
        
        # Sort images to prioritize front and back views
        sorted_images = front_images + back_images + other_images
        
        # Process images
        count = 0
        for img_file in sorted_images:
            if count >= max_images:
                break
            
            img_path = os.path.join(ref_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    # Resize image to a reasonable size
                    img = img.resize((300, 300), Image.LANCZOS)
                    
                    # Convert to base64
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Determine view type
                    view_type = "front" if "front" in img_file.lower() else "back" if "back" in img_file.lower() else "other"
                    
                    # Add to reference images
                    reference_images[f"ref_{count+1}"] = {
                        "image_b64": img_str,
                        "view_type": view_type,
                        "file_name": img_file
                    }
                    
                    count += 1
            except Exception as e:
                logger.error(f"Error processing reference image {img_path}: {str(e)}")
        
        logger.info(f"Found {count} reference images for class {class_name}")
    except Exception as e:
        logger.error(f"Error getting reference images for class {class_name}: {str(e)}")
    
    return reference_images

def predict_amulet(front_image: Image.Image, back_image: Image.Image) -> Dict[str, Any]:
    """Predict amulet class and return results with reference images"""
    # Ensure model is loaded
    if model is None:
        load_tf_model()
    
    # Start timing
    start_time = time.time()
    
    # Preprocess images
    front_array = preprocess_image(front_image)
    back_array = preprocess_image(back_image)
    
    # Combined prediction (average of front and back)
    front_pred = model.predict(front_array)
    back_pred = model.predict(back_array)
    combined_pred = (front_pred + back_pred) / 2.0
    
    # Get top K predictions
    top_indices = combined_pred[0].argsort()[-TOP_K:][::-1]
    top_predictions = []
    
    for idx in top_indices:
        class_name = indices_to_class.get(str(idx), f"Unknown ({idx})")
        confidence = float(combined_pred[0][idx])
        top_predictions.append({
            "class_name": class_name,
            "confidence": confidence
        })
    
    # Get reference images for top prediction
    top_class = top_predictions[0]["class_name"]
    reference_images = get_reference_images(top_class)
    
    # Mock valuation (replace with actual valuation logic)
    valuation = {
        "p05": 1000,
        "p50": 5000,
        "p95": 20000,
        "confidence": "medium",
        "notes": "ราคาประเมินนี้เป็นเพียงการคาดการณ์เบื้องต้น ขึ้นอยู่กับความนิยม สภาพ และปัจจัยอื่นๆ"
    }
    
    # Mock recommendations (replace with actual recommendation logic)
    recommendations = [
        {
            "market": "Facebook Marketplace",
            "rating": 4.5,
            "distance": 0,
            "reason": "แพลตฟอร์มยอดนิยมสำหรับการซื้อขายพระเครื่อง มีผู้ซื้อจำนวนมาก"
        },
        {
            "market": "ตลาดพระเครื่องท่าพระจันทร์",
            "rating": 4.2,
            "distance": 12,
            "reason": "ตลาดพระเครื่องที่ใหญ่ที่สุดในกรุงเทพฯ เหมาะสำหรับการซื้อขายพระเครื่องราคาสูง"
        }
    ]
    
    # End timing
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Create result
    result = {
        "ai_mode": "production",
        "processing_time": processing_time,
        "top1": top_predictions[0],
        "topk": top_predictions,
        "reference_images": reference_images,
        "valuation": valuation,
        "recommendations": recommendations,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return result

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@app.post("/predict")
async def predict(front: UploadFile = File(...), back: UploadFile = File(...)):
    """Predict amulet class from front and back images"""
    try:
        # Validate files
        if not front or not back:
            raise HTTPException(status_code=400, detail="Both front and back images are required")
        
        # Read and validate images
        try:
            front_image = Image.open(BytesIO(await front.read()))
            back_image = Image.open(BytesIO(await back.read()))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Make prediction
        result = predict_amulet(front_image, back_image)
        
        return result
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get available classes"""
    # Ensure model is loaded
    if model is None:
        load_tf_model()
    
    return {"classes": list(class_indices.keys())}

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_tf_model()
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)

# Main function for directly running the API
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8001))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run("api_with_reference_images:app", host=host, port=port, reload=True)
