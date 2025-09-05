"""
Enhanced FastAPI Backend for Amulet-AI
รองรับระบบ AI ใหม่และการตรวจสอบสุขภาพ
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import modern AI model if available
try:
    from ai_models.modern_model import ModernAmuletModel
    MODERN_MODEL_AVAILABLE = True
except ImportError:
    MODERN_MODEL_AVAILABLE = False
    logging.warning("Modern AI model not available, using mock responses")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend/logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Amulet-AI API",
    description="ระบบ API สำหรับการวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None
class_names = [
    "somdej", "phra_rod", "phra_nang_phaya", "phra_pidta", 
    "phra_kong", "phra_chinnarat", "lp_tuad", "wat_rakang", 
    "wat_mahathat", "other"
]

# Thai class names mapping
thai_class_names = {
    "somdej": "สมเด็จ",
    "phra_rod": "พระรอด",
    "phra_nang_phaya": "พระนางพญา",
    "phra_pidta": "พระปิดตา",
    "phra_kong": "พระกลม",
    "phra_chinnarat": "พระพุทธชินราช",
    "lp_tuad": "หลวงปู่ทวด",
    "wat_rakang": "วัดระฆัง",
    "wat_mahathat": "วัดมหาธาตุ",
    "other": "อื่นๆ"
}


def load_model():
    """โหลดโมเดล AI"""
    global model, device
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        if MODERN_MODEL_AVAILABLE:
            # Load modern model
            model = ModernAmuletModel(
                num_classes=len(class_names),
                model_name="efficientnet_v2_s"
            )
            
            # Try to load saved weights
            model_path = project_root / "ai_models" / "saved_models" / "best_model.pth"
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=device))
                logger.info("Loaded trained model weights")
            else:
                logger.warning("No trained weights found, using pretrained model")
            
            model = model.to(device)
            model.eval()
            logger.info("Modern AI model loaded successfully")
        else:
            logger.warning("Using mock model - install AI dependencies for real predictions")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None


def preprocess_image(image: Image.Image):
    """ประมวลผลรูปภาพสำหรับโมเดล"""
    if not MODERN_MODEL_AVAILABLE or model is None:
        return None
    
    try:
        # ใช้ transforms ของโมเดลใหม่
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Convert RGBA to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = transform(image).unsqueeze(0).to(device)
        return tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None


def predict_amulet(image_tensor):
    """ทำนายประเภทพระเครื่อง"""
    if not MODERN_MODEL_AVAILABLE or model is None:
        # Mock prediction
        import random
        predicted_class = random.choice(class_names)
        confidence = random.uniform(0.6, 0.95)
        
        return {
            "prediction": predicted_class,
            "thai_name": thai_class_names[predicted_class],
            "confidence": confidence * 100,
            "all_predictions": [
                {
                    "class": predicted_class,
                    "thai_name": thai_class_names[predicted_class],
                    "confidence": confidence * 100
                }
            ]
        }
    
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(3, len(class_names)))
            
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                class_name = class_names[idx.item()]
                predictions.append({
                    "class": class_name,
                    "thai_name": thai_class_names[class_name],
                    "confidence": prob.item() * 100
                })
            
            # Return best prediction
            best_prediction = predictions[0]
            
            return {
                "prediction": best_prediction["class"],
                "thai_name": best_prediction["thai_name"],
                "confidence": best_prediction["confidence"],
                "all_predictions": predictions
            }
            
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """เริ่มต้นระบบ"""
    logger.info("Starting Amulet-AI API...")
    
    # Create logs directory
    log_dir = Path("backend/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load AI model
    load_model()
    
    logger.info("Amulet-AI API started successfully!")


@app.get("/")
async def root():
    """หน้าหลัก API"""
    return {
        "message": "Amulet-AI API",
        "version": "2.0.0",
        "description": "ระบบ API สำหรับการวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์",
        "endpoints": {
            "/health": "ตรวจสอบสุขภาพระบบ",
            "/predict": "วิเคราะห์รูปภาพพระเครื่อง",
            "/models/info": "ข้อมูลโมเดล AI"
        }
    }


@app.get("/health")
async def health_check():
    """ตรวจสอบสุขภาพระบบ"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "model_loaded": model is not None,
        "device": str(device) if device else "none",
        "modern_model_available": MODERN_MODEL_AVAILABLE,
        "uptime": time.time()
    }
    
    return status


@app.get("/models/info")
async def model_info():
    """ข้อมูลโมเดล AI"""
    if not MODERN_MODEL_AVAILABLE or model is None:
        return {
            "model_type": "mock",
            "description": "Mock model for testing",
            "classes": class_names,
            "thai_names": thai_class_names
        }
    
    return {
        "model_type": "modern_ai",
        "architecture": "EfficientNetV2 + Vision Transformer",
        "classes": class_names,
        "thai_names": thai_class_names,
        "device": str(device),
        "num_parameters": sum(p.numel() for p in model.parameters())
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """วิเคราะห์รูปภาพพระเครื่อง"""
    start_time = time.time()
    
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        logger.info(f"Processing image: {file.filename}, size: {image.size}")
        
        # Preprocess for model
        image_tensor = preprocess_image(image)
        
        # Make prediction
        result = predict_amulet(image_tensor)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add metadata
        result.update({
            "processing_time": round(processing_time, 3),
            "image_size": f"{image.size[0]}x{image.size[1]}",
            "model_type": "modern_ai" if MODERN_MODEL_AVAILABLE else "mock",
            "timestamp": datetime.now().isoformat()
        })
        
        # Add recommendations based on confidence
        confidence = result["confidence"]
        if confidence > 85:
            result["recommendation"] = "ผลการวิเคราะห์มีความน่าเชื่อถือสูง"
        elif confidence > 70:
            result["recommendation"] = "ผลการวิเคราะห์มีความน่าเชื่อถือปานกลาง ควรตรวจสอบเพิ่มเติม"
        else:
            result["recommendation"] = "ผลการวิเคราะห์มีความน่าเชื่อถือต่ำ แนะนำให้ถ่ายรูปใหม่หรือปรึกษาผู้เชี่ยวชาญ"
        
        logger.info(f"Prediction completed: {result['prediction']} ({confidence:.1f}%) in {processing_time:.3f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/stats")
async def get_stats():
    """ส統ิ้อมูลการใช้งาน"""
    # This could be enhanced with actual database tracking
    return {
        "total_predictions": "N/A",
        "average_confidence": "N/A",
        "most_common_class": "N/A",
        "uptime": time.time(),
        "note": "Stats tracking not implemented yet"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Amulet-AI API server...")
    uvicorn.run(
        "api_with_real_model:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
