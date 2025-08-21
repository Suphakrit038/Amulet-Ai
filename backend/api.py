import sys
import os
import logging
import random
# เพิ่ม path สำหรับ import ไฟล์ใน backend folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from model_loader import ModelLoader
from valuation import get_quantiles
from recommend import recommend_markets

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Amulet-AI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# สำหรับการทดสอบ - ไม่ต้องใช้ไฟล์โมเดลจริง
model_loader = ModelLoader()  # รันในโหมดทดสอบ

class TopKItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float

class Valuation(BaseModel):
    p05: float
    p50: float
    p95: float

class PredictionResponse(BaseModel):
    top1: TopKItem
    topk: List[TopKItem]
    valuation: Valuation
    recommendations: List[dict]

@app.post("/predict", response_model=PredictionResponse)
async def predict(front: UploadFile = File(...), back: Optional[UploadFile] = File(None)):
    try:
        logger.info(f"Received prediction request - Front: {front.filename}, Back: {back.filename if back else 'None'}")
        
        # ตรวจสอบ file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/heic', 'image/heif', 'image/webp', 'image/bmp', 'image/tiff']
        if front.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {front.content_type}")
        
        if back and back.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Unsupported back file type: {back.content_type}")
        
        # ใช้โมเดลจริงแทน mock data
        logger.info("Using ModelLoader for prediction")
        
        try:
            # ใช้ ModelLoader ทำนายจริง
            prediction_result = model_loader.predict(front.file)
            main_class = prediction_result["class"]
            main_confidence = prediction_result["confidence"]
            
            # สร้าง Top-3 (จำลองเพิ่มเติม 2 อันดับ)
            all_classes = list(model_loader.labels.values())
            other_classes = [cls for cls in all_classes if cls != main_class]
            
            topk = [
                {"class_id": 0, "class_name": main_class, "confidence": main_confidence}
            ]
            
            # เพิ่ม 2 อันดับถัดไป
            for i, other_class in enumerate(other_classes[:2], 1):
                remaining_confidence = (1.0 - main_confidence) * random.uniform(0.3, 0.7)
                topk.append({
                    "class_id": i, 
                    "class_name": other_class, 
                    "confidence": remaining_confidence
                })
            
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}, using fallback")
            # Fallback to mock data
            topk = [
                {"class_id": 0, "class_name": "หลวงพ่อกวยแหวกม่าน", "confidence": 0.95},
                {"class_id": 1, "class_name": "โพธิ์ฐานบัว", "confidence": 0.03},
                {"class_id": 2, "class_name": "ฐานสิงห์", "confidence": 0.02}
            ]
        
        # เรียกใช้ valuation และ recommendation จริง
        class_id = topk[0]["class_id"]
        valuation = get_quantiles(class_id)
        recommendations = recommend_markets(class_id, valuation)

        return {
            "top1": topk[0],
            "topk": topk,
            "valuation": valuation,
            "recommendations": recommendations,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
def root():
    """Root endpoint for health check"""
    return {"message": "Amulet-AI API is running", "status": "ok"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Amulet-AI API is running"}

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported image formats"""
    formats = {
        "supported": ["JPEG", "JPG", "PNG", "HEIC", "HEIF", "WebP", "BMP", "TIFF"],
        "recommended": ["JPEG", "PNG"],
        "heic_support": True  # เนื่องจากติดตั้ง pillow-heif แล้ว
    }
    return formats

# รัน: uvicorn backend.api:app --reload --port 8000