#!/usr/bin/env python3
"""
Production-Ready API for Amulet-AI
API ที่พร้อมใช้งานจริงสำหรับการจำแนกพระเครื่อง
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from pathlib import Path
import io
from PIL import Image
import sys
import os

# เพิ่ม path สำหรับ import
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_models.optimized_model import OptimizedAmuletModel
import json
from datetime import datetime
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Amulet-AI Production API",
    description="Production-ready API for Thai Amulet Classification",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ใน production ควรระบุ domain ที่แน่นอน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
model_loaded = False

class AmuletAPI:
    """API class สำหรับจัดการการทำนาย"""
    
    def __init__(self):
        self.model = OptimizedAmuletModel()
        self.model_path = Path(__file__).parent.parent.parent / "trained_model_optimized"
        self.temp_upload_dir = Path(__file__).parent.parent.parent / "temp_uploads"
        self.temp_upload_dir.mkdir(exist_ok=True)
        
    def load_model(self):
        """โหลดโมเดลที่เทรนแล้ว"""
        try:
            if self.model_path.exists():
                success = self.model.load_model(self.model_path)
                if success:
                    logger.info("✅ Model loaded successfully")
                    return True
                else:
                    logger.error("❌ Failed to load model")
                    return False
            else:
                logger.error(f"❌ Model directory not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_bytes):
        """เตรียมรูปภาพสำหรับการทำนาย"""
        try:
            # สร้างโฟลเดอร์ temp ถ้ายังไม่มี
            self.temp_upload_dir.mkdir(exist_ok=True)
            
            # แปลงจาก bytes เป็น numpy array
            image = Image.open(io.BytesIO(image_bytes))
            
            # แปลงเป็น RGB ถ้าจำเป็น
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # แปลงเป็น OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # บันทึกเป็นไฟล์ชั่วคราว
            temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            temp_path = self.temp_upload_dir / temp_filename
            
            success = cv2.imwrite(str(temp_path), opencv_image)
            if not success:
                logger.error("Failed to save temporary image")
                return None
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def cleanup_temp_file(self, file_path):
        """ลบไฟล์ชั่วคราว"""
        try:
            if file_path and file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Error cleaning up temp file: {e}")
    
    def predict_amulet(self, image_bytes):
        """ทำนายประเภทพระเครื่อง"""
        temp_path = None
        try:
            # เตรียมรูปภาพ
            temp_path = self.preprocess_image(image_bytes)
            if temp_path is None:
                return None
            
            # ทำนาย
            prediction, confidence = self.model.predict(temp_path)
            
            if prediction is None:
                return None
            
            # แปลชื่อเป็นภาษาไทย
            thai_names = {
                'phra_nang_phya': 'พระนางพญา',
                'phra_rod': 'พระร็อด',
                'phra_somdej': 'พระสมเด็จ'
            }
            
            # สร้างผลลัพธ์
            result = {
                'predicted_class': prediction,
                'thai_name': thai_names.get(prediction, prediction),
                'confidence': float(confidence),
                'confidence_percentage': f"{confidence * 100:.1f}%",
                'timestamp': datetime.now().isoformat(),
                'model_version': '2.0.0'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
        finally:
            # ลบไฟล์ชั่วคราว
            self.cleanup_temp_file(temp_path)

# สร้าง API instance
api_instance = AmuletAPI()

@app.on_event("startup")
async def startup_event():
    """เหตุการณ์เมื่อเริ่มต้น API"""
    global model_loaded
    logger.info("🚀 Starting Amulet-AI Production API...")
    
    # โหลดโมเดล
    model_loaded = api_instance.load_model()
    
    if model_loaded:
        logger.info("✅ API ready for predictions")
    else:
        logger.warning("⚠️ API started but model not loaded")

@app.get("/")
async def root():
    """หน้าแรกของ API"""
    return {
        "message": "Amulet-AI Production API",
        "version": "2.0.0",
        "status": "ready" if model_loaded else "model_not_loaded",
        "description": "Thai Amulet Classification API"
    }

@app.get("/health")
async def health_check():
    """ตรวจสอบสถานะของ API"""
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_amulet(file: UploadFile = File(...)):
    """ทำนายประเภทพระเครื่องจากรูปภาพ"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # ตรวจสอบประเภทไฟล์
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # อ่านไฟล์
        image_bytes = await file.read()
        
        # ทำนาย
        result = api_instance.predict_amulet(image_bytes)
        
        if result is None:
            raise HTTPException(status_code=400, detail="Could not process image")
        
        # เพิ่มข้อมูลไฟล์
        result['file_info'] = {
            'filename': file.filename,
            'content_type': file.content_type,
            'size_bytes': len(image_bytes)
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/classes")
async def get_classes():
    """ดูรายการ classes ที่รองรับ"""
    classes_info = {
        'phra_nang_phya': {
            'english': 'phra_nang_phya',
            'thai': 'พระนางพญา',
            'description': 'Thai amulet type: Phra Nang Phaya'
        },
        'phra_rod': {
            'english': 'phra_rod',
            'thai': 'พระร็อด',
            'description': 'Thai amulet type: Phra Rod'
        },
        'phra_somdej': {
            'english': 'phra_somdej',
            'thai': 'พระสมเด็จ',
            'description': 'Thai amulet type: Phra Somdej'
        }
    }
    
    return {
        'total_classes': len(classes_info),
        'classes': classes_info,
        'model_version': '2.0.0'
    }

@app.get("/model-info")
async def get_model_info():
    """ดูข้อมูลโมเดล"""
    if not model_loaded:
        return {"status": "Model not loaded"}
    
    try:
        # อ่านข้อมูล config ของโมเดล
        config_path = api_instance.model_path / 'model_config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        else:
            return {"status": "Model config not found"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # รัน API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # ปิด reload ใน production
        log_level="info"
    )