"""
🏺 Amulet-AI Integrated API System
ระบบ API รวมสำหรับ Amulet-AI ที่รองรับทั้ง Mock Data และ Real AI Model
"""
import os
import sys
import time
import logging
import asyncio
import random
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

# ตั้งค่า path เพื่อนำเข้าโมดูลได้ถูกต้อง
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# นำเข้า FastAPI components
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
    logging.info("✅ FastAPI imported successfully")
except ImportError as e:
    logging.warning(f"⚠️ FastAPI not available: {e}")
    logging.warning("Install with: pip install fastapi uvicorn python-multipart")
    FASTAPI_AVAILABLE = False
    # สร้าง mock classes สำหรับการพัฒนา
    class FastAPI:
        def __init__(self, *args, **kwargs): pass
        def get(self, *args, **kwargs): return lambda func: func
        def post(self, *args, **kwargs): return lambda func: func
        def add_middleware(self, *args, **kwargs): pass
    class UploadFile: pass
    class File: pass
    class HTTPException(Exception): pass
    class BackgroundTasks: pass
    class Depends: pass
    class Request: pass
    class CORSMiddleware: pass
    class GZipMiddleware: pass
    class JSONResponse: pass
    class BaseModel: pass
    Field = lambda *args, **kwargs: None

# นำเข้า Real Model Loader
try:
    from models.real_model_loader import AmuletModelLoader
    REAL_MODEL_AVAILABLE = True
    model_loader = AmuletModelLoader()
    logging.info("✅ Real Model Loader imported successfully")
except ImportError as e:
    logging.warning(f"⚠️ Real Model Loader not available: {e}")
    REAL_MODEL_AVAILABLE = False
    # สร้าง mock class
    class DummyModelLoader:
        def __init__(self):
            self.model = None
            self.class_names = ["dummy_class"]
            self.device = "cpu"
        def initialize(self): return False
        def get_model_info(self): return {"error": "Model not loaded"}
        def predict_image(self, image_bytes): return {"success": False, "error": "No model"}
    model_loader = DummyModelLoader()

# นำเข้าโมดูลสำหรับการประเมินราคาและคำแนะนำ
try:
    from services.valuation import get_optimized_valuation, get_quantiles
    from services.recommend import get_optimized_recommendations
    from config.app_config import get_config
    SERVICES_AVAILABLE = True
    logging.info("✅ Valuation and Recommendation services imported successfully")
except ImportError as e:
    logging.warning(f"⚠️ Services import error: {e}")
    SERVICES_AVAILABLE = False
    # ฟังก์ชันจำลอง
    async def get_optimized_valuation(class_name, confidence, condition="good"):
        return {
            "p05": 5000, "p50": 25000, "p95": 75000,
            "confidence": "medium" if confidence > 0.5 else "low"
        }
    
    def get_quantiles(class_id):
        return {
            "p05": 5000, "p50": 25000, "p95": 75000,
            "confidence": "medium", "class_name": "unknown"
        }
    
    async def get_optimized_recommendations(class_name, valuation):
        return [
            {"name": "ตลาดพระเครื่องออนไลน์", "distance": 0, "specialty": "พระเครื่องทั่วไป", "rating": 4.5},
            {"name": "ตลาดนัดพระ", "distance": 10, "specialty": "พระเก่า", "rating": 4.2}
        ]
    
    def get_config():
        return type('Config', (), {'debug': True, 'api': type('APIConfig', (), {'cors_origins': ["*"]})})()

# ข้อมูลราคาประเมินเบื้องต้น (ใช้เมื่อไม่มี service)
PRICE_DATA = {
    "somdej_fatherguay": {"min": 15000, "avg": 45000, "max": 150000},
    "somdej_portrait_back": {"min": 18000, "avg": 55000, "max": 180000},
    "somdej_prok_bodhi": {"min": 25000, "avg": 75000, "max": 250000},
    "somdej_waek_man": {"min": 20000, "avg": 60000, "max": 200000},
    "somdej_lion_base": {"min": 12000, "avg": 35000, "max": 85000},
    "wat_nong_e_duk": {"min": 8000, "avg": 22000, "max": 70000},
    "wat_nong_e_duk_misc": {"min": 5000, "avg": 15000, "max": 50000},
}

# ตั้งค่า Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("amulet-api")

# สถิติการใช้งาน
request_stats = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "avg_response_time": 0.0,
    "uptime_start": time.time()
}

# Pydantic Models
class TopKItem(BaseModel):
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")

class Valuation(BaseModel):
    p05: float = Field(..., description="5th percentile price (THB)")
    p50: float = Field(..., description="Median price (THB)")
    p95: float = Field(..., description="95th percentile price (THB)")
    confidence: str = Field(..., description="Valuation confidence level")
    condition_factor: Optional[float] = Field(1.0, description="Condition adjustment factor")

class MarketRecommendation(BaseModel):
    name: str = Field(..., description="Market name")
    distance: float = Field(..., description="Distance in kilometers")
    specialty: str = Field(..., description="Market specialty")
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Market rating")
    estimated_price_range: Optional[str] = Field(None, description="Expected price range")

# App Lifecycle Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """การจัดการวงจรชีวิตของแอปพลิเคชัน"""
    # Startup
    logger.info("🚀 Starting Amulet-AI Integrated API Server...")
    
    # ลองโหลด Real AI Model ถ้ามี
    if REAL_MODEL_AVAILABLE:
        success = model_loader.initialize()
        if success:
            logger.info("✅ Real AI Model initialized successfully!")
        else:
            logger.warning("⚠️ Real AI Model initialization failed, will use simulated predictions")
    
    # App startup complete
    logger.info("🎉 API Server ready!")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Amulet-AI API Server...")
    logger.info("👋 Server shutdown complete")

# สร้าง FastAPI App
app = FastAPI(
    title="Amulet-AI Integrated API",
    description="Advanced Thai Buddhist Amulet Recognition and Valuation System",
    version="2.0.0",
    lifespan=lifespan,
)

# เพิ่ม Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Helper Functions
def estimate_price(class_name: str, confidence: float) -> Dict:
    """ประเมินราคาอย่างง่ายโดยใช้ข้อมูลพื้นฐาน"""
    # ถ้ามี service ให้ใช้ service
    if SERVICES_AVAILABLE:
        try:
            # ดึงข้อมูลแบบ synchronous (สำหรับใช้ในฟังก์ชัน synchronous)
            class_id = 0  # default
            # แปลง class_name เป็น class_id
            if "fatherguay" in class_name or "แหวกม่าน" in class_name:
                class_id = 0
            elif "prok_bodhi" in class_name or "ปรกโพธิ์" in class_name:
                class_id = 1
            elif "lion_base" in class_name or "ฐานสิงห์" in class_name:
                class_id = 2
            elif "nong_e_duk" in class_name or "วัดหนองอีดุก" in class_name:
                class_id = 3
            
            return get_quantiles(class_id)
        except Exception as e:
            logger.error(f"Error using service valuation: {e}")
            # ถ้าเกิดข้อผิดพลาด ให้ใช้วิธีง่ายๆ แทน
    
    # ถ้าไม่มี service หรือเกิดข้อผิดพลาด ให้ใช้วิธีง่ายๆ
    base_prices = PRICE_DATA.get(class_name, {"min": 5000, "avg": 15000, "max": 50000})
    
    # ปรับราคาตาม confidence
    confidence_factor = max(0.4, min(1.3, confidence * 1.2))  # 40% - 130%
    
    estimated_prices = {
        "p05": int(base_prices["min"] * confidence_factor),
        "p50": int(base_prices["avg"] * confidence_factor),
        "p95": int(base_prices["max"] * confidence_factor),
        "confidence": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
    }
    
    return estimated_prices

def generate_recommendations(class_name: str, price_range: dict) -> List[Dict]:
    """สร้างคำแนะนำตลาดขายแบบพื้นฐาน"""
    # ถ้ามี service ให้ใช้ service (แบบ async)
    if SERVICES_AVAILABLE:
        try:
            # สร้าง event loop ถ้าไม่มี (สำหรับเรียกใช้ async function ใน sync function)
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(get_optimized_recommendations(class_name, price_range))
        except Exception as e:
            logger.error(f"Error using recommendation service: {e}")
            # ถ้าเกิดข้อผิดพลาด ให้ใช้วิธีง่ายๆ แทน
    
    recommendations = []
    
    # คำแนะนำตามประเภทพระเครื่อง
    if "somdej" in class_name.lower():
        recommendations = [
            {
                "name": "ตลาดพระเครื่องออนไลน์",
                "distance": 0,
                "specialty": "พระหลากหลาย",
                "rating": 4.5,
                "estimated_price_range": f"฿{price_range['p05']:,} - ฿{price_range['p95']:,}"
            },
            {
                "name": "ตลาดพระจตุจักร ซ.26",
                "distance": 12,
                "specialty": "พระโบราณ",
                "rating": 4.7,
                "estimated_price_range": f"฿{int(price_range['p50'] * 0.9):,} - ฿{int(price_range['p95'] * 1.1):,}"
            },
            {
                "name": "ตลาดพระเครื่องบางแค",
                "distance": 18,
                "specialty": "พระสมเด็จโดยเฉพาะ",
                "rating": 4.3,
                "estimated_price_range": f"฿{int(price_range['p50'] * 0.8):,} - ฿{int(price_range['p95'] * 1.2):,}"
            }
        ]
    elif "nong_e_duk" in class_name:
        recommendations = [
            {
                "name": "ศูนย์การค้าพระเครื่อง MBK",
                "distance": 8,
                "specialty": "ศูนย์กลางการค้าพระเครื่อง",
                "rating": 4.2,
                "estimated_price_range": f"฿{int(price_range['p50'] * 0.9):,} - ฿{int(price_range['p95'] * 1.0):,}"
            },
            {
                "name": "ตลาดพระออนไลน์",
                "distance": 0,
                "specialty": "การขายออนไลน์",
                "rating": 4.4,
                "estimated_price_range": f"฿{int(price_range['p50'] * 0.8):,} - ฿{int(price_range['p95'] * 1.1):,}"
            },
            {
                "name": "ตลาดนัดเสาร์-อาทิตย์ จตุจักร",
                "distance": 15,
                "specialty": "ตลาดนัดทั่วไป",
                "rating": 4.6,
                "estimated_price_range": f"฿{int(price_range['p50'] * 1.0):,} - ฿{int(price_range['p95'] * 1.2):,}"
            }
        ]
    else:
        # คำแนะนำทั่วไป
        recommendations = [
            {
                "name": "แพลตฟอร์มออนไลน์",
                "distance": 0,
                "specialty": "ขายได้ทั่วประเทศ",
                "rating": 4.3,
                "estimated_price_range": f"฿{int(price_range['p50'] * 0.9):,} - ฿{int(price_range['p95'] * 1.0):,}"
            },
            {
                "name": "ตลาดพราหมณ์",
                "distance": 10,
                "specialty": "ตลาดท้องถิ่น",
                "rating": 4.1,
                "estimated_price_range": f"฿{int(price_range['p50'] * 0.8):,} - ฿{int(price_range['p95'] * 0.9):,}"
            },
            {
                "name": "ร้านพระเครื่องย่านสีลม",
                "distance": 12,
                "specialty": "ร้านค้าที่มีประสบการณ์",
                "rating": 4.4,
                "estimated_price_range": f"฿{int(price_range['p50'] * 1.0):,} - ฿{int(price_range['p95'] * 1.1):,}"
            }
        ]
    
    return recommendations

async def process_prediction(image_bytes, filename) -> Dict:
    """ประมวลผลการทำนาย โดยใช้ Real Model หรือ Mock Data"""
    if REAL_MODEL_AVAILABLE and model_loader.model is not None:
        # ใช้ Real AI Model
        prediction_result = model_loader.predict_image(image_bytes)
        
        if not prediction_result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"ไม่สามารถประมวลผลรูปภาพได้: {prediction_result.get('error', 'Unknown error')}"
            )
        
        # ดึงผลลัพธ์หลัก
        top1 = prediction_result["top1"]
        predictions = prediction_result["predictions"]
        
        return {
            "ai_mode": "real_trained_model",
            "top1": top1,
            "predictions": predictions,
            "metadata": prediction_result.get("metadata", {})
        }
    else:
        # ใช้ Mock Data
        # จำลองการทำนายด้วยข้อมูลปลอม
        class_names = [
            "somdej_fatherguay", 
            "somdej_portrait_back", 
            "somdej_prok_bodhi",
            "somdej_waek_man", 
            "somdej_lion_base",
            "wat_nong_e_duk", 
            "wat_nong_e_duk_misc"
        ]
        
        # สุ่มคลาสและความเชื่อมั่น
        main_class = random.choice(class_names)
        main_confidence = random.uniform(0.6, 0.95)
        
        # สร้าง top-k predictions
        predictions = [
            {"class_name": main_class, "confidence": main_confidence, "class_id": class_names.index(main_class)}
        ]
        
        # สุ่มคลาสอื่นๆ เพิ่ม
        remaining = 1.0 - main_confidence
        for i in range(4):
            other_class = random.choice([c for c in class_names if c != main_class])
            other_conf = remaining * (0.7 ** (i+1))
            predictions.append({
                "class_name": other_class, 
                "confidence": other_conf,
                "class_id": class_names.index(other_class)
            })
        
        return {
            "ai_mode": "simulated_prediction",
            "top1": predictions[0],
            "predictions": predictions,
            "metadata": {"simulation": True}
        }

def update_stats(success: bool, processing_time: float):
    """อัปเดตสถิติการใช้งาน"""
    request_stats["total_requests"] += 1
    if success:
        request_stats["successful_predictions"] += 1
    else:
        request_stats["failed_predictions"] += 1
    
    # อัปเดตเวลาเฉลี่ย
    total = request_stats["successful_predictions"] + request_stats["failed_predictions"]
    current_avg = request_stats["avg_response_time"]
    request_stats["avg_response_time"] = (current_avg * (total - 1) + processing_time) / total

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - แสดงข้อมูลทั่วไป"""
    return {
        "message": "🏺 Amulet-AI Integrated API",
        "status": "running",
        "model_loaded": REAL_MODEL_AVAILABLE and model_loader.model is not None,
        "services_available": SERVICES_AVAILABLE,
        "version": "2.0.0",
        "uptime": time.time() - request_stats["uptime_start"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_info = {}
    if REAL_MODEL_AVAILABLE:
        model_info = model_loader.get_model_info()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_service": {
            "available": REAL_MODEL_AVAILABLE,
            "model_loaded": REAL_MODEL_AVAILABLE and model_loader.model is not None,
            "device": model_info.get("device", "N/A"),
            "classes": len(model_info.get("classes", [])),
        },
        "valuation_service": {
            "available": SERVICES_AVAILABLE
        },
        "recommendation_service": {
            "available": SERVICES_AVAILABLE
        },
        "request_stats": {
            "total_requests": request_stats["total_requests"],
            "successful": request_stats["successful_predictions"],
            "failed": request_stats["failed_predictions"],
            "avg_response_time": request_stats["avg_response_time"]
        },
        "endpoints": ["/predict", "/model-info", "/health", "/system-status", "/docs"]
    }

@app.get("/model-info")
async def get_model_info():
    """ข้อมูลเกี่ยวกับ model และบริการ"""
    model_info = {}
    if REAL_MODEL_AVAILABLE:
        model_info = model_loader.get_model_info()
    
    return {
        **model_info,
        "price_categories": list(PRICE_DATA.keys()),
        "total_price_ranges": len(PRICE_DATA),
        "api_version": "2.0.0",
        "features": [
            "AI Model" if REAL_MODEL_AVAILABLE else "Simulated AI",
            "Price Estimation", 
            "Market Recommendations",
            "Multi-class Classification"
        ]
    }

@app.post("/predict")
async def predict(
    front: UploadFile = File(..., description="รูปภาพหน้าพระเครื่อง"),
    back: Optional[UploadFile] = File(None, description="รูปภาพหลังพระเครื่อง (ไม่บังคับ)")
):
    """ทำนายประเภทพระเครื่องจากรูปภาพ"""
    start_time = time.time()
    
    if REAL_MODEL_AVAILABLE and model_loader.model is None:
        logger.warning("AI Model ยังไม่ได้โหลด")
    
    try:
        # อ่านรูปภาพหน้า (หลัก)
        front_bytes = await front.read()
        logger.info(f"Processing: {front.filename} ({len(front_bytes)} bytes)")
        
        # ประมวลผลการทำนาย
        prediction_result = await process_prediction(front_bytes, front.filename)
        
        # ดึงผลลัพธ์หลัก
        top1 = prediction_result["top1"]
        predictions = prediction_result["predictions"]
        
        # ประเมินราคา
        valuation = estimate_price(top1["class_name"], top1["confidence"])
        
        # สร้างคำแนะนำตลาด
        recommendations = generate_recommendations(top1["class_name"], valuation)
        
        processing_time = time.time() - start_time
        
        # อัปเดตสถิติ
        update_stats(True, processing_time)
        
        # สร้าง response ที่สมบูรณ์
        response = {
            "ai_mode": prediction_result["ai_mode"],
            "model_info": {
                "name": "Amulet-AI Model" if REAL_MODEL_AVAILABLE else "Simulated Model",
                "real_model": REAL_MODEL_AVAILABLE and model_loader.model is not None,
            },
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "filename": front.filename,
                "size": len(front_bytes),
                "format": "image"
            },
            "top1": top1,
            "topk": predictions,
            "valuation": valuation,
            "recommendations": recommendations,
            "metadata": prediction_result.get("metadata", {})
        }
        
        logger.info(f"Prediction successful: {top1['class_name']} ({top1['confidence']:.2f}) in {processing_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Prediction error: {str(e)}")
        
        # อัปเดตสถิติ
        update_stats(False, processing_time)
        
        raise HTTPException(
            status_code=500, 
            detail=f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}"
        )

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """ทำนายหลายรูปพร้อมกัน"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="สามารถอัปโหลดได้สูงสุด 10 รูป")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            file_bytes = await file.read()
            prediction_result = await process_prediction(file_bytes, file.filename)
            
            # ประเมินราคาอย่างง่าย
            top1 = prediction_result["top1"]
            valuation = estimate_price(top1["class_name"], top1["confidence"])
            
            results.append({
                "index": i,
                "filename": file.filename,
                "prediction": top1,
                "valuation": valuation,
                "success": True
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    successful_predictions = sum(1 for r in results if r.get("success", False))
    
    return {
        "total_files": len(files),
        "successful_predictions": successful_predictions,
        "failed_predictions": len(files) - successful_predictions,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/classes")
async def get_classes():
    """ดึงรายชื่อ classes ทั้งหมด"""
    if REAL_MODEL_AVAILABLE and model_loader.model is not None:
        return {
            "classes": model_loader.class_names,
            "total": len(model_loader.class_names),
            "price_data_available": [cls for cls in model_loader.class_names if cls in PRICE_DATA]
        }
    else:
        # กรณีไม่มี model ให้ใช้รายชื่อจาก PRICE_DATA
        return {
            "classes": list(PRICE_DATA.keys()),
            "total": len(PRICE_DATA),
            "price_data_available": list(PRICE_DATA.keys())
        }

@app.get("/system-status")
async def get_system_status():
    """แสดงสถานะระบบโดยละเอียด"""
    uptime = time.time() - request_stats["uptime_start"]
    
    return {
        "status": "operational",
        "uptime": uptime,
        "uptime_formatted": f"{int(uptime // 86400)}d {int((uptime % 86400) // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
        "ai_mode": {
            "real_model_available": REAL_MODEL_AVAILABLE,
            "model_loaded": REAL_MODEL_AVAILABLE and model_loader.model is not None,
            "using_simulation": not (REAL_MODEL_AVAILABLE and model_loader.model is not None)
        },
        "services": {
            "valuation_available": SERVICES_AVAILABLE,
            "recommendation_available": SERVICES_AVAILABLE
        },
        "performance": {
            "total_requests": request_stats["total_requests"],
            "successful_predictions": request_stats["successful_predictions"],
            "failed_predictions": request_stats["failed_predictions"],
            "success_rate": request_stats["successful_predictions"] / max(1, request_stats["total_requests"]),
            "avg_response_time": request_stats["avg_response_time"]
        },
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal server error: {str(exc)}"}
    )

# Main function
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Amulet-AI Integrated API Server...")
    if not FASTAPI_AVAILABLE:
        logger.error("❌ FastAPI is not available. Please install it with: pip install fastapi uvicorn python-multipart")
        sys.exit(1)
    
    host = "127.0.0.1"
    port = 8000
    
    logger.info(f"🌐 Server will run at http://{host}:{port}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)
