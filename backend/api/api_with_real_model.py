"""
🏺 Amulet-AI Backend with Real Trained Model
API ที่ใช้ AI Model จริงแทน Mock Data
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import time
from datetime import datetime
import logging
import os
import sys
from pathlib import Path

# เพิ่ม backend path เพื่อ import modules โดยอ้างอิงจาก root project
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

try:
    # นำเข้าจากโมดูล models
    from models.real_model_loader import AmuletModelLoader
    # สร้าง global instance
    model_loader = AmuletModelLoader()
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    # Fallback แบบง่าย
    class DummyModelLoader:
        def __init__(self):
            self.model = None
            self.class_names = ["dummy_class"]
            self.device = "cpu"
        def initialize(self): return False
        def get_model_info(self): return {"error": "Model not loaded"}
        def predict_image(self, image_bytes): return {"success": False, "error": "No model"}
    model_loader = DummyModelLoader()

app = FastAPI(
    title="Amulet-AI with Real Trained Model", 
    description="Backend API ที่ใช้ AI Model จริงที่เทรนแล้ว",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ข้อมูลราคาประเมิน (ปรับปรุงจากข้อมูลจริง)
PRICE_DATA = {
    "somdej_fatherguay": {"min": 15000, "avg": 45000, "max": 150000},
    "somdej_portrait_back": {"min": 18000, "avg": 55000, "max": 180000},
    "somdej_prok_bodhi": {"min": 25000, "avg": 75000, "max": 250000},
    "somdej_waek_man": {"min": 20000, "avg": 60000, "max": 200000},
    "wat_nong_e_duk": {"min": 8000, "avg": 22000, "max": 70000},
    "wat_nong_e_duk_misc": {"min": 5000, "avg": 15000, "max": 50000},
}

def estimate_price(class_name: str, confidence: float):
    """ประเมินราคาตาม class และ confidence"""
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

def generate_recommendations(class_name: str, price_range: dict):
    """สร้างคำแนะนำตลาดขาย"""
    recommendations = []
    
    # คำแนะนำตามประเภทพระเครื่อง
    if "somdej" in class_name.lower():
        recommendations = [
            {
                "market": "ตลาดพระเครื่องออนไลน์",
                "distance": 0,
                "rating": 4.5,
                "reason": f"เหมาะสำหรับขาย{class_name} มีผู้สะสมเฉพาะทาง"
            },
            {
                "market": "ตลาดพระจตุจักร ซ.26",
                "distance": 12,
                "rating": 4.7,
                "reason": "ตลาดพระที่มีชื่อเสียง ผู้ซื้อมีความรู้สูง"
            },
            {
                "market": "ตลาดพระเครื่องบางแค",
                "distance": 18,
                "rating": 4.3,
                "reason": "ร้านค้าที่เชี่ยวชาญพระสมเด็จโดยเฉพาะ"
            }
        ]
    elif "nong_e_duk" in class_name:
        recommendations = [
            {
                "market": "ศูนย์การค้าพระเครื่อง MBK",
                "distance": 8,
                "rating": 4.2,
                "reason": "ศูนย์กลางการค้าพระเครื่อง มีลูกค้าหลากหลาย"
            },
            {
                "market": "ตลาดพระออนไลน์",
                "distance": 0,
                "rating": 4.4,
                "reason": f"เหมาะสำหรับ{class_name} การขายออนไลน์"
            },
            {
                "market": "ตลาดนัดเสาร์-อาทิตย์ จตุจักร",
                "distance": 15,
                "rating": 4.6,
                "reason": "ผู้ซื้อจำนวนมาก โอกาสขายได้ดี"
            }
        ]
    else:
        # คำแนะนำทั่วไป
        recommendations = [
            {
                "market": "แพลตฟอร์มออนไลน์",
                "distance": 0,
                "rating": 4.3,
                "reason": "ขายได้ทั่วประเทศ เข้าถึงลูกค้าได้มาก"
            },
            {
                "market": "ตลาดพราหมณ์",
                "distance": 10,
                "rating": 4.1,
                "reason": "ตลาดท้องถิ่น ผู้ซื้อรู้จักของดี"
            },
            {
                "market": "ร้านพระเครื่องย่านสีลม",
                "distance": 12,
                "rating": 4.4,
                "reason": "ร้านค้าที่มีประสบการณ์และราคายุติธรรม"
            }
        ]
    
    return recommendations

@app.on_event("startup")
async def startup_event():
    """โหลด model เมื่อเซิร์ฟเวอร์เริ่มทำงาน"""
    print("🚀 เริ่มต้น Amulet-AI Backend with Real Model...")
    success = model_loader.initialize()
    if success:
        print("✅ Backend พร้อมใช้งานกับ Real AI Model!")
    else:
        print("⚠️ Backend เริ่มทำงานแต่ Model อาจมีปัญหา")

@app.get("/")
async def root():
    return {
        "message": "🏺 Amulet-AI Backend with Real Trained Model",
        "status": "running",
        "model_loaded": model_loader.model is not None,
        "classes": len(model_loader.class_names),
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    model_info = model_loader.get_model_info()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_service_available": model_loader.model is not None,
        "model_status": "loaded" if model_loader.model is not None else "not_loaded",
        "device": str(model_loader.device),
        "classes": model_loader.class_names,
        "num_classes": len(model_loader.class_names),
        "endpoints": ["/predict", "/model-info", "/health", "/docs"]
    }

@app.get("/model-info")
async def get_model_info():
    """ข้อมูลเกี่ยวกับ model ที่โหลด"""
    model_info = model_loader.get_model_info()
    return {
        **model_info,
        "price_categories": list(PRICE_DATA.keys()),
        "total_price_ranges": len(PRICE_DATA),
        "api_version": "2.0.0",
        "features": [
            "Real AI Model",
            "Price Estimation", 
            "Market Recommendations",
            "Multi-class Classification",
            "Confidence Scoring"
        ]
    }

@app.post("/predict")
async def predict(
    front: UploadFile = File(..., description="รูปภาพหน้าพระเครื่อง"),
    back: UploadFile = File(None, description="รูปภาพหลังพระเครื่อง (ไม่บังคับ)")
):
    """ทำนายด้วย Real AI Model ที่เทรนไว้"""
    start_time = time.time()
    
    if model_loader.model is None:
        raise HTTPException(
            status_code=503, 
            detail="AI Model ยังไม่ได้โหลด กรุณาลองใหม่อีกครั้ง"
        )
    
    try:
        # อ่านรูปภาพหน้า (หลัก)
        front_bytes = await front.read()
        
        print(f"📤 Processing: {front.filename} ({len(front_bytes)} bytes)")
        
        # ทำนายด้วย Real AI Model
        prediction_result = model_loader.predict_image(front_bytes)
        
        if not prediction_result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"ไม่สามารถประมวลผลรูปภาพได้: {prediction_result.get('error', 'Unknown error')}"
            )
        
        # ดึงผลลัพธ์หลัก
        top1 = prediction_result["top1"]
        predictions = prediction_result["predictions"]
        model_info = prediction_result.get("model_info", {})
        
        # ประเมินราคา
        valuation = estimate_price(top1["class_name"], top1["confidence"])
        
        # สร้างคำแนะนำตลาด
        recommendations = generate_recommendations(top1["class_name"], valuation)
        
        processing_time = time.time() - start_time
        
        # สร้าง response ที่สมบูรณ์
        response = {
            "ai_mode": "real_trained_model",
            "model_info": {
                "name": "Amulet-AI Real Model",
                "architecture": model_info.get("architecture", "Deep Learning"),
                "classes": model_info.get("num_classes", len(model_loader.class_names)),
                "device": model_info.get("device", str(model_loader.device)),
                "training_data": "Thai Buddhist Amulet Dataset"
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
        
        print(f"✅ Prediction successful: {top1['class_name']} ({top1['confidence']:.2%}) in {processing_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}"
        )

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """ทำนายหลายรูปพร้อมกัน"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="สามารถอัปโหลดได้สูงสุด 10 รูป")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            file_bytes = await file.read()
            result = model_loader.predict_image(file_bytes)
            
            if result.get("success", False):
                results.append({
                    "index": i,
                    "filename": file.filename,
                    "prediction": result["top1"],
                    "confidence": result["top1"]["confidence"],
                    "success": True
                })
            else:
                results.append({
                    "index": i,
                    "filename": file.filename,
                    "error": result.get("error", "Unknown error"),
                    "success": False
                })
                
        except Exception as e:
            results.append({
                "index": i,
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    successful_predictions = sum(1 for r in results if r["success"])
    
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
    return {
        "classes": model_loader.class_names,
        "total": len(model_loader.class_names),
        "price_data_available": [cls for cls in model_loader.class_names if cls in PRICE_DATA]
    }

if __name__ == "__main__":
    import uvicorn
    print("🏺 เริ่มต้น Amulet-AI Backend with Real Model...")
    uvicorn.run(app, host="127.0.0.1", port=8001)
