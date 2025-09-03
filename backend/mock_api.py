"""
Simple Mock API server for testing the frontend connection
"""
import time
import json
import random
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="Amulet-AI Mock API",
    description="Simple mock API for testing frontend connection",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🏺 Amulet-AI Mock API Server",
        "status": "ready",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "api_version": "1.0.0",
        "ai_service_available": True,
        "ai_status": {"status": "mock"},
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.post("/predict")
async def predict(
    front: UploadFile = File(..., description="Front image of the amulet"),
    back: Optional[UploadFile] = File(None, description="Back image of the amulet (optional)")
):
    """Mock predict endpoint that simulates the real API response"""
    start_time = time.time()
    
    # Simulate processing time
    processing_delay = random.uniform(0.5, 1.5)
    time.sleep(processing_delay)
    
    # Realistic Thai Amulet Classes
    amulet_classes = [
        "somdej-fatherguay",
        "พระพุทธเจ้าในวิหาร",
        "พระสมเด็จฐานสิงห์", 
        "พระสมเด็จประทานพร พุทธกวัก",
        "พระสมเด็จหลังรูปเหมือน",
        "พระสรรค์",
        "พระสิวลี",
        "สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ",
        "สมเด็จแหวกม่าน",
        "ออกวัดหนองอีดุก"
    ]
    
    # Generate main confidence
    main_confidence = random.uniform(0.75, 0.92)
    main_class = random.choice(amulet_classes)
    
    # Generate top-k predictions
    topk_predictions = []
    remaining_confidence = 1.0 - main_confidence
    
    # Top 1
    topk_predictions.append({
        "class_id": 0,
        "class_name": main_class,
        "confidence": main_confidence
    })
    
    # Get other classes for top-k
    other_classes = [c for c in amulet_classes if c != main_class]
    random.shuffle(other_classes)
    
    second_conf = remaining_confidence * random.uniform(0.4, 0.7)
    third_conf = (remaining_confidence - second_conf) * random.uniform(0.3, 0.8)
    
    topk_predictions.append({
        "class_id": 1,
        "class_name": other_classes[0],
        "confidence": second_conf
    })
    
    topk_predictions.append({
        "class_id": 2,
        "class_name": other_classes[1],
        "confidence": third_conf
    })
    
    # Generate mock valuation
    valuation = {
        "p05": random.randint(5000, 15000),
        "p50": random.randint(15000, 50000),
        "p95": random.randint(50000, 150000),
        "confidence": random.choice(["low", "medium", "high"]),
        "condition_factor": random.uniform(0.8, 1.2)
    }
    
    # Generate mock recommendations
    recommendations = [
        {
            "market": "ตลาดพระเครื่องเวปไทย",
            "distance": 0,  # Online
            "rating": 4.5,
            "reason": "ตลาดออนไลน์ที่ใหญ่ที่สุดสำหรับพระเครื่องไทย มีผู้ซื้อผู้ขายจำนวนมาก"
        },
        {
            "market": "ตลาดนัดพระเครื่องพันทิป",
            "distance": 0,  # Online
            "rating": 4.0,
            "reason": "ชุมชนพระเครื่องออนไลน์ที่มีการอภิปรายและแลกเปลี่ยนความคิดเห็น"
        },
        {
            "market": "ตลาดพระเครื่องจตุจักร",
            "distance": 10.5,
            "rating": 4.8,
            "reason": "ตลาดพระเครื่องที่ใหญ่ที่สุดในกรุงเทพฯ ที่คุณสามารถพบพระเครื่องได้หลากหลาย"
        }
    ]
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Prepare comprehensive response
    response = {
        "top1": topk_predictions[0],
        "topk": topk_predictions,
        "valuation": valuation,
        "recommendations": recommendations,
        "ai_mode": "mock_data",
        "processing_time": processing_time,
        "image_quality": "good",
        "timestamp": datetime.now().isoformat()
    }
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
