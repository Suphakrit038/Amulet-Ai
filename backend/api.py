from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from model_loader import ModelLoader

app = FastAPI(title="Amulet-AI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

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
    # TODO: สร้าง ModelLoader instance และฟังก์ชันที่จำเป็น
    # ตอนนี้ return ข้อมูลตัวอย่างเพื่อไม่ให้เกิด error
    
    # Mock data for now - จะต้องแทนที่ด้วยการทำนายจริง
    mock_topk = [
        {"class_id": 0, "class_name": "หลวงพ่อกวยแหวกม่าน", "confidence": 0.95},
        {"class_id": 1, "class_name": "โพธิ์ฐานบัว", "confidence": 0.03},
        {"class_id": 2, "class_name": "ฐานสิงห์", "confidence": 0.02}
    ]
    
    mock_valuation = {"p05": 1000.0, "p50": 5000.0, "p95": 15000.0}
    
    mock_recommendations = [
        {"market": "Facebook Marketplace", "reason": "ราคาดี เหมาะสำหรับพระเครื่องทั่วไป"}
    ]

    return {
        "top1": mock_topk[0],
        "topk": mock_topk,
        "valuation": mock_valuation,
        "recommendations": mock_recommendations,
    }

# รัน: uvicorn backend.api:app --reload --port 8000