from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from model_loader import predict_topk
from valuation import get_quantiles
from recommend import recommend_markets

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
async def predict(front: UploadFile = File(...), back: UploadFile | None = File(None)):
    # 1) ทำนายรุ่น/พิมพ์
    topk = predict_topk(front, back, k=3)  # คืน list dict {class_id,class_name,confidence}

    # 2) ประเมินมูลค่าจาก top1
    class_id = topk[0]["class_id"]
    q = get_quantiles(class_id)           # {p05,p50,p95}

    # 3) แนะนำช่องทางขาย
    recs = recommend_markets(class_id, q)

    return {
        "top1": topk[0],
        "topk": topk,
        "valuation": q,
        "recommendations": recs,
    }

# รัน: uvicorn backend.api:app --reload --port 8000