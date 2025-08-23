"""
Super Minimal API for testing
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI(title="Amulet-AI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "🤖 Amulet-AI API is working!", "status": "ready"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ai_mode": "testing"}

@app.post("/predict")
async def predict(front: UploadFile = File(...)):
    # Simple mock prediction
    classes = ["หลวงพ่อกวยแหวกม่าน", "โพธิ์ฐานบัว", "ฐานสิงห์", "สีวลี"]
    main_class = random.choice(classes)
    confidence = random.uniform(0.7, 0.95)
    
    return {
        "top1": {"class_id": 0, "class_name": main_class, "confidence": confidence},
        "topk": [
            {"class_id": 0, "class_name": main_class, "confidence": confidence},
            {"class_id": 1, "class_name": random.choice(classes), "confidence": random.uniform(0.05, 0.2)},
            {"class_id": 2, "class_name": random.choice(classes), "confidence": random.uniform(0.02, 0.1)}
        ],
        "valuation": {"p05": 15000, "p50": 45000, "p95": 120000, "confidence": "high"},
        "recommendations": [{"name": "ตลาดพระ", "distance": 5.2}],
        "ai_mode": "testing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
