from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import time
import base64
import json
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

    # โหลด mapping ชื่อไทย
    thai_labels_path = Path(__file__).parent.parent.parent / "ai_models" / "labels_thai.json"
    thai_labels = {}
    if thai_labels_path.exists():
        try:
            with open(thai_labels_path, 'r', encoding='utf-8') as f:
                thai_labels = json.load(f)
            print(f"SUCCESS: Loaded Thai labels from {thai_labels_path}")
        except Exception as e:
            print(f"WARNING: Cannot load Thai labels: {e}")

    # กำหนด path สำหรับรูปอ้างอิง
    REFERENCE_IMAGES_DIR = Path(__file__).parent.parent.parent / "unified_dataset" / "reference_images"
    if not REFERENCE_IMAGES_DIR.exists():
        print(f"WARNING: Reference images directory not found: {REFERENCE_IMAGES_DIR}")
        # Fallback paths
        fallback_paths = [
            Path(__file__).parent.parent.parent / "data_base",
            Path(__file__).parent.parent.parent / "dataset_organized",
            Path(__file__).parent.parent / "reference_images",
            Path(__file__).parent.parent.parent / "ai_models" / "reference_images"
        ]

        for path in fallback_paths:
            if path.exists():
                REFERENCE_IMAGES_DIR = path
                print(f"SUCCESS: Using fallback reference images path: {REFERENCE_IMAGES_DIR}")
                break
except ImportError as e:
    print(f"WARNING: Import error: {e}")
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

def get_reference_images(class_name: str, view_type: str = None, limit: int = 3):
    """ดึงรูปภาพอ้างอิงสำหรับคลาสที่ระบุ"""
    reference_images = {}

    # ปรับชื่อคลาสให้เป็นมาตรฐาน (ลบอักขระพิเศษและแปลงเป็น lowercase)
    normalized_class_name = class_name.replace('-', '_').replace(' ', '_').lower()

    # ลองหลายไดเร็กทอรีและชื่อคลาส
    class_ref_dirs = [
        REFERENCE_IMAGES_DIR / normalized_class_name,
        REFERENCE_IMAGES_DIR / class_name,
        REFERENCE_IMAGES_DIR / class_name.replace('_', '-'),
        REFERENCE_IMAGES_DIR / class_name.replace('-', '_')
    ]

    # Mapping ระหว่างชื่อภาษาไทยและภาษาอังกฤษ
    thai_to_english = {
        "พระสมเด็จหลวงพ่อกวย": "somdej_fatherguay",
        "พระพุทธเจ้าในวิหาร": "buddha_in_vihara",
        "พระสมเด็จฐานสิงห์": "somdej_lion_base",
        "พระสมเด็จประทานพร": "somdej_buddha_blessing",
        "พระสมเด็จหลังรูปเหมือน": "somdej_portrait_back",
        "พระสรรค์": "phra_san",
        "พระสิวลี": "phra_sivali",
        "สมเด็จพิมพ์ปรกโพธิ์": "somdej_prok_bodhi",
        "สมเด็จแหวกม่าน": "somdej_waek_man",
        "ออกวัดหนองอีดุก": "wat_nong_e_duk"
    }

    english_to_thai = {v: k for k, v in thai_to_english.items()}

    # เพิ่มชื่อภาษาไทยถ้ามี
    if normalized_class_name in thai_to_english:
        class_ref_dirs.append(REFERENCE_IMAGES_DIR / thai_to_english[normalized_class_name])
    elif normalized_class_name in english_to_thai:
        class_ref_dirs.append(REFERENCE_IMAGES_DIR / english_to_thai[normalized_class_name])

    # ดูในทุกไดเร็กทอรีที่เป็นไปได้
    class_ref_dir = None
    for dir_path in class_ref_dirs:
        if dir_path.exists():
            class_ref_dir = dir_path
            print(f"SUCCESS: Found reference directory: {class_ref_dir}")
            break
            
    if not class_ref_dir:
        print(f"WARNING: Reference directory not found for class: {class_name}")
        print(f"         Normalized name: {normalized_class_name}")
        print(f"         Searched in: {[str(d) for d in class_ref_dirs]}")

        # ลองค้นหาในทุกไดเร็กทอรีด้วย fuzzy matching
        subdirs = [d for d in REFERENCE_IMAGES_DIR.iterdir() if d.is_dir()]
        # จัดการ encoding สำหรับชื่อไดเร็กทอรี
        try:
            available_dirs = [d.name for d in subdirs]
            print(f"         Available directories: {available_dirs}")
        except UnicodeDecodeError:
            print(f"         Available directories: {len(subdirs)} directories found")

        for subdir in subdirs:
            try:
                subdir_name = subdir.name.lower()
                search_name = normalized_class_name.lower()

                # ตรวจสอบ fuzzy matching
                if (any(name in subdir_name for name in search_name.split('_')) or
                    any(name in search_name for name in subdir_name.split('_')) or
                    search_name in subdir_name or
                    subdir_name in search_name):
                    class_ref_dir = subdir
                    print(f"SUCCESS: Found similar reference directory: {subdir}")
                    break
            except UnicodeDecodeError:
                # ถ้ามีปัญหา encoding ให้ข้ามไดเร็กทอรีนี้ไป
                continue

    if not class_ref_dir:
        print(f"ERROR: No reference directory found for {class_name}")
        return reference_images
    
    try:
        # ดึงรูปภาพทั้งหมดหรือตาม view_type
        if view_type:
            img_files = list(class_ref_dir.glob(f"{view_type}_*.*"))[:limit]
            
            # ถ้าไม่พบไฟล์ที่ขึ้นต้นด้วย view_type ให้ลองหาไฟล์ที่มี view_type ในชื่อ
            if not img_files:
                img_files = [f for f in class_ref_dir.glob("*.*") 
                           if view_type in f.name.lower() and 
                           f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']][:limit]
        else:
            # ดึงรูป front ก่อน หากมี
            front_files = list(class_ref_dir.glob("front_*.*"))[:limit]
            
            # ถ้าไม่พบไฟล์ที่ขึ้นต้นด้วย front_ ให้ลองหาไฟล์ที่มี front ในชื่อ
            if not front_files:
                front_files = [f for f in class_ref_dir.glob("*.*") 
                             if "front" in f.name.lower() and 
                             f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']][:limit]
            
            back_files = []
            
            # ถ้ารูป front ไม่พอตามจำนวน limit ให้เอารูป back มาเพิ่ม
            if len(front_files) < limit:
                back_files = list(class_ref_dir.glob("back_*.*"))[:limit - len(front_files)]
                
                # ถ้าไม่พบไฟล์ที่ขึ้นต้นด้วย back_ ให้ลองหาไฟล์ที่มี back ในชื่อ
                if not back_files:
                    back_files = [f for f in class_ref_dir.glob("*.*") 
                                if "back" in f.name.lower() and 
                                f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']][:limit - len(front_files)]
                
            img_files = front_files + back_files
        
        # ถ้าไม่มีรูปอ้างอิงเลย ให้ลองหาไฟล์ภาพทั้งหมด
        if not img_files:
            try:
                all_files = list(class_ref_dir.glob("*.*"))[:limit]
                img_files = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']]
            except UnicodeDecodeError:
                # ถ้ามีปัญหา encoding ให้หาไฟล์แบบ alternative
                import os
                all_files = []
                for item in os.listdir(class_ref_dir):
                    if os.path.isfile(os.path.join(class_ref_dir, item)):
                        if item.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                            all_files.append(class_ref_dir / item)
                img_files = all_files[:limit]
        
        # แปลงรูปเป็น base64
        for i, img_file in enumerate(img_files):
            try:
                with open(img_file, "rb") as f:
                    img_bytes = f.read()
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

                    # ตรวจสอบ view_type จากชื่อไฟล์ (จัดการ encoding)
                    file_view_type = "unknown"
                    try:
                        filename_lower = img_file.name.lower()
                        if "front" in filename_lower:
                            file_view_type = "front"
                        elif "back" in filename_lower:
                            file_view_type = "back"
                    except UnicodeDecodeError:
                        # ถ้ามีปัญหา encoding ให้ใช้ค่า default
                        file_view_type = "unknown"

                    # จัดการ filename ที่มีปัญหา encoding
                    try:
                        safe_filename = img_file.name
                    except UnicodeDecodeError:
                        # ถ้ามีปัญหา encoding ให้สร้างชื่อไฟล์ใหม่
                        safe_filename = f"reference_image_{i+1}.jpg"

                    reference_images[f"ref_{i+1}"] = {
                        "image_b64": img_b64,
                        "filename": safe_filename,
                        "view_type": file_view_type,
                        "path": str(img_file.relative_to(REFERENCE_IMAGES_DIR.parent))
                    }
                    print(f"INFO: Added reference image: {safe_filename} ({file_view_type})")
            except Exception as e:
                print(f"WARNING: Error processing reference image {img_file}: {e}")
                # ถ้ามีปัญหา ให้ข้ามไฟล์นี้ไป
                continue
                
        return reference_images
    except Exception as e:
        print(f"WARNING: Error getting reference images for {class_name}: {e}")
        return reference_images

@app.on_event("startup")
async def startup_event():
    """โหลด model เมื่อเซิร์ฟเวอร์เริ่มทำงาน"""
    print("Starting Amulet-AI Backend with Real Model...")
    success = model_loader.initialize()
    if success:
        print("SUCCESS: Backend ready with Real AI Model!")
    else:
        print("WARNING: Backend started but Model may have issues")

@app.get("/")
async def root():
    return {
        "message": "Amulet-AI Backend with Real Trained Model",
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
        
        # อ่านรูปภาพหลัง (ถ้ามี)
        back_bytes = None
        if back:
            back_bytes = await back.read()
            print(f"Processing front: {front.filename} ({len(front_bytes)} bytes) and back: {back.filename} ({len(back_bytes)} bytes)")
        else:
            print(f"Processing front only: {front.filename} ({len(front_bytes)} bytes)")
        
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
        
        # ดึงรูปภาพอ้างอิง - ด้านหน้า
        front_reference_images = get_reference_images(top1["class_name"], "front", 2)
        
        # ดึงรูปภาพอ้างอิง - ด้านหลัง
        back_reference_images = get_reference_images(top1["class_name"], "back", 2)
        
        # รวมรูปอ้างอิงทั้งหมด
        reference_images = {**front_reference_images, **back_reference_images}
        
        # ถ้าไม่มีรูปอ้างอิงเลย ลองดึงโดยไม่ระบุ view_type
        if not reference_images:
            reference_images = get_reference_images(top1["class_name"], limit=4)
        
        processing_time = time.time() - start_time
        
        # แปลงรูปภาพที่อัพโหลดเป็น base64 เพื่อส่งกลับไป
        front_b64 = base64.b64encode(front_bytes).decode('utf-8')
        back_b64 = None
        if back_bytes:
            back_b64 = base64.b64encode(back_bytes).decode('utf-8')
        
        # เพิ่มชื่อไทยให้กับ top1 และ predictions
        top1_with_thai = top1.copy()
        top1_with_thai["class_name_thai"] = thai_labels.get(top1["class_name"], top1["class_name"])

        predictions_with_thai = []
        for pred in predictions:
            pred_with_thai = pred.copy()
            pred_with_thai["class_name_thai"] = thai_labels.get(pred["class_name"], pred["class_name"])
            predictions_with_thai.append(pred_with_thai)

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
                "front": {
                    "filename": front.filename,
                    "size": len(front_bytes),
                    "format": "image",
                    "image_b64": front_b64
                },
                "back": {
                    "filename": back.filename if back else None,
                    "size": len(back_bytes) if back_bytes else 0,
                    "format": "image" if back_bytes else None,
                    "image_b64": back_b64
                } if back_bytes else None
            },
            "top1": top1_with_thai,
            "topk": predictions_with_thai,
            "valuation": valuation,
            "recommendations": recommendations,
            "reference_images": reference_images,
            "metadata": prediction_result.get("metadata", {}),
            "thai_labels": thai_labels  # รวม mapping ชื่อไทยทั้งหมด
        }
        
        print(f"SUCCESS: Prediction successful: {top1['class_name']} ({top1['confidence']:.2%}) in {processing_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"ERROR: Prediction error: {str(e)}")
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
    print("Starting Amulet-AI Backend with Real Model...")
    uvicorn.run(app, host="127.0.0.1", port=8001)
