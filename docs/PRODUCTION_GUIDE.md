# 🔮 Amulet-AI v2.0 Production Guide
## คู่มือการใช้งานระบบ Production

### 📋 ภาพรวมระบบ
Amulet-AI v2.0 เป็นระบบจำแนกพระเครื่องไทยด้วย AI ที่ปรับปรุงให้เหมาะสำหรับการใช้งานจริง:

- **ข้อมูลเทรน**: 20 รูปต่อประเภท (เหมาะสำหรับข้อมูลจำกัด)
- **AI Model**: Random Forest + Feature Engineering
- **API**: FastAPI (Production-ready)
- **Frontend**: Streamlit (User-friendly)
- **ประเภทพระเครื่อง**: 3 ประเภท (พระนางพญา, พระร็อด, พระสมเด็จ)

---

## 🚀 การเริ่งต้นระบบ

### วิธีที่ 1: เริ่มต้นแบบครบครัน (แนะนำสำหรับครั้งแรก)
```bash
# ติดตั้ง dependencies
pip install -r requirements_production.txt

# เริ่มต้นระบบครบครัน
python launch_production.py
```

### วิธีที่ 2: เริ่มต้นแบบเร็ว (หลังจากตั้งค่าแล้ว)
```bash
python launch_production.py --quick
```

### วิธีที่ 3: เริ่มต้นแยกส่วน
```bash
# 1. ปรับปรุง dataset
python tools/optimize_dataset.py

# 2. เทรนโมเดล
python ai_models/optimized_model.py

# 3. เริ่ม API server
python backend/api/production_ready_api.py

# 4. เริ่ม Frontend (terminal ใหม่)
streamlit run frontend/production_app.py
```

---

## 🌐 การเข้าถึงระบบ

หลังจากเริ่มต้นระบบแล้ว:

- **Frontend**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 📊 โครงสร้างข้อมูล

### Dataset Structure (หลังจากปรับปรุง)
```
dataset_optimized/
├── train/                 # ข้อมูลเทรน
│   ├── phra_nang_phya/   # 20 รูป
│   ├── phra_rod/         # 20 รูป
│   └── phra_somdej/      # 20 รูป
├── validation/           # ข้อมูล validation
│   ├── phra_nang_phya/   # 5 รูป
│   ├── phra_rod/         # 5 รูป
│   └── phra_somdej/      # 5 รูป
└── test/                 # ข้อมูลทดสอบ
    ├── phra_nang_phya/   # 10 รูป
    ├── phra_rod/         # 10 รูป
    └── phra_somdej/      # 10 รูป
```

### Model Structure
```
trained_model_optimized/
├── optimized_model.joblib    # โมเดล Random Forest
├── scaler.joblib            # Feature scaler
├── label_encoder.joblib     # Label encoder
└── model_config.json        # การตั้งค่าโมเดล
```

---

## 🤖 คุณสมบัติของ AI Model

### Feature Engineering
- **HOG Features**: 200 dimensions
- **ORB Features**: 32 dimensions  
- **Color Histogram**: 512 dimensions
- **LBP Features**: 16 dimensions
- **Statistical Features**: 7 dimensions
- **Total Features**: ~767 dimensions

### Model Configuration
- **Algorithm**: Random Forest
- **Trees**: 50 (ปรับให้เหมาะกับข้อมูลเล็ก)
- **Max Depth**: 10
- **Class Balancing**: Enabled
- **Cross Validation**: 3-fold

---

## 🌐 API Reference

### Endpoints

#### `GET /`
ข้อมูลพื้นฐานของ API
```json
{
  "message": "Amulet-AI Production API",
  "version": "2.0.0",
  "status": "ready"
}
```

#### `GET /health`
ตรวจสอบสถานะระบบ
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-09-26T..."
}
```

#### `POST /predict`
ทำนายประเภทพระเครื่อง
- **Input**: รูปภาพ (multipart/form-data)
- **Output**: 
```json
{
  "predicted_class": "phra_somdej",
  "thai_name": "พระสมเด็จ",
  "confidence": 0.85,
  "confidence_percentage": "85.0%",
  "timestamp": "2025-09-26T...",
  "model_version": "2.0.0"
}
```

#### `GET /classes`
ดูประเภทพระเครื่องที่รองรับ

#### `GET /model-info`
ดูข้อมูลโมเดล

---

## 🎨 การใช้งาน Frontend

### ขั้นตอนการใช้งาน
1. เปิดเว็บไซต์ที่ http://localhost:8501
2. ตรวจสอบสถานะ API ใน Sidebar
3. อัพโหลดรูปภาพพระเครื่อง
4. กดปุ่ม "เริ่มจำแนก"
5. ดูผลการทำนายและความมั่นใจ

### ไฟล์ที่รองรับ
- JPG, JPEG, PNG
- ขนาดไฟล์: ไม่จำกัด (แนะนำไม่เกิน 10MB)
- ความละเอียด: ปรับอัตโนมัติเป็น 128x128

---

## 🔧 การปรับแต่งระบบ

### ปรับจำนวนข้อมูลเทรน
แก้ไขไฟล์ `tools/optimize_dataset.py`:
```python
self.config = {
    'train_samples_per_class': 30,  # เปลี่ยนจาก 20
    'val_samples_per_class': 8,     # เปลี่ยนจาก 5
    'test_samples_per_class': 12,   # เปลี่ยนจาก 10
}
```

### ปรับพารามิเตอร์โมเดล
แก้ไขไฟล์ `ai_models/optimized_model.py`:
```python
self.model = RandomForestClassifier(
    n_estimators=100,     # เพิ่มจำนวน trees
    max_depth=15,         # เพิ่มความลึก
    min_samples_split=2,  # ลดค่าต่ำสุด
    random_state=42
)
```

### ปรับพอร์ต API
แก้ไขไฟล์ `backend/api/production_ready_api.py`:
```python
uvicorn.run(
    "production_ready_api:app",
    host="0.0.0.0",
    port=9000,  # เปลี่ยนพอร์ต
    reload=False
)
```

---

## 📈 การติดตามประสิทธิภาพ

### ตรวจสอบสถานะ API
```bash
curl http://localhost:8000/health
```

### ทดสอบการทำนาย
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

### ดู Log ของระบบ
- API logs จะแสดงใน terminal ที่รัน API
- Frontend logs จะแสดงใน terminal ที่รัน Streamlit

---

## 🐛 การแก้ไขปัญหา

### ปัญหาที่พบบ่อย

#### 1. API ไม่เริ่มต้น
```bash
# ตรวจสอบว่าพอร์ต 8000 ว่างหรือไม่
netstat -an | findstr :8000

# หรือเปลี่ยนพอร์ต
python backend/api/production_ready_api.py --port 9000
```

#### 2. โมเดลไม่โหลด
```bash
# ตรวจสอบว่าโมเดลถูกเทรนแล้ว
python ai_models/optimized_model.py

# ตรวจสอบไฟล์โมเดล
ls trained_model_optimized/
```

#### 3. Frontend ไม่เชื่อมต่อ API
- ตรวจสอบว่า API server ทำงาน
- ตรวจสอบ URL ใน `frontend/production_app.py`

#### 4. Dataset ไม่พบ
```bash
# สร้าง dataset ใหม่
python tools/optimize_dataset.py
```

---

## 🚀 การ Deploy Production

### สำหรับ Development
```bash
python launch_production.py
```

### สำหรับ Production Server
```bash
# ใช้ Gunicorn (Linux/Mac)
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.api.production_ready_api:app --bind 0.0.0.0:8000

# หรือใช้ Uvicorn (Windows/Linux/Mac)
uvicorn backend.api.production_ready_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment (อนาคต)
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements_production.txt
EXPOSE 8000
CMD ["python", "launch_production.py", "--quick"]
```

---

## 📞 การติดต่อและสนับสนุน

- **เวอร์ชัน**: v2.0.0
- **สถานะ**: Production Ready
- **อัพเดตล่าสุด**: 26 กันยายน 2025

### การอัพเกรดระบบ
1. สำรองข้อมูลโมเดลเก่า
2. อัพเดตโค้ดใหม่
3. รัน `python launch_production.py` ใหม่
4. ทดสอบระบบ

---

**🎉 ยินดีต้อนรับสู่ Amulet-AI v2.0 - ระบบจำแนกพระเครื่องไทยที่พร้อมใช้งานจริง!**