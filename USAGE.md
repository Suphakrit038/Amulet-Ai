# Amulet-AI ระบบปัญญาประดิษฐ์วิเคราะห์พระเครื่อง

## 🚀 การติดตั้งและใช้งาน

### ขั้นตอนการติดตั้ง

1. **ติดตั้ง Python 3.8+**
   ```bash
   # ตรวจสอบเวอร์ชัน Python
   python --version
   ```

2. **Clone โปรเจค**
   ```bash
   git clone https://github.com/your-repo/Amulet-AI.git
   cd Amulet-AI
   ```

3. **รันการติดตั้งอัตโนมัติ**
   ```bash
   python setup.py
   ```

4. **เริ่มใช้งานระบบ**
   - Windows: ดับเบิลคลิก `start_system.bat`
   - หรือรันด้วยมือ:
     ```bash
     # Terminal 1: Backend
     python -m uvicorn backend.api:app --reload --port 8000
     
     # Terminal 2: Frontend  
     python -m streamlit run frontend/app_streamlit.py
     ```

### การเข้าใช้งาน

- **Frontend (UI)**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🏗️ สถาปัตยกรรมระบบ

### องค์ประกอบหลัก

1. **🤖 TensorFlow**: ฝึกและใช้งานโมเดล AI
2. **🔍 FAISS**: ค้นหาภาพที่คล้ายกัน
3. **📈 Scikit-learn**: ประเมินราคาด้วย ML
4. **🕷️ Scrapy**: เก็บข้อมูลราคาจากตลาด

### โครงสร้างไฟล์

```
Amulet-AI/
├── 📁 backend/           # API Server
│   ├── api.py           # FastAPI main
│   ├── model_loader.py  # TensorFlow model
│   ├── similarity_search.py  # FAISS
│   ├── price_estimator.py    # Scikit-learn
│   ├── market_scraper.py     # Scrapy
│   ├── valuation.py     # ระบบประเมินราคา
│   └── recommend.py     # ระบบแนะนำ
├── 📁 frontend/         # Streamlit UI
├── 📁 dataset/          # ข้อมูลรูปภาพ
├── 📁 models/           # โมเดล AI ที่ฝึกแล้ว
├── train_model.py       # สคริปต์ฝึกโมเดล
└── setup.py            # การติดตั้งอัตโนมัติ
```

## 🤖 การฝึกโมเดล AI

### เตรียมข้อมูล

1. **จัดเตรียมรูปภาพ**
   ```
   dataset/
   ├── หลวงพ่อกวยแหวกม่าน/
   │   ├── image1.jpg
   │   └── image2.jpg
   ├── โพธิ์ฐานบัว/
   ├── ฐานสิงห์/
   └── สีวลี/
   ```

2. **ฝึกโมเดล**
   ```bash
   python train_model.py
   ```

### การเลือกโมเดล

รองรับโมเดลพรีเทรน:
- **EfficientNetV2B0** (แนะนำ): สมดุลระหว่างความแม่นยำและความเร็ว
- **ResNet50V2**: ความแม่นยำสูง
- **MobileNetV3Large**: เหมาะสำหรับ mobile deployment

### การปรับแต่ง

```python
# ใน train_model.py
MODEL_NAME = "EfficientNetV2B0"
EPOCHS = 30
FINE_TUNE_EPOCHS = 15
BATCH_SIZE = 16
```

## 🔍 ระบบค้นหาภาพคล้ายกัน (FAISS)

### การใช้งาน

```python
from backend.similarity_search import find_similar_amulets

# ค้นหาภาพคล้ายกัน
similar = find_similar_amulets("query_image.jpg", top_k=5)
```

### การสร้าง Index

```python
from backend.similarity_search import SimilaritySearchEngine

engine = SimilaritySearchEngine()
engine.build_index("dataset/")  # สร้าง index จาก dataset
```

## 📈 ระบบประเมินราคา (Scikit-learn)

### การฝึกโมเดลประเมินราคา

```python
from backend.price_estimator import PriceEstimator, create_mock_training_data

# สร้างข้อมูลจำลอง
data = create_mock_training_data()

# ฝึกโมเดล
estimator = PriceEstimator()
estimator.train_model(data)
```

### การใช้งาน

```python
# ประเมินราคา
features = {
    'class_name': 'หลวงพ่อกวยแหวกม่าน',
    'condition': 'ใช้แล้ว',
    'age_years': 20
}

price = estimator.predict_price(features)
print(f"ราคาประเมิน: {price['p50']:,.0f} บาท")
```

## 🕷️ ระบบเก็บข้อมูลตลาด (Scrapy)

### การเก็บข้อมูล

```python
from backend.market_scraper import MarketDataCollector

collector = MarketDataCollector()
data_file = collector.collect_market_data()
```

### การวิเคราะห์ตลาด

```python
# วิเคราะห์เทรนด์ตลาด
insights = collector.analyze_market_trends('หลวงพ่อกวยแหวกม่าน')
print(f"ราคาเฉลี่ย: {insights['average_price']:,.0f} บาท")
```

## 🔧 API Endpoints

### หลัก

- `POST /predict`: วิเคราะห์พระเครื่องจากรูปภาพ
- `GET /health`: ตรวจสอบสถานะระบบ
- `GET /system-status`: ข้อมูลฟีเจอร์ที่ใช้งานได้

### เพิ่มเติม

- `POST /similarity-search`: ค้นหาภาพคล้ายกัน
- `GET /market-insights/{class_name}`: ข้อมูลตลาด
- `GET /supported-formats`: ฟอร์แมตรูปภาพที่รองรับ

### ตัวอย่างการเรียกใช้ API

```python
import requests

# อัปโหลดรูปภาพ
files = {'front': open('amulet.jpg', 'rb')}
response = requests.post('http://localhost:8000/predict', files=files)
result = response.json()

print(f"ประเภท: {result['top1']['class_name']}")
print(f"ความน่าจะเป็น: {result['top1']['confidence']:.2%}")
print(f"ราคาประเมิน: {result['valuation']['p50']:,.0f} บาท")
```

## 🎯 การพัฒนาต่อ

### เพิ่มข้อมูลใหม่

1. เพิ่มรูปภาพใน `dataset/`
2. รัน `python train_model.py` ใหม่
3. Restart ระบบ

### ปรับปรุงโมเดล

```python
# ใน train_model.py
# เปลี่ยนพารามิเตอร์
EPOCHS = 50  # เพิ่มจำนวน epoch
BATCH_SIZE = 32  # เพิ่ม batch size
```

### เพิ่มคลาสใหม่

1. เพิ่มโฟลเดอร์ใหม่ใน `dataset/`
2. อัปเดต `labels.json`
3. ฝึกโมเดลใหม่

## 🔍 การ Debug

### ตรวจสอบ Logs

```bash
# ดู logs ของ backend
tail -f backend.log

# ดู logs ของ Streamlit
tail -f frontend.log
```

### ปัญหาที่พบบ่อย

1. **Memory Error**: ลด `BATCH_SIZE` ใน train_model.py
2. **CUDA Error**: ติดตั้ง CUDA สำหรับ TensorFlow GPU
3. **Import Error**: รัน `python setup.py` ใหม่

### Test API

```bash
# ทดสอบ health check
curl http://localhost:8000/health

# ทดสอบ system status
curl http://localhost:8000/system-status
```

## 📊 Performance Monitoring

### ตรวจสอบประสิทธิภาพ

```python
# ใน train_model.py หลังจากฝึกเสร็จ
classifier.evaluate_model(validation_generator)
classifier.plot_training_history()
```

### Metrics ที่ต้องติดตาม

- **Accuracy**: ความแม่นยำโดยรวม
- **Top-2 Accuracy**: ความแม่นยำใน 2 อันดับแรก
- **Inference Time**: เวลาในการทำนาย
- **Memory Usage**: การใช้หน่วยความจำ

## 🚀 Deployment

### สำหรับ Production

1. **ใช้ Gunicorn**
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.api:app
   ```

2. **ใช้ Docker**
   ```dockerfile
   FROM python:3.9
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0"]
   ```

3. **TensorFlow Lite สำหรับ Mobile**
   ```python
   # โมเดล TFLite จะถูกสร้างอัตโนมัติใน train_model.py
   # ไฟล์: models/amulet_classifier/model.tflite
   ```

## 📞 การสนับสนุน

- **Issues**: สร้าง GitHub Issue
- **Documentation**: อ่านเพิ่มเติมใน `/docs`
- **Examples**: ดูตัวอย่างใน `/examples`

---

💡 **หมายเหตุ**: ระบบยังอยู่ในช่วงการพัฒนา บางฟีเจอร์อาจใช้ข้อมูลจำลอง (Mock Data) ในการทดสอบ
