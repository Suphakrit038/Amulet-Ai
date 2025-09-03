# Amulet AI System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-red.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![Status](https://img.shields.io/badge/Status-Real%20AI%20Model-brightgreen.svg)](https://github.com)

**ระบบปัญญาประดิษฐ์สำหรับการตรวจสอบและประเมินพระเครื่องไทย โดยใช้ Deep Learning**

## การติดตั้ง

1. ติดตั้ง requirements
```
pip install -r requirements.txt
```

2. ดาวน์โหลดโมเดล AI
```
python setup_models.py
```

## การใช้งาน

เริ่มระบบทั้งหมดด้วยคำสั่ง:

```
python amulet_launcher.py
```

หรือใช้ไฟล์ batch:

```
amulet_launcher.bat
```

ระบบจะเริ่มทำงานทั้ง Backend API และ Frontend พร้อมกัน

### พารามิเตอร์เพิ่มเติม

- `--test` - ทดสอบระบบโดยไม่เริ่มเซิร์ฟเวอร์
- `--api-only` - เปิดเฉพาะ Backend API
- `--real-model` - ใช้โมเดล AI จริง (แทนที่จะใช้ข้อมูลจำลอง)
- `--no-browser` - ไม่เปิดเว็บเบราว์เซอร์อัตโนมัติ

## โครงสร้างระบบ

- **frontend/** - ส่วนติดต่อผู้ใช้ด้วย Streamlit
- **backend/** - API สำหรับการวิเคราะห์ด้วย FastAPI
- **ai_models/** - โมเดล Deep Learning และระบบการฝึกสอน
- **dataset/** - ข้อมูลสำหรับการฝึกสอนโมเดล
- **utils/** - ฟังก์ชันและเครื่องมือช่วยเหลือ

## ข้อกำหนดของระบบ

- Python 3.8+
- CPU หรือ GPU (แนะนำสำหรับโมเดลจริง)
- RAM อย่างน้อย 4GB
- ความจุดิสก์: 200MB สำหรับโค้ด + 1GB สำหรับโมเดล

## การพัฒนา

กรุณาดูเอกสารในโฟลเดอร์ `docs/` สำหรับรายละเอียดเพิ่มเติม:
- `API.md` - เอกสาร API
- `SYSTEM_GUIDE.md` - คู่มือระบบโดยละเอียด
- `DEPLOYMENT.md` - คำแนะนำการ deploy

---

## สถาปัตยกรรมระบบ

### **แผนภาพสถาปัตยกรรม**
```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                    │
├─────────────────┬───────────────────┬───────────────────────┤
│  Web UI         │  API Docs         │  Admin Dashboard      │
│  (Streamlit)    │  (Swagger/ReDoc)  │  (System Monitor)     │
└─────────────────┴───────────────────┴───────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    API GATEWAY LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Server │ Request Router │ Authentication │ CORS    │
│  Rate Limiting  │ Error Handler  │ Validation     │ Logging │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   BUSINESS LOGIC LAYER                     │
├──────────────┬──────────────┬──────────────┬───────────────┤
│   AI Engine │   Valuation  │   Recommend  │   Analytics   │
│ - Image Proc │ - ML Models  │ - Market API │ - Statistics  │
│ - CNN Model  │ - Price Calc │ - Location   │ - Performance │
│ - Features   │ - Confidence │ - Rating     │ - Monitoring  │
└──────────────┴──────────────┴──────────────┴───────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACCESS LAYER                      │
├─────────────┬───────────────┬───────────────┬───────────────┤
│   Database  │   File Store  │   Cache      │   External    │
│ - SQLite    │ - Model Files │ - Redis/Mem  │ - Market APIs │
│ - Metadata  │ - Images      │ - Results    │ - Web Scraper │
│ - Logs      │ - Config      │ - Sessions   │ - Price Data  │
└─────────────┴───────────────┴───────────────┴───────────────┘
```

## ไฟล์หลักและหน้าที่

### **Backend (backend/)**

- **api.py** - FastAPI แอปพลิเคชัน หลักพร้อม endpoints ทั้งหมด
- **model_loader.py** - ระบบโหลดและจัดการโมเดล AI 
- **valuation.py** - ระบบประเมินราคาพระเครื่อง
- **recommend.py** - ระบบแนะนำตลาดและร้านค้า
- **similarity_search.py** - ค้นหาพระเครื่องที่คล้ายกัน
- **price_estimator.py** - คำนวณราคาด้วย ML
- **market_scraper.py** - เก็บข้อมูลตลาด

### **Frontend (frontend/)**

- **app_streamlit.py** - หน้าเว็บแอปพลิเคชันหลัก
- **utils.py** - ฟังก์ชันช่วยเหลือสำหรับ frontend

### **AI Models (ai_models/)**

- **amulet_model.h5** - โมเดล TensorFlow หลัก
- **amulet_model.tflite** - โมเดลสำหรับอุปกรณ์มือถือ
- **labels.json** - คลาสสำหรับการจำแนก
- **dataset_organizer.py** - จัดการข้อมูลการฝึกสอน
- **master_training_system.py** - ระบบฝึกสอนหลัก

### **Utils (utils/)**

- **config_manager.py** - จัดการการตั้งค่า
- **image_utils.py** - เครื่องมือประมวลผลรูปภาพ
- **error_handler.py** - จัดการข้อผิดพลาด
- **logger.py** - ระบบบันทึกข้อมูล

### **Root Directory**

- **amulet_launcher.py** - ไฟล์หลักสำหรับเริ่มระบบทั้งหมด
- **setup_models.py** - ดาวน์โหลดและตรวจสอบโมเดล AI
- **config.json** - การตั้งค่าระบบหลัก
- **requirements.txt** - รายการแพ็คเกจที่ต้องติดตั้ง

## การเข้าถึงระบบ

- **Streamlit UI**: http://localhost:8501
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **System Health**: http://localhost:8000/health

## คลาสพระเครื่องที่รองรับ

| คลาส | ชื่อพระ | คำอธิบาย | ช่วงราคา |
|-------|-----------|-------------|-------------|
| 1 | หลวงพ่อกวยแหวกม่าน | LP Kuay curtain-parting amulet | ฿15,000 - ฿120,000 |
| 2 | โพธิ์ฐานบัว | Buddha with lotus base | ฿8,000 - ฿75,000 |
| 3 | ฐานสิงห์ | Lion-base Buddha | ฿12,000 - ฿85,000 |
| 4 | สีวลี | Sivali amulet | ฿5,000 - ฿50,000 |

---

## License

MIT License

Copyright (c) 2025 Amulet-AI Project

---

**🌟 ขอบคุณที่ใช้ระบบ Amulet-AI! พวกเรากำลังอนุรักษ์มรดกทางพุทธศาสนาไทยด้วยเทคโนโลยี 🙏**
