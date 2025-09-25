# 🔮 Amulet-AI - ระบบวิเคราะห์### วิธีที่ 4: รันตรงๆ (รวดเร็ว)
```bash
streamlit run frontend/main_app.py
```ครื่องด้วยปัญญาประดิษฐ์

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)
![Status](https://img.shields.io/badge/status-production--ready-success)

**ระบบปัญญาประดิษฐ์สำหรับการตรวจสอบและประเมินพระเครื่องไทย โดยใช้ Deep Learning**
- ✅ **พร้อมใช้งาน Production** - ระบบเสถียรและทดสอบแล้ว
- 🤖 **AI ทันสมัย** - Vision Transformer + EfficientNet
- 📱 **Responsive UI** - รองรับมือถือและเดสก์ทอป
- 🇹🇭 **ภาษาไทย** - รองรับภาษาไทยเต็มรูปแบบ

## 🚀 เริ่มใช้งานด่วน

### วิธีที่ 1: ใช้ Batch Script (แนะนำสำหรับ Windows)
```bash
start.bat
```

### วิธีที่ 2: ใช้ Python Launcher (ครบครัน)
```bash
python launch_complete.py
```

### วิธีที่ 3: ทดสอบระบบ
```bash
python launch_complete.py --test
```

### วิธีที่ 4: รันตरงๆ (รวดเร็ว)
```bash
streamlit run frontend/app_streamlit.py
```

## 📁 โครงสร้างโปรเจค (อัพเดท 2025)

```
Amulet-Ai/
├── 🤖 ai_models/           # AI Models และ ML Pipeline  
├── 🌐 backend/             # Backend APIs
├── 🎨 frontend/            # User interfaces
├── 📊 dataset_realistic/   # Training/Test datasets
├── 🔧 tools/              # Development tools
├── 📖 docs/               # Documentation และ Reports
├── 🎯 trained_model/      # Active ML model (ล่าสุด)
├── 🔍 robustness_analysis/ # Model testing results
└── 💾 feature_cache/      # Performance cache (1,257 files)
```

### 📋 รายงานและเอกสาร
- **System Status**: `docs/reports/REAL_SYSTEM_TRUTH_TABLE.md`
- **Performance Report**: `docs/reports/ACCURACY_PERFORMANCE_REPORT.md` 
- **Priority Matrix**: `docs/reports/SYSTEM_PRIORITY_MATRIX.md`
- **Action Plan**: `docs/reports/PROBLEM_TRACKING_MATRIX.md`

### พารามิเตอร์เพิ่มเติม

- `--test` - ทดสอบระบบโดยไม่เริ่มเซิร์ฟเวอร์
- `--api-only` - เปิดเฉพาะ Backend API
- `--real-model` - ใช้โมเดล AI จริง (แทนที่จะใช้ข้อมูลจำลอง)
- `--no-browser` - ไม่เปิดเว็บเบราว์เซอร์อัตโนมัติ

## โครงสร้างระบบ

ระบบได้รับการจัดโครงสร้างใหม่เพื่อความเป็นระเบียบและบำรุงรักษาง่าย:

> **หมายเหตุ**: ข้อมูลฝึกสอนได้รับการจัดระเบียบใหม่ตามรูปแบบสากลเพื่อความสะดวกในการประมวลผล ([ดูรายละเอียด](docs/DATASET_REORGANIZATION.md))

- **ai_models/** - โมเดล Deep Learning และระบบการฝึกสอน
  - **core/** - โมเดลหลักและไฟล์ labels
  - **training/** - สคริปต์และโมดูลสำหรับการฝึกสอน
  - **pipelines/** - กระบวนการประมวลผลข้อมูล
  - **evaluation/** - การทดสอบและประเมินโมเดล
  - **configs/** - การตั้งค่าสำหรับโมเดล
  
- **backend/** - API สำหรับการวิเคราะห์ด้วย FastAPI
  - **api/** - API endpoints และอินเตอร์เฟซ
  - **models/** - การโหลดและประมวลผลโมเดล
  - **services/** - บริการหลังบ้านต่างๆ
  - **config/** - การตั้งค่าสำหรับ backend
  
- **frontend/** - ส่วนติดต่อผู้ใช้ด้วย Streamlit
  - **pages/** - หน้าและมุมมองหลัก
  - **components/** - คอมโพเนนต์ UI ที่ใช้ซ้ำได้
  - **utils/** - ยูทิลิตี้สำหรับ frontend
  
- **utils/** - ฟังก์ชันและเครื่องมือช่วยเหลือ
  - **config/** - ยูทิลิตี้การตั้งค่า
  - **image/** - ยูทิลิตี้การประมวลผลภาพ
  - **logging/** - การบันทึกและการจัดการข้อผิดพลาด
  - **data/** - ยูทิลิตี้การประมวลผลข้อมูล
  
- **docs/** - เอกสารประกอบทั้งหมด
  - **api/** - เอกสาร API
  - **guides/** - คู่มือสำหรับผู้ใช้และผู้พัฒนา
  - **system/** - สถาปัตยกรรมและการออกแบบระบบ
  - **development/** - คู่มือการพัฒนาและการเผยแพร่
  
- **scripts/** - สคริปต์สำหรับเริ่มระบบและการทำงานหลัก
- **tests/** - ชุดทดสอบระบบทั้งหมด
- **tools/** - เครื่องมือสำหรับบำรุงรักษาระบบ
- **config/** - ไฟล์การตั้งค่าต่างๆ

ดูรายละเอียดเพิ่มเติมได้ที่:
- [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) - โครงสร้างไฟล์และโฟลเดอร์โดยละเอียด
- [DIRECTORY_STRUCTURE.md](docs/DIRECTORY_STRUCTURE.md) - การจัดหมวดหมู่โฟลเดอร์และการจัดระเบียบภายใน

## เครื่องมือบำรุงรักษาระบบ

ระบบมาพร้อมกับเครื่องมือบำรุงรักษาที่ครบครัน:

```
python tools/amulet_toolkit.py --menu
```

เครื่องมือนี้มีฟังก์ชันการทำงานหลายอย่าง:
- ตรวจสอบระบบ (`--verify`)
- ซ่อมแซมระบบ (`--repair`)
- บำรุงรักษาระบบ (`--maintain`)
- ทดสอบไฟล์ (`--test-file PATH`)

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
- `PROJECT_STRUCTURE.md` - โครงสร้างโปรเจคอย่างละเอียด
- `MODULAR_ARCHITECTURE.md` - คู่มือสถาปัตยกรรมแบบโมดูลาร์

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

- **main_api.py** - FastAPI แอปพลิเคชัน หลักพร้อม endpoints ทั้งหมด
- **production_api.py** - API สำหรับการใช้งาน production (รวมจากหลาย API ย่อย)
- **ai_model_api.py** - API เฉพาะสำหรับ AI Model
- **model_inference.py** - บริการจัดการโมเดล AI และ inference
- **valuation.py** - ระบบประเมินราคาพระเครื่อง
- **recommend.py** - ระบบแนะนำตลาดและร้านค้า
- **recommendation_engine.py** - เครื่องมือแนะนำแบบ optimized
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

- **utils.py** - ฟังก์ชันยูทิลิตี้รวม (รวมจากหลายไฟล์)
- **config_manager.py** - จัดการการตั้งค่า
- **image_utils.py** - เครื่องมือประมวลผลรูปภาพ
- **error_handler.py** - จัดการข้อผิดพลาด
- **logger.py** - ระบบบันทึกข้อมูล

### **Tools (tools/)**

- **amulet_toolkit.py** - เครื่องมือตรวจสอบ ซ่อมแซม และบำรุงรักษาระบบ
- **restructure_project.py** - เครื่องมือจัดโครงสร้างโปรเจค

### **Scripts (scripts/)**

- **amulet_launcher.py** - ไฟล์หลักสำหรับเริ่มระบบทั้งหมด
- **setup_models.py** - ดาวน์โหลดและตรวจสอบโมเดล AI
- **launch.bat** - ไฟล์แบตช์สำหรับเริ่มระบบใน Windows

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
