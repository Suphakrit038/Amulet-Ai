# Amulet-AI - ระบบปัญญาประดิษฐ์ระบุพระเครื่อง

ระบบปัญญาประดิษฐ์สำหรับการวิเคราะห์และระบุพระเครื่องไทยโดยใช้เทคโนโลยี Deep Learning

## คุณสมบัติหลัก

- **วิเคราะห์รูปภาพพระเครื่อง**: ใช้ AI วิเคราะห์รูปภาพและระบุประเภทพระเครื่อง
- **ตรวจสอบความถูกต้อง**: ประเมินความน่าเชื่อถือและคุณภาพของพระเครื่อง  
- **ประมาณราคา**: ให้ข้อมูลราคาตลาดโดยประมาณ
- **ส่วนติดต่อผู้ใช้ที่ใช้งานง่าย**: เว็บแอปพลิเคชันที่สวยงามและใช้งานสะดวก

## โครงสร้างโปรเจค

```
Amulet-AI/
├── frontend/           # Streamlit web application
├── backend/            # FastAPI backend services  
├── ai_models/          # AI model training and inference
├── dataset/            # Training dataset
├── docs/               # Documentation
├── tests/              # Test files
└── utils/              # Utility functions
```

## การติดตั้งและใช้งาน

### ข้อกำหนดระบบ

- Python 3.8 หรือสูงกว่า
- RAM อย่างน้อย 4GB (แนะนำ 8GB+)
- พื้นที่ว่างบนดิสก์ 2GB

### การติดตั้ง

1. **Clone repository**
```powershell
git clone https://github.com/yourusername/Amulet-AI.git
cd Amulet-AI
```

2. **สร้าง virtual environment**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **ติดตั้ง dependencies**
```powershell
pip install -r requirements.txt
```

### การเริ่มระบบ

1. **เริ่ม Backend API**
```powershell
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000
```

2. **เริ่ม Frontend (Terminal ใหม่)**
```powershell
cd frontend
streamlit run app_straemlit.py --server.port 8501
```

3. **เข้าใช้งานระบบ**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## การใช้งาน

1. เปิดเว็บไซต์ที่ http://localhost:8501
2. อัปโหลดรูปภาพพระเครื่องที่ต้องการวิเคราะห์
3. รอระบบประมวลผลและแสดงผลการวิเคราะห์
4. ดูข้อมูลการระบุประเภท ความน่าเชื่อถือ และราคาประมาณ

## การพัฒนา

### โครงสร้าง API

- `/predict` - วิเคราะห์รูปภาพพระเครื่อง
- `/health` - ตรวจสอบสถานะระบบ
- `/models` - ข้อมูลโมเดล AI ที่ใช้

### การเพิ่มโมเดลใหม่

1. วางไฟล์โมเดลใน `ai_models/saved_models/`
2. อัปเดต configuration ใน `backend/config.py`
3. เพิ่ม endpoint ใหม่ใน `backend/api.py`

## การทดสอบ

```powershell
cd tests
python -m pytest
```

## เทคโนโลยีที่ใช้

- **Frontend**: Streamlit, CSS3, HTML5
- **Backend**: FastAPI, Uvicorn
- **AI/ML**: PyTorch, TensorFlow, PIL
- **Image Processing**: OpenCV, Pillow
- **Data**: NumPy, Pandas

## ผู้พัฒนา

- **พัฒนาโดย**: Amulet-AI Team
- **ติดต่อ**: [อีเมลของคุณ]
- **เวอร์ชัน**: 1.0.0

## ใบอนุญาต

MIT License - ดูรายละเอียดใน LICENSE file

## การสนับสนุน

หากพบปัญหาหรือมีคำแนะนำ กรุณาสร้าง Issue ใน GitHub Repository

---
*ระบบนี้พัฒนาขึ้นเพื่อเป็นเครื่องมือช่วยในการศึกษาและวิเคราะห์พระเครื่องไทยเท่านั้น*
