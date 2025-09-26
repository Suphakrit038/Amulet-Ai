# 🔮 Amulet-AI - Thai Buddhist Amulet Recognition

ระบบ AI สำหรับจำแนกพระเครื่องไทย

## 🎯 ความสามารถ

- จำแนกพระเครื่อง 3 ประเภท: พระสมเด็จ, พระรอด, พระนางพญา
- ความแม่นยำ: 100% (บนชุดข้อมูลทดสอบ)
- เวลาตอบสนอง: < 0.2 วินาที
- รองรับการใช้งานผ่านเว็บและ API

## 🚀 การใช้งาน

### เริ่มระบบ
```bash
start.bat
```

### เข้าใช้งาน
- **เว็บไซต์**: http://localhost:8501
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📋 ความต้องการระบบ

- Python 3.8+
- RAM อย่างน้อย 1GB
- พื้นที่ดิสก์ 500MB

## 📦 การติดตั้ง

1. Clone repository
```bash
git clone https://github.com/your-repo/Amulet-Ai.git
cd Amulet-Ai
```

2. ติดตั้ง dependencies
```bash
pip install -r requirements_production.txt
```

3. เริ่มระบบ
```bash
start.bat
```

## 📁 โครงสร้างโปรเจค

```
Amulet-Ai/
├── ai_models/                    # โมเดล AI หลัก
│   └── enhanced_production_system.py
├── backend/                      # Backend services
│   └── api/
│       └── main_api.py          # API หลัก
├── frontend/                     # Frontend web app
│   └── production_app.py
├── dataset/                      # ชุดข้อมูลฝึก
├── trained_model/               # โมเดลที่ฝึกเสร็จ
├── requirements_production.txt  # Dependencies
├── start.bat                    # Startup script
└── README.md
```

## 🔧 API Usage

### อัพโหลดรูป
```python
import requests

files = {'file': open('amulet.jpg', 'rb')}
response = requests.post('http://localhost:8000/predict', files=files)
result = response.json()
print(f"ประเภท: {result['class_thai']}")
print(f"ความเชื่อมั่น: {result['confidence']:.2%}")
```

## 📊 ข้อมูลโมเดล

- **Algorithm**: Random Forest with Calibration
- **Features**: 81 dimensions (color, texture, shape)
- **Training Data**: 60 images (20 per class)
- **Validation**: Cross-validation score > 95%

## 🤝 การสนับสนุน

หากพบปัญหาการใช้งาน กรุณา:
1. ตรวจสอบ requirements
2. ตรวจสอบ log files
3. ปรึกษาเอกสาร API

## 📄 License

โปรเจคนี้อยู่ภายใต้ MIT License
