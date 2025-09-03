# Amulet-AI API System

ระบบ API สำหรับ Amulet-AI ที่รองรับทั้งการทำงานด้วย Real AI Model และ Mock Data

## โครงสร้างระบบ API

ระบบ API ถูกพัฒนาให้ทำงานได้ในสองโหมดหลัก:

1. **โหมด Real AI Model** - ใช้ AI Model จริงที่ผ่านการเทรนแล้วในการทำนายประเภทพระเครื่อง
2. **โหมด Simulation** - จำลองการทำงานด้วย Mock Data เมื่อไม่มี AI Model

นอกจากนี้ยังรองรับการทำงานร่วมกับ services ต่างๆ:

- **Valuation Service** - บริการประเมินราคาพระเครื่อง
- **Recommendation Service** - บริการแนะนำตลาดที่เหมาะสม

## การใช้งาน API

### วิธีเริ่มต้นใช้งาน

```bash
# เริ่มระบบด้วย launcher อย่างง่าย
python backend/api/launcher.py

# หรือสามารถกำหนด host และ port ได้
python backend/api/launcher.py --host 0.0.0.0 --port 8080
```

### Endpoints ที่สำคัญ

- `GET /` - ข้อมูลพื้นฐานของ API
- `GET /health` - ตรวจสอบสถานะการทำงานของระบบ
- `GET /model-info` - ข้อมูลเกี่ยวกับ AI Model
- `POST /predict` - ทำนายประเภทพระเครื่องจากรูปภาพ
- `POST /predict-batch` - ทำนายพระเครื่องจากหลายรูปพร้อมกัน
- `GET /classes` - ดึงรายชื่อ classes ทั้งหมด
- `GET /system-status` - ดูสถานะระบบโดยละเอียด

### ตัวอย่างการใช้งาน

#### การทำนายด้วยรูปภาพ

```python
import requests

url = "http://127.0.0.1:8000/predict"
files = {
    "front": ("amulet.jpg", open("path/to/amulet.jpg", "rb"), "image/jpeg")
}

response = requests.post(url, files=files)
result = response.json()

print(f"ประเภท: {result['top1']['class_name']}")
print(f"ความเชื่อมั่น: {result['top1']['confidence']:.2%}")
print(f"ราคาประเมิน: ฿{result['valuation']['p50']:,}")
```

## โครงสร้างโค้ด

- `integrated_api.py` - ไฟล์หลักของระบบ API ที่รวมฟังก์ชันการทำงานทั้งหมด
- `launcher.py` - เครื่องมือสำหรับเริ่มต้นระบบ API อย่างง่าย

## ความต้องการระบบ

- Python 3.7+
- FastAPI
- Uvicorn
- Python-multipart
- PyTorch (สำหรับโหมด Real AI Model)
- PIL (สำหรับประมวลผลรูปภาพ)

## การแก้ไขปัญหาเบื้องต้น

1. **ไม่สามารถโหลด AI Model ได้**
   - ตรวจสอบว่ามีไฟล์ model ในโฟลเดอร์ `ai_models`
   - ระบบจะใช้โหมด Simulation โดยอัตโนมัติหากไม่พบ model

2. **Import Error**
   - ตรวจสอบว่าได้ติดตั้ง packages ที่จำเป็นแล้ว: `pip install fastapi uvicorn python-multipart`
   - ตรวจสอบโครงสร้างโฟลเดอร์ว่าถูกต้อง

3. **API เริ่มทำงานไม่ได้**
   - ตรวจสอบว่าไม่มี service อื่นใช้ port เดียวกัน
   - ลองใช้ `--port` เพื่อเปลี่ยน port
