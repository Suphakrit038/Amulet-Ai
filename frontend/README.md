# 🔮 Amulet AI Frontend Dashboard

ระบบ Frontend สำหรับ Amulet AI ที่ใช้ Streamlit เป็นหลักในการสร้าง Web Application Dashboard

## 🌟 คุณสมบัติ

- **Dashboard แบบไม่มี Sidebar** - หน้าจอสะอาดและโฟกัสได้ดี
- **Real-time Analytics** - แสดงข้อมูลสถิติแบบเรียลไทม์
- **Image Upload & Prediction** - อัปโหลดรูปภาพและทำนายผล
- **Interactive Charts** - กราฟแบบโต้ตอบได้ด้วย Plotly
- **Responsive Design** - รองรับหน้าจอทุกขนาด
- **Custom CSS Styling** - การออกแบบที่สวยงามและทันสมัย

## 📁 โครงสร้างไฟล์

```
frontend/
├── main_streamlit_app.py    # แอปหลัก Streamlit
├── run_frontend.py          # ตัวเรียกใช้แอป
├── style.css               # ไฟล์ CSS สำหรับ styling
├── requirements.txt        # Dependencies
└── README.md              # เอกสารนี้
```

## 🚀 การติดตั้งและใช้งาน

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 2. เรียกใช้แอป

#### วิธีที่ 1: ใช้ Runner Script (แนะนำ)
```bash
python run_frontend.py
```

#### วิธีที่ 2: เรียกใช้ Streamlit โดยตรง
```bash
streamlit run main_streamlit_app.py --server.port 8501
```

### 3. เปิดเว็บเบราว์เซอร์

เปิด URL: `http://localhost:8501`

## 🎨 คุณสมบัติของ Dashboard

### 📊 เมตริกส์หลัก
- จำนวนการทำนายทั้งหมด
- ความแม่นยำเฉลี่ย
- จำนวนพระเครื่องแท้
- จำนวนของปลอมที่ตรวจพบ

### 📈 การแสดงผลข้อมูล
- **การทำนายรายวัน**: กราฟเส้นแสดงจำนวนการทำนายในแต่ละวัน
- **ความแม่นยำ**: กราฟพื้นที่แสดงความแม่นยำเฉลี่ย
- **การแจกแจง**: กราฟวงกลมแสดงสัดส่วนผลการทำนาย

### 🔍 ทดสอบระบบ
- อัปโหลดรูปภาพพระเครื่อง
- ทำการทำนายด้วย AI
- แสดงผลลัพธ์และความเชื่อมั่น

### 📊 สถานะระบบ
- สถานะ API Server
- สถานะ AI Model
- ข้อมูลประสิทธิภาพระบบ

## 🎨 การปรับแต่ง CSS

ไฟล์ `style.css` ประกอบด้วย:

- **Color Scheme**: ใช้ Gradient สีน้ำเงิน-ม่วง
- **Card Design**: ดีไซน์การ์ดแบบ Modern
- **Button Styles**: ปุ่มแบบ Gradient พร้อม Hover Effects
- **Animation**: เอฟเฟคการเคลื่อนไหวแบบ Smooth
- **Responsive**: รองรับหน้าจอมือถือ

## 🔧 การปรับแต่งระบบ

### เปลี่ยน Port
แก้ไขใน `run_frontend.py`:
```python
"--server.port", "8501"  # เปลี่ยนเป็นพอร์ตที่ต้องการ
```

### เปลี่ยน API URL
แก้ไขใน `main_streamlit_app.py`:
```python
base_url = "http://localhost:8000"  # เปลี่ยน URL ของ API
```

### เพิ่มสีธีม
แก้ไขใน `style.css`:
```css
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
}
```

## 🐛 การแก้ไขปัญหา

### ปัญหา: แอปไม่เริ่มต้น
- ตรวจสอบว่าติดตั้ง Dependencies ครบแล้ว
- ใช้คำสั่ง `python run_frontend.py` แทน

### ปัญหา: CSS ไม่โหลด
- ตรวจสอบว่าไฟล์ `style.css` อยู่ในตำแหน่งเดียวกันกับ `main_streamlit_app.py`
- ลองรีเฟรชหน้าเว็บ

### ปัญหา: การเชื่อมต่อ API
- ตรวจสอบว่า API Server ทำงานอยู่
- ตรวจสอบ URL ใน `call_api()` function

## 📝 การพัฒนาต่อ

### เพิ่มฟีเจอร์ใหม่
1. เพิ่ม function ใน `main_streamlit_app.py`
2. อัปเดต CSS ใน `style.css`
3. ทดสอบด้วย `python run_frontend.py`

### เชื่อมต่อ Database
- เพิ่ม database connector
- สร้าง functions สำหรับ CRUD operations
- อัปเดต requirements.txt

## 🔐 ความปลอดภัย

- ไม่เก็บข้อมูลผู้ใช้ใน Frontend
- ใช้ HTTPS ในการ deploy จริง
- ตรวจสอบ input validation

## 📈 Performance

- ใช้ caching ของ Streamlit: `@st.cache_data`
- โหลดข้อมูลแบบ lazy loading
- ปรับขนาดรูปภาพก่อนประมวลผล

## 🌐 การ Deploy

### Local Development
```bash
python run_frontend.py
```

### Production (Streamlit Cloud)
1. Push โค้ดไป GitHub
2. เชื่อมต่อกับ Streamlit Cloud
3. Deploy จาก `main_streamlit_app.py`

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "main_streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📞 การติดต่อ

- **Email**: support@amulet-ai.com
- **Documentation**: [Link to docs]
- **GitHub Issues**: [Link to issues]

---

**เวอร์ชัน**: 2.1.0  
**อัปเดตล่าสุด**: September 2025