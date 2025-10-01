# 🏺 Amulet-AI Unified Frontend

## ✨ ภาพรวม
ไฟล์ `main_streamlit_app.py` คือระบบ Frontend แบบรวมฟีเจอร์ทั้งหมดในไฟล์เดียว ที่รวบรวมความสามารถจากทุกเวอร์ชันมาไว้ด้วยกัน

## 🎯 ฟีเจอร์ที่รวมเข้ามา

### 🎨 จาก Enhanced Version
- ✅ Multi-tab interface (Classification, Analytics, Documentation, Tools)
- ✅ System status monitoring แบบเรียลไทม์
- ✅ Performance analytics และ metrics
- ✅ Debug mode สำหรับ developers
- ✅ Enhanced error handling ที่ครอบคลุม

### 🇹🇭 จาก Simple Version  
- ✅ Thai-themed UI design สวยงาม
- ✅ Camera support สำหรับถ่ายรูปในเบราว์เซอร์
- ✅ Loading animations และ effects
- ✅ Partnership logos (DEPA, Thai-Austrian)
- ✅ Responsive design ที่ใช้งานง่าย

### ⚡ จาก Production Version
- ✅ Modular architecture
- ✅ Performance monitoring system
- ✅ Advanced image processing
- ✅ Comprehensive error handling
- ✅ Production-ready code structure

## 🚀 การใช้งาน

### เริ่มระบบ
```bash
cd E:\Amulet-Ai
python -m streamlit run frontend/main_streamlit_app.py --server.port 8503
```

### เข้าถึงระบบ
- **Local:** http://localhost:8503
- **Network:** http://192.168.1.41:8503

## 🎛️ โหมดการวิเคราะห์

### 📱 Single Image Mode
- อัปโหลดรูปเดียว (ไฟล์ หรือ กล้อง)
- เหมาะสำหรับการทดสอบเร็ว
- ผลลัพธ์ทันที

### 🖼️ Dual Image Mode (แนะนำ)
- อัปโหลดรูปทั้งหน้าและหลัง
- การวิเคราะห์ที่แม่นยำขึ้น
- เปรียบเทียบผลลัพธ์จากทั้งสองด้าน
- ตรวจจับความสอดคล้องของผลลัพธ์

## 📊 Tabs หลัก

### 🖼️ Image Classification
- อัปโหลดและจำแนกรูปพระเครื่อง
- รองรับ JPG, PNG
- แสดงความเชื่อมั่นและความน่าจะเป็น
- Tips การถ่ายรูปที่ดี

### 📊 System Analytics  
- Performance metrics แบบเรียลไทม์
- Model statistics และ accuracy
- Analysis history ย้อนหลัง
- System resource monitoring

### 📚 Documentation
- คู่มือการใช้งานแบบครบถ้วน
- รายการพระเครื่องที่รองรับ
- Technical specifications
- API endpoints documentation

### 🔧 System Tools
- Health check ระบบ
- Cache management
- Settings export/import
- System information
- Reset functions

## 🎯 ประเภทพระเครื่องที่รองรับ

1. **พระศิวลี** (Phra Sivali) - พระป้องกันภัยอันตราย
2. **พระสมเด็จ** (Phra Somdej) - พระยอดนิยมที่สุด
3. **ปรกโพธิ์ 9 ใบ** (Prok Bodhi 9 Leaves) - พระคุ้มครองจากภัยธรรมชาติ
4. **แหวกม่าน** (Waek Man) - พระแคล้วคลาดภัยอันตราย
5. **หลังรูปเหมือน** (Portrait Back) - พระที่มีรูปเหมือนหลัง
6. **วัดหนองอีดุก** (Wat Nong E Duk) - พระจากวัดหนองอีดุก

## ⚙️ การตั้งค่าใน Sidebar

### System Control
- 📊 System Status - สถานะ API และ Model
- 📈 Quick Stats - CPU, Memory, Accuracy
- 🔧 Advanced Options - Debug mode, Confidence details

### Analysis Options
- Show Confidence Details
- Show All Probabilities  
- Debug Mode (สำหรับ developers)
- Analysis Mode (Single/Dual)

## 🔄 API Fallback System

ระบบจะลองใช้ API ก่อน หากไม่สำเร็จจะใช้ Local prediction:

1. **API Mode** - ใช้ FastAPI server (แนะนำ)
2. **Local Mode** - ใช้ model ในเครื่องโดยตรง
3. **Automatic Fallback** - เปลี่ยนโหมดอัตโนมัติ

## 💡 Tips การใช้งาน

### 📸 การถ่ายรูปที่ดี
- ใช้แสงสว่างเพียงพอ
- รูปภาพชัดเจน ไม่เบลอ
- พื้นหลังสีเรียบ
- ขนาดไฟล์ไม่เกิน 10MB
- ถ่ายให้เห็นรายละเอียดของพระเครื่อง

### 📊 การตีความผลลัพธ์
- **>90%**: ความเชื่อมั่นสูงมาก ✅
- **70-90%**: ความเชื่อมั่นปานกลาง ⚠️
- **<70%**: ความเชื่อมั่นต่ำ ❌
- **เวลาประมวลผล**: 2-10 วินาที
- **คำแนะนำ**: ใช้ข้อมูลร่วมกับความรู้ของผู้เชี่ยวชาญ

## 🗂️ โครงสร้างไฟล์ใหม่

```
frontend/
├── main_streamlit_app.py          # ← ไฟล์หลักรวมทุกอย่าง
├── backup_old_versions/           # ← ไฟล์เก่าที่สำรองไว้
│   ├── enhanced_streamlit_app.py
│   ├── main_streamlit_app_simple.py
│   ├── main_streamlit_app_problematic.py
│   └── main_app_unified.py
├── components/                    # ← Components ที่ใช้ถ้ามี
├── utils/                        # ← Utilities ที่ใช้ถ้ามี
├── imgae/                        # ← โลโก้และรูปภาพ
├── style.css                     # ← CSS เพิ่มเติม
└── README.md                     # ← คู่มือนี้
```

## 🔧 Technical Details

### Dependencies ที่ใช้
```python
streamlit>=1.25.0
requests>=2.31.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
psutil>=5.9.0
joblib>=1.3.0
```

### System Requirements
- **Python**: 3.8+
- **RAM**: 4GB+
- **Storage**: 2GB+
- **Browser**: Chrome, Firefox, Safari
- **Internet**: สำหรับ API mode

### Model Specifications
- **Algorithm**: Random Forest Classifier
- **Input Size**: 224x224 pixels
- **Features**: 150,528 (flattened pixels)
- **Classes**: 6 amulet types
- **Accuracy**: ~72% on test dataset

## 🚨 Troubleshooting

### ปัญหาที่อาจพบ

1. **API ไม่ทำงาน**
   - ตรวจสอบ FastAPI server รันอยู่หรือไม่
   - ระบบจะใช้ Local mode อัตโนมัติ

2. **Model ไม่พบ**
   - ตรวจสอบไฟล์ใน `trained_model/`
   - รัน training script ใหม่

3. **Camera ไม่ทำงาน**
   - อนุญาต browser เข้าถึงกล้อง
   - ใช้ HTTPS หรือ localhost

4. **ความช้า**
   - ลด image size ก่อน upload
   - ปิด debug mode
   - ล้าง cache

### การรีเซ็ตระบบ
```python
# ใน System Tools tab
1. Clear Cache
2. Clear Analysis History  
3. Reset to Defaults
```

## 📞 การสนับสนุน

- **GitHub Issues**: สำหรับ bug reports
- **Documentation**: ในตัวระบบ
- **API Docs**: `/docs` endpoint

---

## 🎉 สรุป

ไฟล์ `main_streamlit_app.py` ตัวใหม่นี้รวมความสามารถจากทุกเวอร์ชันเก่า:
- ✅ **ใช้งานง่าย** - Interface ที่เข้าใจง่าย  
- ✅ **ครบครัน** - ฟีเจอร์ที่จำเป็นทั้งหมด
- ✅ **เสถียร** - Error handling ที่ดี
- ✅ **รวดเร็ว** - Performance ที่ปรับปรุงแล้ว
- ✅ **สวยงาม** - UI/UX ที่ดีขึ้น

**ไม่ต้องสับสนว่าจะใช้ไฟล์ไหนแล้ว - ใช้ไฟล์เดียวนี้ได้ทุกอย่าง! 🚀**