# 🔮 Amulet-AI - ระบบวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)
![Status](https://img.shields.io/badge/status-production--ready-success)

## 🌟 ระบบที่ปรับปรุงใหม่ (System Modernization Complete)

ระบบ Amulet-AI ได้รับการปรับปรุงให้เป็นระบบที่ทันสมัย มีประสิทธิภาพสูง และพร้อมใช้งานจริง โดยมีการปรับปรุงครอบคลุม 4 Phase:

### ✅ Phase 1: โครงสร้างและการจัดการโค้ด
- ลบไฟล์ซ้ำซ้อนทั้งหมด
- รวมโค้ดเป็นโมดูลาร์
- จัดการ configuration แบบรวมศูนย์
- ระบบ analytics และ performance monitoring

### ✅ Phase 2: ปรับปรุงโมเดล AI  
- รองรับ Vision Transformer (ViT)
- EfficientNetV2 และ ConvNeXt architectures
- Mixed precision training
- Knowledge distillation

### ✅ Phase 3: UI/UX ที่ทันสมัย
- Responsive design รองรับมือถือ
- CSS Design System
- Enhanced UI components
- Thai language optimization

### ✅ Phase 4: การปรับใช้งานและตรวจสอบ
- Production deployment scripts
- Health monitoring system
- Performance metrics
- Error handling

## 🚀 การเริ่มใช้งาน (Quick Start)

### วิธีที่ 1: ใช้ Batch Script (แนะนำ)
```bash
# Windows
start_amulet_ai.bat
```

### วิธีที่ 2: ใช้ Python Launcher
```bash
python launch_amulet_ai.py
```

### วิธีที่ 3: รันตรงๆ
```bash
# ติดตั้ง dependencies
pip install -r requirements.txt

# รัน frontend
streamlit run frontend/app_modern.py

# รัน backend (optional)
python backend/api_with_real_model.py
```

## 📋 ความต้องการระบบ (System Requirements)

### พื้นฐาน (Basic)
- **Python**: 3.8 หรือสูงกว่า
- **RAM**: 4GB ขึ้นไป  
- **Storage**: 2GB พื้นที่ว่าง
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### แนะนำ (Recommended)
- **Python**: 3.9+
- **RAM**: 8GB ขึ้นไป
- **GPU**: NVIDIA GTX 1060 ขึ้นไป (สำหรับ AI model)
- **Storage**: 5GB พื้นที่ว่าง

## 📦 การติดตั้ง Dependencies

### การติดตั้งพื้นฐาน
```bash
pip install streamlit requests pillow numpy
```

### การติดตั้งเต็มรูปแบบ
```bash
pip install -r requirements.txt
```

### สำหรับ AI Model (Optional)
```bash
pip install torch torchvision timm transformers
```

## 🎯 คุณสมบัติหลัก (Main Features)

### 🖼️ การวิเคราะห์รูปภาพ
- รองรับไฟล์: JPG, PNG, WebP, BMP, TIFF
- ขนาดไฟล์สูงสุด: 10MB
- ตรวจสอบคุณภาพรูปภาพอัตโนมัติ
- แสดงผลแบบ real-time

### 🤖 AI Model
- Vision Transformer สำหรับความแม่นยำสูง
- EfficientNetV2 สำหรับความเร็ว
- Mixed precision สำหรับประหยัดหน่วยความจำ
- รองรับการเรียนรู้แบบ transfer learning

### 🎨 User Interface
- Responsive design รองรับทุกอุปกรณ์
- Thai language support
- Dark/Light mode
- Modern CSS framework
- Touch-friendly สำหรับมือถือ

### 📊 Monitoring & Analytics
- Real-time performance metrics
- System health monitoring  
- Usage analytics
- Error tracking
- Alert system

## 🌐 การเข้าถึงระบบ (System Access)

หลังจากเริ่มระบบแล้ว:

- **Frontend (หน้าเว็บหลัก)**: http://localhost:8501
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

## 📱 การใช้งานบนมือถือ (Mobile Usage)

ระบบรองรับการใช้งานบนมือถือ:
- เปิดเบราว์เซอร์บนมือถือ
- ไปที่ `http://[your-ip]:8501`
- UI จะปรับตัวอัตโนมัติ

## 🔧 การตั้งค่า (Configuration)

### การตั้งค่าพื้นฐาน
แก้ไขไฟล์ `frontend/config.py`:

```python
# API Settings
API_URL = "http://localhost:8001"

# Image Settings
IMAGE_SETTINGS = {
    "MAX_FILE_SIZE_MB": 10,
    "SUPPORTED_FORMATS": ["jpg", "jpeg", "png", "webp"]
}

# UI Settings
UI_SETTINGS = {
    "PRIMARY_COLOR": "#2563EB",
    "THEME": "light"  # or "dark"
}
```

### การตั้งค่า AI Model
แก้ไขไฟล์ `ai_models/modern_model.py`:

```python
# Model Configuration
MODEL_CONFIG = {
    "architecture": "efficientnet_v2_s",  # or "vit_base_patch16_224"
    "num_classes": 10,
    "use_mixed_precision": True
}
```

## 📁 โครงสร้างไฟล์ (File Structure)

```
Amulet-AI/
├── 🚀 start_amulet_ai.bat          # Start script (Windows)
├── 🐍 launch_amulet_ai.py          # Python launcher
├── 🧪 test_system.py               # System test
├── 📋 requirements.txt             # Dependencies
│
├── frontend/                       # Frontend application
│   ├── 🎨 app_modern.py           # Modern UI app
│   ├── ⚙️ config.py                # Configuration
│   ├── 📊 analytics.py             # Analytics system
│   ├── 🔧 amulet_unified.py        # Core functions
│   ├── assets/css/main.css         # CSS framework
│   └── components/                 # UI components
│       ├── ui_components.py
│       ├── layout_manager.py
│       ├── image_validator.py
│       └── result_display.py
│
├── backend/                        # Backend API
│   └── 🔌 api_with_real_model.py   # FastAPI server
│
├── ai_models/                      # AI models
│   ├── 🤖 modern_model.py          # Modern AI architecture
│   └── configs/                    # Model configurations
│
├── logs/                          # System logs
├── uploads/                       # Uploaded images
└── data_base/                     # Image database
```

## 🔍 การแก้ไขปัญหา (Troubleshooting)

### ปัญหาที่พบบ่อย

#### 1. ไม่สามารถเริ่มระบบได้
```bash
# ตรวจสอบ Python version
python --version

# ตรวจสอบ dependencies
python test_system.py
```

#### 2. Backend ไม่ทำงาน
```bash
# ตรวจสอบ port
netstat -an | findstr :8001

# รัน backend แยก
python backend/api_with_real_model.py
```

#### 3. Frontend แสดงผิดพลาด
```bash
# Clear cache
streamlit cache clear

# รัน debug mode
streamlit run frontend/app_modern.py --logger.level debug
```

#### 4. AI Model ไม่ทำงาน
```bash
# ติดตั้ง AI dependencies
pip install torch torchvision timm

# ตรวจสอบ GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### การรีเซ็ตระบบ
```bash
# ลบ cache
rm -rf .streamlit/
rm -rf __pycache__/
rm -rf frontend/__pycache__/

# รีสตาร์ทระบบ
python launch_amulet_ai.py
```

## 🎓 การใช้งานขั้นสูง (Advanced Usage)

### การใช้งาน API ตรงๆ

```python
import requests

# Upload image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8001/predict',
        files={'file': f}
    )
    result = response.json()
    print(result)
```

### การ Customize UI

```python
# เพิ่ม custom CSS
st.markdown("""
<style>
.custom-style {
    background: linear-gradient(45deg, #667eea, #764ba2);
}
</style>
""", unsafe_allow_html=True)
```

### การเพิ่ม AI Model ใหม่

```python
# สร้างไฟล์ ai_models/my_custom_model.py
class CustomAmuletModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your custom architecture
        
    def forward(self, x):
        # Your forward pass
        return x
```

## 📈 Performance Tips

### การเพิ่มความเร็ว
1. **ใช้ GPU**: ติดตั้ง CUDA สำหรับ PyTorch
2. **Optimize Images**: ปรับขนาดรูปภาพก่อนอัปโหลด
3. **Enable Caching**: ตั้งค่า caching ในไฟล์ config
4. **Use SSD**: ใช้ SSD สำหรับเก็บข้อมูล

### การลดการใช้หน่วยความจำ
1. **Mixed Precision**: เปิดใช้ในการตั้งค่า AI model
2. **Batch Size**: ลดขนาด batch ถ้าหน่วยความจำไม่เพียงพอ
3. **Model Pruning**: ใช้โมเดลที่เล็กกว่า

## 🤝 การสนับสนุน (Support)

### การรายงานปัญหา
- สร้าง issue ใน GitHub repository
- แนบ log files จากโฟลเดอร์ `logs/`
- ระบุ OS และ Python version

### การขอฟีเจอร์ใหม่
- สร้าง feature request ใน GitHub
- อธิบายการใช้งานที่ต้องการ
- แนบตอนอย่าง mockup ถ้ามี

## 📄 License

MIT License - ดูรายละเอียดในไฟล์ LICENSE

## 🙏 Credits

- **Streamlit**: Web framework
- **FastAPI**: Backend API framework  
- **PyTorch**: AI/ML framework
- **Pillow**: Image processing
- **timm**: Pre-trained models

---

## 🎉 Ready to Use!

ระบบ Amulet-AI ได้รับการปรับปรุงให้เป็นระบบที่ทันสมัย มีประสิทธิภาพสูง และพร้อมใช้งานจริงแล้ว!

**เริ่มใช้งานได้เลย**: `start_amulet_ai.bat` หรือ `python launch_amulet_ai.py`

---
*Last Updated: September 2025 - Amulet-AI v2.0.0*
