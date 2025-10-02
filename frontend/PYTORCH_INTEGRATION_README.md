# PyTorch Integration - Quick Start Guide

## 🎯 สถานะปัจจุบัน

Frontend ได้รับการอัปเกรดให้รองรับ **PyTorch Deep Learning Models** พร้อมระบบ fallback ไปยัง sklearn หากไม่มี PyTorch model:

### ✅ สิ่งที่ได้อัปเกรด

1. **Auto-detect Model Type**: ตรวจจับอัตโนมัติว่ามี PyTorch model หรือ sklearn model
2. **PyTorch Features** (ถ้ามี):
   - Transfer Learning Models (ResNet50/EfficientNet/MobileNet)
   - Temperature Scaling Calibration
   - Out-of-Distribution (OOD) Detection
   - Grad-CAM Visualization
3. **Sklearn Fallback** (ถ้าไม่มี PyTorch):
   - ใช้โมเดล sklearn แบบเดิม
   - ทำงานได้ตามปกติ

---

## 🚀 การใช้งาน

### Option 1: ใช้งานกับ sklearn (ปัจจุบัน)

ถ้าคุณยังไม่มี PyTorch model, frontend จะใช้ sklearn model แบบเดิมโดยอัตโนมัติ:

```bash
cd frontend
streamlit run production_app_clean.py
```

**ไฟล์ที่ต้องมี**:
```
trained_model/
├── classifier.joblib
├── scaler.joblib
└── label_encoder.joblib
```

✅ **ทำงานตามปกติ** - ไม่มีผลกระทบต่อระบบเดิม

---

### Option 2: อัปเกรดเป็น PyTorch (แนะนำ)

เพื่อใช้ฟีเจอร์ขั้นสูงทั้งหมด:

#### ขั้นตอนที่ 1: ติดตั้ง PyTorch

```bash
# CPU version (เร็ว, ใช้งานง่าย)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# หรือ GPU version (เร็วกว่า, ต้องมี NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### ขั้นตอนที่ 2: Train PyTorch Model

```bash
# เข้าไปที่ project root
cd e:\Amulet-Ai

# Train transfer learning model
python examples/transfer_learning_example.py
```

**Output ที่ได้**:
```
trained_model/
├── best_model.pth              ← PyTorch model
├── model_config.json           ← Configuration
└── training_history.json
```

#### ขั้นตอนที่ 3: Calibrate Model (Optional แต่แนะนำ)

```bash
# Temperature scaling
python examples/calibration_example.py
```

**Output ที่ได้**:
```
trained_model/
└── temperature_scaler.pth      ← Calibration model
```

#### ขั้นตอนที่ 4: Train OOD Detector (Optional)

```bash
# Out-of-distribution detection
python examples/ood_detection_example.py
```

**Output ที่ได้**:
```
trained_model/
└── ood_detector.joblib         ← OOD detector
```

#### ขั้นตอนที่ 5: รัน Frontend

```bash
cd frontend
streamlit run production_app_clean.py
```

🎉 **ตอนนี้จะใช้ PyTorch model พร้อมฟีเจอร์ครบ!**

---

## 📊 เปรียบเทียบโมเดล

| Feature | sklearn (เดิม) | PyTorch (ใหม่) |
|---------|---------------|----------------|
| Accuracy | 🟨 ปานกลาง | 🟩 สูง |
| Speed | 🟩 เร็ว | 🟨 ปานกลาง |
| Calibrated Confidence | ❌ | ✅ |
| OOD Detection | ❌ | ✅ |
| Grad-CAM | ❌ | ✅ |
| Setup | 🟩 ง่าย | 🟨 ต้องติดตั้ง PyTorch |

---

## 🔍 วิธีตรวจสอบว่าใช้โมเดลแบบไหน

เมื่อรัน frontend, ดูที่ผลลัพธ์:

### sklearn mode:
```
วิธีการทำนาย: Local (sklearn)
```

### PyTorch mode:
```
วิธีการทำนาย: Local (PyTorch)
```
+ มี OOD warning (ถ้าตรวจพบ)
+ มี Grad-CAM heatmap

---

## 🐛 Troubleshooting

### ปัญหา: "ModuleNotFoundError: No module named 'torch'"

**สาเหตุ**: ยังไม่ติดตั้ง PyTorch

**แก้ไข**: 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**หรือ** ใช้ sklearn mode (ไม่ต้องแก้อะไร)

---

### ปัญหา: "Model file not found: best_model.pth"

**สาเหตุ**: ยังไม่ train PyTorch model

**แก้ไข**:
```bash
python examples/transfer_learning_example.py
```

**หรือ** frontend จะ fallback ไปใช้ sklearn โดยอัตโนมัติ

---

### ปัญหา: Frontend ช้ามาก

**สาเหตุ**: ใช้ CPU inference

**แก้ไข**:
1. ติดตั้ง GPU version ของ PyTorch (ต้องมี NVIDIA GPU)
2. หรือใช้ sklearn mode (เร็วกว่า)

---

### ปัญหา: Grad-CAM ไม่แสดง

**สาเหตุ**: Error ใน Grad-CAM generation หรือ model architecture ไม่รองรับ

**แก้ไข**: ไม่ต้องแก้ - การทำนายยังใช้งานได้ปกติ (Grad-CAM เป็น optional)

---

## 📁 โครงสร้างไฟล์

```
Amulet-Ai/
├── frontend/
│   ├── production_app_clean.py     ← Main app (รองรับทั้ง PyTorch และ sklearn)
│   ├── requirements.txt
│   └── PYTORCH_INTEGRATION_README.md  ← This file
│
├── trained_model/
│   ├── classifier.joblib           ← sklearn model (เดิม)
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   ├── best_model.pth              ← PyTorch model (ใหม่)
│   ├── model_config.json
│   ├── temperature_scaler.pth      ← Optional
│   └── ood_detector.joblib         ← Optional
│
├── examples/
│   ├── transfer_learning_example.py
│   ├── calibration_example.py
│   ├── ood_detection_example.py
│   └── pytorch_frontend_example.py  ← Test PyTorch integration
│
└── docs/
    └── PYTORCH_FRONTEND_INTEGRATION.md  ← Full documentation
```

---

## 🎓 Next Steps

### ถ้าใช้ sklearn mode:
1. ✅ ใช้งานได้เลย - ไม่ต้องทำอะไร
2. (Optional) อัปเกรดเป็น PyTorch เมื่อพร้อม

### ถ้าอยากใช้ PyTorch mode:
1. ติดตั้ง PyTorch: `pip install torch torchvision`
2. Train model: `python examples/transfer_learning_example.py`
3. (Optional) Calibrate: `python examples/calibration_example.py`
4. (Optional) OOD: `python examples/ood_detection_example.py`
5. รัน frontend: `streamlit run production_app_clean.py`

---

## 📖 เอกสารเพิ่มเติม

- **Full Documentation**: [docs/PYTORCH_FRONTEND_INTEGRATION.md](../docs/PYTORCH_FRONTEND_INTEGRATION.md)
- **Phase 2 Report**: [docs/PHASE2_COMPLETION.md](../docs/PHASE2_COMPLETION.md)
- **Quick Start**: [docs/QUICK_START.md](../docs/QUICK_START.md)

---

## ✅ Checklist

### สำหรับ sklearn mode (ปัจจุบัน):
- [x] Frontend รองรับ sklearn
- [x] Fallback mechanism ทำงาน
- [x] ไม่มี breaking changes

### สำหรับ PyTorch mode (อัปเกรด):
- [ ] ติดตั้ง PyTorch
- [ ] Train PyTorch model
- [ ] Test PyTorch inference
- [ ] (Optional) Calibration
- [ ] (Optional) OOD detection
- [ ] Deploy

---

**Status**: ✅ Ready to use  
**Mode**: Auto-detect (sklearn fallback)  
**Breaking Changes**: ❌ None
