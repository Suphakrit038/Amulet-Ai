# 📋 แผนการรวม Dataset และเทรนโมเดลใหม่

**📅 วันที่:** October 2, 2025  
**🎯 เป้าหมาย:** รวม dataset, ลบโมเดลเก่า, และเทรนโมเดลใหม่

---

## 🗂️ Phase 1: การรวมและจัดระเบียบ Dataset

### **🔍 Step 1.1: ตรวจสอบสถานะปัจจุบัน**
- ✅ ตรวจสอบโครงสร้าง dataset ปัจจุบัน
- ✅ วิเคราะห์ข้อมูลที่มีอยู่
- ✅ ระบุไฟล์ที่ซ้ำซ้อน

**สถานะปัจจุบัน:**
```
organized_dataset/
├── raw/main_dataset/          # ข้อมูลต้นฉบับ (172 ไฟล์)
├── processed/                 # ข้อมูล processed (172 ไฟล์)
├── augmented/                 # ข้อมูล augmented (860+ ไฟล์)
├── splits/                    # แบ่ง train/val/test (1,076 ไฟล์)
└── DATA SET/                  # ข้อมูลต้นฉบับซ้ำ (172 ไฟล์)
```

### **🧹 Step 1.2: ล้างข้อมูลซ้ำซ้อน**
- ลบโฟลเดอร์ `DATA SET` (ซ้ำกับ `raw/main_dataset/`)
- ลบโฟลเดอร์ย่อย `back/` และ `front/` ที่ว่าง
- รวมข้อมูลจาก `processed/` และ `augmented/` เข้าด้วยกัน
- สร้างโครงสร้างใหม่ที่เป็นระเบียบ

### **📁 Step 1.3: สร้างโครงสร้าง Dataset ใหม่**
```
final_dataset/
├── train/                     # 751 ไฟล์ (70%)
│   ├── phra_sivali/
│   ├── portrait_back/
│   ├── prok_bodhi_9_leaves/
│   ├── somdej_pratanporn_buddhagavak/
│   ├── waek_man/
│   └── wat_nong_e_duk/
├── val/                       # 214 ไฟล์ (20%)
└── test/                      # 111 ไฟล์ (10%)
```

---

## 🗑️ Phase 2: ลบโมเดลเก่าและไฟล์ที่ไม่จำเป็น

### **🔍 Step 2.1: ตรวจสอบโมเดลเก่า**
- ตรวจสอบโฟลเดอร์ `trained_model/`
- ระบุไฟล์โมเดลที่ต้องลบ

### **🗑️ Step 2.2: ลบโมเดลเก่า**
- ลบไฟล์ `.pth`, `.pkl`, `.joblib`
- ลบ checkpoint เก่า
- เก็บเฉพาะ config files

### **🧹 Step 2.3: ล้าง cache และ temporary files**
- ลบโฟลเดอร์ `__pycache__/`
- ลบ log files เก่า
- ล้าง GPU memory cache

---

## 🤖 Phase 3: เตรียมโมเดลใหม่

### **🏗️ Step 3.1: สร้าง Model Architecture**
- เลือก backbone: ResNet50 หรือ EfficientNet-B3
- กำหนด input size: 224×224×3
- กำหนด output classes: 6 คลาส

### **⚡ Fast Training Configuration**
```python
FAST_TRAINING_CONFIG = {
    'model': 'resnet50',           # เร็วกว่า EfficientNet
    'input_size': (224, 224),
    'num_classes': 6,
    'batch_size': 64,              # เพิ่ม batch size
    'learning_rate': 3e-4,         # เพิ่ม learning rate
    'epochs': 25,                  # ลดจาก 100 เหลือ 25
    'optimizer': 'SGD',            # เร็วกว่า AdamW
    'scheduler': 'StepLR',         # เร็วกว่า CosineAnnealing
    'early_stopping': True,
    'patience': 5                  # หยุดเร็วถ้าไม่ดีขึ้น
}
```

### **🚀 Speed Optimizations:**
- ใช้ dataset ที่จัดเรียบร้อยแล้วใน `splits/`
- ใช้ ResNet50 แทน EfficientNet (เร็วกว่า 2x)
- ลด epochs เหลือ 25 แทน 100
- เพิ่ม batch size เป็น 64
- ข้าม hyperparameter tuning
- ใช้ simple evaluation

### **📊 Step 3.3: สร้าง Data Loaders**
- สร้าง training transforms
- สร้าง validation transforms
- กำหนด batch size และ sampling strategy

---

## 🚀 Phase 4: การเทรนโมเดล

### **📈 Step 4.1: Initial Training**
- Baseline training (10 epochs)
- ตรวจสอบ overfitting
- วิเคราะห์ loss curves

### **🎯 Step 4.2: Hyperparameter Tuning**
- ปรับ learning rate
- ปรับ batch size
- ทดสอบ augmentation strategies

### **⚡ Step 4.3: Full Training**
- เทรนเต็ม 100 epochs
- ใช้ early stopping
- บันทึก best model

### **📊 Step 4.4: Model Evaluation**
- ประเมินผลบน test set
- สร้าง confusion matrix
- คำนวณ per-class metrics

---

## 📋 Execution Plan (⚡ FAST VERSION)

### **� Timeline (ลดเวลาลง 60%):**

**Phase 1 (10 นาที):**
- [x] Step 1.1: ตรวจสอบสถานะ (เสร็จแล้ว)
- [ ] Step 1.2: ล้างข้อมูลซ้ำ + รวมโครงสร้าง (10 นาที) ⚡

**Phase 2 (5 นาที):**
- [ ] Step 2.1-2.3: ลบโมเดลเก่า + ล้าง cache (5 นาที) ⚡

**Phase 3 (5 นาที):**
- [ ] Step 3.1-3.3: ใช้ template พร้อม + config ที่มีอยู่ (5 นาที) ⚡

**Phase 4 (30-45 นาที):**
- [ ] Step 4.1: Quick training (20-30 epochs) (20-30 นาที) ⚡
- [ ] Step 4.2: Simple evaluation (5-10 นาที) ⚡

---

## 💾 Expected Results (Fast Version)

### **📊 Dataset Summary:**
- **ใช้ dataset ปัจจุบัน:** `splits/` (1,076 ไฟล์)
- **Classes:** 6 คลาส
- **Split ratio:** 70:20:10 (ที่มีอยู่แล้ว)

### **🎯 Model Performance Target (Realistic):**
- **Training Accuracy:** > 90% (ลดจาก 95%)
- **Validation Accuracy:** > 75% (ลดจาก 85%)
- **Test Accuracy:** > 70% (ลดจาก 80%)
- **Training Time:** 20-30 นาที (ลดจาก 60-120 นาที)

### **⏱️ Total Time: 50-65 นาที** (ลดจาก 2-3 ชั่วโมง)

### **📁 Output Files:**
```
trained_model/
├── best_model.pth          # Best model weights
├── final_model.pth         # Final model weights
├── training_history.json   # Training logs
├── model_config.json       # Model configuration
├── class_mapping.json      # Class mappings
└── evaluation_report.json  # Performance metrics
```

---

## ❓ Ready to Start?

**คุณพร้อมที่จะเริ่ม Phase 1 หรือยัง?**

กรุณาตอบ "พร้อม" หรือถ้าต้องการแก้ไขแผนใดๆ โปรดแจ้งมาครับ!