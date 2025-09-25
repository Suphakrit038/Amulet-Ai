# 📝 FILE NAMING STANDARDS PROPOSAL
## ข้อเสนอการปรับปรุงชื่อไฟล์ให้เป็นมาตรฐานมืออาชีพ

**วันที่จัดทำ**: 26 กันยายน 2025  
**จุดประสงค์**: ปรับปรุงชื่อไฟล์ให้เป็นมาตรฐานสากล สะท้อนหน้าที่การใช้งาน และง่ายต่อการบำรุงรักษา

---

## 🎯 **หลักการตั้งชื่อไฟล์มืออาชีพ**

### **✅ มาตรฐานที่ดี**
- **Snake Case**: `model_loader.py`, `data_processor.py`
- **Descriptive**: ชื่อสะท้อนหน้าที่ได้ชัดเจน
- **Consistent**: รูปแบบเดียวกันทั้งโปรเจค
- **Hierarchical**: จัดกลุ่มตามหน้าที่

### **❌ ปัญหาปัจจุบัน**
- ชื่อไฟล์ยาวเกินไป: `api_with_real_model.py`
- ไม่สม่ำเสมอ: `app_streamlit.py` vs `amulet_unified.py`
- ชื่อคลุมเครือ: `final_steps_6_and_7.py`
- ซ้ำซ้อน: `real_model_loader.py` (หลายไฟล์)

---

## 🔄 **การเปลี่ยนแปลงที่เสนอ**

### **1. 🎨 Frontend Applications**

| **ไฟล์เดิม** | **ไฟล์ใหม่** | **เหตุผล** |
|-------------|-------------|------------|
| `frontend/app_streamlit.py` | `frontend/main_app.py` | ✅ แอปหลัก ชื่อสั้นและชัดเจน |
| `frontend/app_modern.py` | `frontend/modern_ui.py` | ✅ เน้น UI มากกว่า app |
| `frontend/amulet_unified.py` | `frontend/unified_interface.py` | ✅ เน้นการรวมระบบ |
| `frontend/analytics.py` | `frontend/analytics_dashboard.py` | ✅ ระบุประเภท dashboard |

### **2. 🔧 Backend API Services**

| **ไฟล์เดิม** | **ไฟล์ใหม่** | **เหตุผล** |
|-------------|-------------|------------|
| `backend/api/api.py` | `backend/api/main_api.py` | ✅ API หลัก ชื่อชัดเจน |
| `backend/api/integrated_api.py` | `backend/api/production_api.py` | ✅ เน้นการใช้งาน production |
| `backend/api/api_with_real_model.py` | `backend/api/ai_model_api.py` | ✅ เน้น AI model service |
| `backend/api/api_with_reference_images.py` | `backend/api/reference_api.py` | ✅ เน้น reference images |
| `backend/api/optimized_api.py` | `backend/api/performance_api.py` | ✅ เน้น performance |

### **3. 🤖 AI Models & Training**

| **ไฟล์เดิม** | **ไฟล์ใหม่** | **เหตุผล** |
|-------------|-------------|------------|
| `ai_models/final_steps_6_and_7.py` | `ai_models/deployment_pipeline.py` | ✅ ระบุหน้าที่ deployment |
| `ai_models/hybrid_implementation.py` | `ai_models/hybrid_model.py` | ✅ เน้นโมเดลมากกว่า implementation |
| `ai_models/lightweight_ml_system.py` | `ai_models/lightweight_model.py` | ✅ สั้นและเน้นโมเดล |
| `ai_models/compatible_data_pipeline.py` | `ai_models/data_processor.py` | ✅ เน้นการประมวลผลข้อมูล |
| `ai_models/compatible_visualizer.py` | `ai_models/visualization_tools.py` | ✅ เน้น tools มากกว่า compatible |

### **4. 🔗 Backend Services**

| **ไฟล์เดิม** | **ไฟล์ใหม่** | **เหตุผล** |
|-------------|-------------|------------|
| `backend/services/recommend_optimized.py` | `backend/services/recommendation_engine.py` | ✅ เน้น engine และไม่ใช้ optimized |
| `backend/services/ai_model_service.py` | `backend/services/model_inference.py` | ✅ เน้น inference service |
| `backend/models/real_model_loader.py` | `backend/models/model_loader.py` | ✅ ไม่ต้องใช้ "real" |
| `backend/models/optimized_model_loader.py` | `backend/models/cached_model_loader.py` | ✅ เน้น caching mechanism |

### **5. 🛠️ Tools & Utilities**

| **ไฟล์เดิม** | **ไฟล์ใหม่** | **เหตุผล** |
|-------------|-------------|------------|
| `tools/create_realistic_dataset.py` | `tools/dataset_generator.py` | ✅ เน้นการสร้าง dataset |
| `tools/create_sample_dataset.py` | `tools/sample_data_creator.py` | ✅ เน้นการสร้าง sample |
| `tools/clean_project.py` | `tools/project_cleaner.py` | ✅ เป็น noun มากกว่า verb |

### **6. 📊 Training & Evaluation**

| **ไฟล์เดิม** | **ไฟล์ใหม่** | **เหตุผล** |
|-------------|-------------|------------|
| `ai_models/training/unified_training_system.py` | `ai_models/training/training_pipeline.py` | ✅ เน้น pipeline |
| `ai_models/training/master_training_system.py` | `ai_models/training/training_orchestrator.py` | ✅ เน้น orchestration |
| `ai_models/training/advanced_transfer_learning.py` | `ai_models/training/transfer_learning.py` | ✅ ไม่ต้องใช้ "advanced" |

---

## 📋 **การดำเนินการที่เสนอ**

### **Phase 1: Core Files (ไฟล์หลัก)**
```bash
# Frontend
mv frontend/app_streamlit.py frontend/main_app.py
mv frontend/amulet_unified.py frontend/unified_interface.py

# Backend API
mv backend/api/api.py backend/api/main_api.py
mv backend/api/integrated_api.py backend/api/production_api.py
```

### **Phase 2: AI Models**
```bash
# AI Models
mv ai_models/final_steps_6_and_7.py ai_models/deployment_pipeline.py
mv ai_models/lightweight_ml_system.py ai_models/lightweight_model.py
```

### **Phase 3: Services & Tools**
```bash
# Tools
mv tools/create_realistic_dataset.py tools/dataset_generator.py
mv tools/clean_project.py tools/project_cleaner.py
```

### **Phase 4: Update References**
- อัพเดต imports ในไฟล์ที่เกี่ยวข้อง
- อัพเดต documentation
- อัพเดต launch scripts

---

## 🔍 **ผลประโยชน์ที่คาดหวัง**

### **✅ ข้อดี**
1. **ชื่อไฟล์สั้นลง**: `api.py` → `main_api.py`
2. **เข้าใจง่าย**: `deployment_pipeline.py` แทน `final_steps_6_and_7.py`
3. **มาตรฐานสม่ำเสมอ**: snake_case ทั่วทั้งโปรเจค
4. **จัดกลุ่มได้ดี**: main_, production_, lightweight_, etc.
5. **บำรุงรักษาง่าย**: นักพัฒนาใหม่เข้าใจได้เร็ว

### **⚠️ ความเสี่ยง**
1. **Breaking Changes**: ต้องอัพเดต imports
2. **Documentation**: ต้องอัพเดตเอกสาร
3. **Scripts**: ต้องแก้ไข launch scripts

---

## 📅 **Timeline การดำเนินการ**

| **Phase** | **ระยะเวลา** | **รายละเอียด** |
|-----------|-------------|---------------|
| **Phase 1** | 1 วัน | เปลี่ยนไฟล์หลัก + อัพเดต imports |
| **Phase 2** | 1 วัน | AI Models + Training files |
| **Phase 3** | 1 วัน | Tools + Services |
| **Phase 4** | 1 วัน | Testing + Documentation |

**รวม**: 4 วันทำการ

---

## 🚀 **การอนุมัติและดำเนินการ**

**สถานะ**: ⏳ รอการอนุมัติ  
**ผู้เสนอ**: GitHub Copilot  
**วันที่เสนอ**: 26 กันยายน 2025

**หมายเหตุ**: การเปลี่ยนชื่อไฟล์จะช่วยให้โปรเจคมีมาตรฐานที่ดีขึ้น แต่ต้องระวังการอัพเดต references ให้ครบถ้วน