# 📁 Amulet-AI Organized Project Structure

## 🏗️ โครงสร้างโฟลเดอร์ที่จัดระเบียบแล้ว

```
Amulet-Ai/
├── 📚 docs/                        # Documentation
│   ├── README.md                   # คู่มือหลัก
│   ├── ARCHITECTURE_WORKFLOW.md    # สถาปัตยกรรมระบบ
│   ├── PHASE2_COMPLETION.md        # สรุป Phase 2
│   ├── PHASE2_IMPROVEMENTS.md      # การปรับปรุง Phase 2
│   └── QUICK_START.md             # คู่มือเริ่มต้นใช้งาน
│
├── ⚙️ config/                      # Configuration Files
│   ├── .env.example               # Environment template
│   └── config_template.env        # Configuration template
│
├── 🔧 core/                        # Core System Modules
│   ├── __init__.py                # Core module exports
│   ├── config.py                  # Configuration management
│   ├── error_handling.py          # Error handling & retry mechanisms
│   ├── memory_management.py       # Memory optimization & streaming
│   ├── performance.py             # Caching & performance optimization
│   ├── thread_safety.py           # Thread-safe operations
│   ├── security.py                # Security utilities
│   └── rate_limiter.py            # Rate limiting
│
├── 🚀 scripts/                     # Utility Scripts
│   ├── production_runner.py       # Production deployment manager
│   └── usage_examples.py          # Feature usage examples
│
├── 🧪 tests/                       # Testing Framework
│   └── test_enhanced_features.py  # Comprehensive feature tests
│
├── 🤖 ai_models/                   # AI Model Components
│   ├── __init__.py
│   ├── compatibility_loader.py
│   ├── enhanced_production_system.py
│   ├── labels.json
│   └── twobranch/                 # CNN Architecture
│       ├── config.py
│       ├── dataset.py
│       ├── enhanced_integration.py
│       ├── enhanced_multilayer_cnn.py
│       ├── enhanced_preprocessing.py
│       ├── enhanced_training.py
│       ├── inference.py
│       ├── model.py
│       ├── preprocess.py
│       └── realistic_amulet_generator.py
│
├── ⚡ api/                          # Backend API
│   ├── __init__.py
│   └── main_api.py                # FastAPI backend server
│
├── 🎨 frontend/                    # Frontend Interface
│   ├── main_streamlit_app.py      # Streamlit web interface
│   └── run_frontend.py            # (deprecated placeholder)
│
├── 🏛️ trained_model/               # Trained Models
│   ├── classifier.joblib
│   ├── deployment_info.json
│   ├── label_encoder.joblib
│   ├── model_info.json
│   ├── ood_detector.joblib
│   ├── pca.joblib
│   └── scaler.joblib
│
├── requirements.txt               # Python dependencies
└── .gitignore                    # Git ignore rules
```

## 🎯 การใช้งานหลัก

### 🚀 รันระบบ Production

```bash
# API Server
python scripts/production_runner.py api

# Frontend
python scripts/production_runner.py frontend

# ทดสอบระบบ
python scripts/production_runner.py test
```

### 🧪 Demo และทดสอบ

```bash
# Demo การใช้งาน Enhanced Features
python scripts/usage_examples.py

# ทดสอบ Memory Management และ Thread Safety
python tests/test_enhanced_features.py
```

### 📦 Import โมดูล

```python
# Import ทั้งหมดจาก core
from core import config, memory_monitor, image_cache, thread_safe_operation

# หรือ import แต่ละโมดูล
from core.memory_management import memory_monitor
from core.performance import image_cache
from core.thread_safety import ThreadSafeDict
from core.error_handling import retry_on_failure
```

## 🔄 การย้ายไฟล์

### ✅ ไฟล์ที่ย้ายแล้ว

| ไฟล์เดิม | ตำแหน่งใหม่ | ประเภท |
|---------|-------------|--------|
| `config.py` | `core/config.py` | Core Module |
| `error_handling.py` | `core/error_handling.py` | Core Module |
| `memory_management.py` | `core/memory_management.py` | Core Module |
| `performance.py` | `core/performance.py` | Core Module |
| `thread_safety.py` | `core/thread_safety.py` | Core Module |
| `security.py` | `core/security.py` | Core Module |
| `rate_limiter.py` | `core/rate_limiter.py` | Core Module |
| `*.md` | `docs/` | Documentation |
| `*.env` | `config/` | Configuration |
| `production_runner.py` | `scripts/production_runner.py` | Script |
| `usage_examples.py` | `scripts/usage_examples.py` | Script |
| `test_enhanced_features.py` | `tests/test_enhanced_features.py` | Test |

### 🔧 การอัปเดต Import

การ import ทั้งหมดได้รับการอัปเดตแล้ว:

- ✅ `api/main_api.py` - อัปเดต imports เป็น `core.*`
- ✅ `frontend/main_streamlit_app.py` - อัปเดต imports เป็น `core.*`
- ✅ `scripts/production_runner.py` - อัปเดต imports เป็น `core.*`
- ✅ `scripts/usage_examples.py` - อัปเดต imports เป็น `core.*`
- ✅ `tests/test_enhanced_features.py` - อัปเดต imports เป็น `core.*`
- ✅ ไฟล์ใน `core/` - ใช้ relative imports (`from .config import`)

## 🎉 ข้อดีของโครงสร้างใหม่

### 📦 การจัดระเบียบ
- **โฟลเดอร์แยกตามหน้าที่**: แต่ละโฟลเดอร์มีหน้าที่ชัดเจน
- **ง่ายต่อการค้นหา**: ไฟล์จัดเรียงตามประเภท
- **โมดูลาร์**: แต่ละส่วนแยกจากกันชัดเจน

### 🔧 การพัฒนา
- **Import ง่ายขึ้น**: `from core import ...`
- **Testing แยกออกมา**: โฟลเดอร์ tests เฉพาะ
- **Scripts รวมกัน**: เครื่องมือช่วยเหลือในที่เดียว

### 🚀 Production Ready
- **Configuration แยกออก**: ไฟล์ config ในโฟลเดอร์เฉพาะ
- **Documentation ครบถ้วน**: คู่มือทั้งหมดในโฟลเดอร์ docs
- **Scripts Production**: เครื่องมือ deployment ใน scripts

## 🔄 Backward Compatibility

ระบบยังคงทำงานได้เหมือนเดิม เพียงแต่การ import จะต้องใช้:

```python
# เดิม
from config import config

# ใหม่
from core.config import config
```

ไฟล์ `core/__init__.py` จะทำให้สามารถ import ได้ง่าย:

```python
# Simple import
from core import config, memory_monitor, image_cache
```

## 📋 Checklist สำหรับการใช้งาน

- ✅ อัปเดต imports ในไฟล์ที่ใช้งาน
- ✅ ใช้ `scripts/production_runner.py` สำหรับ production
- ✅ Documentation อยู่ใน `docs/`
- ✅ Configuration templates อยู่ใน `config/`
- ✅ ทดสอบด้วย `tests/test_enhanced_features.py`

---

**🎊 โครงสร้างใหม่พร้อมใช้งาน! ระบบยังคงทำงานได้เหมือนเดิมแต่จัดระเบียบดีขึ้น**