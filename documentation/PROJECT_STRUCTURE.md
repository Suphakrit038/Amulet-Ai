# 🏗️ Amulet-AI Project Structure

## 📁 โครงสร้างโปรเจกต์ที่จัดระเบียบแล้ว

```
Amulet-AI/
├── 📱 ai_models/              # AI Models & Machine Learning
│   ├── compatibility_loader.py
│   ├── enhanced_production_system.py
│   └── labels.json
│
├── 🌐 api/                    # API Backend Services
│   ├── main_api.py           # Main API
│   ├── main_api_fast.py      # FastAPI version
│   └── __init__.py
│
├── 🖥️ frontend/              # User Interface
│   ├── main_streamlit_app.py # Main UI
│   ├── components/           # UI Components
│   └── utils/               # UI Utilities
│
├── ⚙️ core/                  # Core System Components
│   ├── config.py            # Configuration
│   ├── security.py          # Security features
│   └── performance.py       # Performance monitoring
│
├── 📊 organized_dataset/      # Organized Training Data
│   ├── raw/                 # Raw images
│   ├── processed/           # Processed images
│   ├── augmented/           # Augmented data
│   ├── splits/              # Train/Val/Test splits
│   └── metadata/            # Dataset metadata
│
├── 🤖 trained_model/         # Active Model Files
│   ├── classifier.joblib    # Trained classifier
│   ├── scaler.joblib        # Feature scaler
│   ├── label_encoder.joblib # Label encoder
│   └── model_info.json      # Model metadata
│
├── 🧪 tests/                 # Testing Suite
│   ├── comprehensive_test_suite.py
│   └── system_analyzer.py
│
├── 📚 documentation/         # Project Documentation
│   ├── reports/             # Analysis reports
│   └── analysis/            # Technical analysis
│
├── 🔧 utilities/             # Utility Scripts
│   ├── dataset_tools/       # Dataset management
│   └── testing/             # Testing utilities
│
├── 🚀 deployment/            # Deployment Configuration
│   ├── docker-compose.*.yml
│   └── deployment configs
│
├── 📋 configuration/         # Configuration Files
│   └── config files
│
├── 📦 archive/               # Archived Files
│   ├── scripts/             # Old scripts
│   ├── temp_files/          # Temporary files
│   └── old_models/          # Backup models
│
└── 📄 ROOT FILES             # Essential Project Files
    ├── README.md            # Project documentation
    ├── requirements.txt     # Dependencies
    ├── Makefile            # Build automation
    └── .gitignore          # Git ignore rules
```

## 🎯 หน้าที่ของแต่ละโฟลเดอร์

### 🤖 AI Models
- **หน้าที่:** จัดการโมเดล AI และการประมวลผล ML
- **เทคโนโลยี:** scikit-learn, OpenCV, NumPy
- **ไฟล์สำคัญ:** โมเดลที่เทรนแล้ว, label mappings

### 🌐 API Backend
- **หน้าที่:** จัดการ REST API และ business logic
- **เทคโนโลยี:** FastAPI, Uvicorn
- **ฟีเจอร์:** Image upload, prediction, health checks

### 🖥️ Frontend
- **หน้าที่:** User Interface และ User Experience
- **เทคโนโลยี:** Streamlit, HTML/CSS
- **ฟีเจอร์:** File upload, result display, interactive UI

### ⚙️ Core System
- **หน้าที่:** Core utilities และ system management
- **เทคโนโลยี:** Python utilities
- **ฟีเจอร์:** Configuration, security, performance monitoring

### 📊 Dataset Management
- **หน้าที่:** จัดการข้อมูลเทรนและทดสอบ
- **เทคโนโลยี:** OpenCV, PIL
- **ฟีเจอร์:** Data preprocessing, augmentation, organization

## 🔄 Workflow การทำงาน

1. **Data Flow:** Raw Images → Processing → Augmentation → Training
2. **Model Flow:** Training → Validation → Testing → Deployment
3. **API Flow:** Request → Processing → Prediction → Response
4. **User Flow:** Upload → Display → Results → Export

## 🛠️ เทคโนโลジีสแตก

- **Backend:** Python 3.13, FastAPI, scikit-learn
- **Frontend:** Streamlit, HTML/CSS
- **AI/ML:** Random Forest, OpenCV, NumPy
- **Data:** JSON, Joblib, File System
- **DevOps:** Docker (planned), Git

---
**สร้างเมื่อ:** 2025-10-01 18:55:11
**เวอร์ชัน:** 3.0
**สถานะ:** ✅ Organized
