#!/usr/bin/env python3
"""
🧹 Project Structure Cleanup & Organization Tool
จัดระเบียบโครงสร้างโปรเจกต์และไฟล์ทั้งหมด
"""
import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import glob

class ProjectOrganizer:
    def __init__(self, project_root="E:/Amulet-Ai"):
        self.project_root = Path(project_root)
        self.cleanup_report = {
            "timestamp": datetime.now().isoformat(),
            "actions": [],
            "files_moved": 0,
            "files_deleted": 0,
            "folders_created": 0
        }
        
    def analyze_root_files(self):
        """วิเคราะห์ไฟล์ในโฟลเดอร์ root"""
        print("🔍 วิเคราะห์ไฟล์ในโฟลเดอร์ root...")
        
        root_files = []
        for item in self.project_root.iterdir():
            if item.is_file():
                root_files.append({
                    "name": item.name,
                    "size": item.stat().st_size,
                    "extension": item.suffix,
                    "path": str(item)
                })
        
        # จัดกลุ่มไฟล์ตามประเภท
        file_categories = {
            "scripts": [],      # Python scripts
            "configs": [],      # Configuration files
            "docs": [],         # Documentation
            "tests": [],        # Test files
            "temp": [],         # Temporary files
            "models": [],       # Model files
            "datasets": [],     # Dataset files
            "reports": [],      # Report files
            "keep_root": []     # ไฟล์ที่ควรอยู่ใน root
        }
        
        for file_info in root_files:
            name = file_info["name"].lower()
            ext = file_info["extension"].lower()
            
            # ไฟล์ที่ควรอยู่ใน root
            if name in ["readme.md", "requirements.txt", "makefile", ".gitignore", ".env.example"]:
                file_categories["keep_root"].append(file_info)
            
            # Python scripts
            elif ext == ".py":
                if "test" in name:
                    file_categories["tests"].append(file_info)
                elif name in ["dataset_organizer.py", "phase1_dataset_organizer.py", 
                             "phase2_preprocessing.py", "phase3_model_training.py"]:
                    file_categories["scripts"].append(file_info)
                elif "train" in name or "model" in name:
                    file_categories["scripts"].append(file_info)
                else:
                    file_categories["scripts"].append(file_info)
            
            # Documentation
            elif ext == ".md":
                file_categories["docs"].append(file_info)
            
            # Configuration
            elif ext in [".json", ".yaml", ".yml", ".ini", ".cfg"]:
                file_categories["configs"].append(file_info)
            
            # Temporary files
            elif "temp" in name or "tmp" in name or ext in [".log", ".cache"]:
                file_categories["temp"].append(file_info)
            
            # Reports
            elif "report" in name or "analysis" in name or "result" in name:
                file_categories["reports"].append(file_info)
            
            # Models
            elif ext in [".joblib", ".pkl", ".model"]:
                file_categories["models"].append(file_info)
        
        return file_categories
    
    def create_organized_structure(self):
        """สร้างโครงสร้างโฟลเดอร์ใหม่"""
        print("\n🏗️ สร้างโครงสร้างโฟลเดอร์ใหม่...")
        
        new_folders = [
            "archive/scripts",
            "archive/old_models", 
            "archive/temp_files",
            "documentation/reports",
            "documentation/analysis",
            "configuration",
            "utilities/dataset_tools",
            "utilities/testing",
            "backup/models",
            "backup/configs"
        ]
        
        for folder in new_folders:
            folder_path = self.project_root / folder
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                self.cleanup_report["folders_created"] += 1
                print(f"   ✅ สร้าง: {folder}")
        
        return new_folders
    
    def move_files_to_organized_structure(self, file_categories):
        """ย้ายไฟล์เข้าโครงสร้างใหม่"""
        print("\n📦 ย้ายไฟล์เข้าโครงสร้างใหม่...")
        
        # Mapping การย้ายไฟล์
        move_mapping = {
            "scripts": "utilities/dataset_tools",
            "tests": "utilities/testing", 
            "docs": "documentation/reports",
            "configs": "configuration",
            "temp": "archive/temp_files",
            "reports": "documentation/analysis"
        }
        
        for category, files in file_categories.items():
            if category in move_mapping and files:
                target_folder = self.project_root / move_mapping[category]
                
                print(f"\n   📁 ย้าย {category} ({len(files)} ไฟล์)")
                for file_info in files:
                    source = Path(file_info["path"])
                    target = target_folder / source.name
                    
                    try:
                        if source.exists() and source != target:
                            shutil.move(str(source), str(target))
                            print(f"      ✅ {source.name} → {move_mapping[category]}")
                            self.cleanup_report["files_moved"] += 1
                            self.cleanup_report["actions"].append({
                                "action": "move",
                                "file": source.name,
                                "from": "root",
                                "to": move_mapping[category]
                            })
                    except Exception as e:
                        print(f"      ❌ Error moving {source.name}: {e}")
    
    def handle_duplicate_folders(self):
        """จัดการโฟลเดอร์ที่ซ้ำซ้อน"""
        print("\n🔄 จัดการโฟลเดอร์ที่ซ้ำซ้อน...")
        
        # ตรวจสอบโฟลเดอร์ที่อาจซ้ำซ้อน
        potential_duplicates = [
            ("Data set", "organized_dataset"),
            ("trained_model_backup", "backup/models")
        ]
        
        for old_folder, suggested_location in potential_duplicates:
            old_path = self.project_root / old_folder
            if old_path.exists():
                print(f"   📁 ตรวจพบ: {old_folder}")
                # สร้างโฟลเดอร์ archive สำหรับย้าย
                archive_path = self.project_root / "archive" / old_folder.replace(" ", "_")
                if not archive_path.exists():
                    archive_path.mkdir(parents=True, exist_ok=True)
                
                try:
                    shutil.move(str(old_path), str(archive_path))
                    print(f"      ✅ ย้าย {old_folder} → archive/")
                    self.cleanup_report["actions"].append({
                        "action": "archive",
                        "folder": old_folder,
                        "to": f"archive/{old_folder.replace(' ', '_')}"
                    })
                except Exception as e:
                    print(f"      ❌ Error: {e}")
    
    def clean_redundant_files(self):
        """ลบไฟล์ที่ซ้ำซ้อนหรือไม่จำเป็น"""
        print("\n🗑️ ลบไฟล์ที่ซ้ำซ้อน...")
        
        # ไฟล์ที่อาจซ้ำซ้อน
        redundant_patterns = [
            "*_backup_*.py",
            "*_old.py", 
            "*_temp.py",
            "*.tmp",
            "*.cache",
            "__pycache__"
        ]
        
        for pattern in redundant_patterns:
            files = list(self.project_root.glob(pattern))
            for file_path in files:
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        print(f"   🗑️ ลบ: {file_path.name}")
                        self.cleanup_report["files_deleted"] += 1
                        self.cleanup_report["actions"].append({
                            "action": "delete",
                            "file": file_path.name,
                            "reason": "redundant"
                        })
                    except Exception as e:
                        print(f"   ❌ Error deleting {file_path.name}: {e}")
    
    def create_project_structure_doc(self):
        """สร้างเอกสารโครงสร้างโปรเจกต์"""
        print("\n📋 สร้างเอกสารโครงสร้างโปรเจกต์...")
        
        structure_doc = """# 🏗️ Amulet-AI Project Structure

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
**สร้างเมื่อ:** {timestamp}
**เวอร์ชัน:** 3.0
**สถานะ:** ✅ Organized
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        doc_path = self.project_root / "documentation" / "PROJECT_STRUCTURE.md"
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(structure_doc)
        
        print(f"   ✅ สร้างเอกสาร: {doc_path}")
    
    def save_cleanup_report(self):
        """บันทึกรายงานการจัดระเบียบ"""
        report_path = self.project_root / "documentation" / "analysis" / "cleanup_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.cleanup_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 บันทึกรายงาน: {report_path}")
    
    def run_full_cleanup(self):
        """รันการจัดระเบียบทั้งหมด"""
        print("🧹 เริ่มการจัดระเบียบโปรเจกต์...")
        print("=" * 60)
        
        # Step 1: วิเคราะห์ไฟล์ root
        file_categories = self.analyze_root_files()
        
        # Step 2: สร้างโครงสร้างใหม่
        self.create_organized_structure()
        
        # Step 3: ย้ายไฟล์
        self.move_files_to_organized_structure(file_categories)
        
        # Step 4: จัดการโฟลเดอร์ซ้ำซ้อน
        self.handle_duplicate_folders()
        
        # Step 5: ลบไฟล์ที่ไม่จำเป็น
        self.clean_redundant_files()
        
        # Step 6: สร้างเอกสาร
        self.create_project_structure_doc()
        
        # Step 7: บันทึกรายงาน
        self.save_cleanup_report()
        
        # สรุปผล
        print("\n" + "=" * 60)
        print("✅ การจัดระเบียบเสร็จสิ้น!")
        print(f"📁 โฟลเดอร์ใหม่: {self.cleanup_report['folders_created']}")
        print(f"📦 ไฟล์ย้าย: {self.cleanup_report['files_moved']}")
        print(f"🗑️ ไฟล์ลบ: {self.cleanup_report['files_deleted']}")
        print(f"📋 การกระทำทั้งหมด: {len(self.cleanup_report['actions'])}")

def main():
    organizer = ProjectOrganizer()
    organizer.run_full_cleanup()

if __name__ == "__main__":
    main()