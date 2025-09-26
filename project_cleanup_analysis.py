#!/usr/bin/env python3
"""
🔍 Project Cleanup & Analysis Tool
ตรวจสอบและจัดระบบโปรเจค Amulet-AI ให้เหลือเฉพาะไฟล์หลักที่จำเป็น
และระบุจุดอ่อน ข้อผิดพลาด และสิ่งที่ไม่เหมาะสม
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectAnalyzer:
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "duplicates": [],
            "secondary_systems": [],
            "unnecessary_files": [],
            "missing_connections": [],
            "project_weaknesses": [],
            "recommendations": []
        }
    
    def analyze_project(self):
        """วิเคราะห์โปรเจคทั้งหมด"""
        logger.info("🔍 เริ่มการวิเคราะห์โปรเจค Amulet-AI")
        
        # 1. หาไฟล์ที่ซ้ำกัน
        self._find_duplicate_systems()
        
        # 2. หาระบบรอง/ตัวเลือก
        self._find_secondary_systems()
        
        # 3. หาไฟล์ที่ไม่จำเป็น
        self._find_unnecessary_files()
        
        # 4. วิเคราะห์การเชื่อมต่อระหว่างไฟล์
        self._analyze_file_connections()
        
        # 5. ระบุจุดอ่อนของโปรเจค
        self._identify_project_weaknesses()
        
        # 6. สร้างคำแนะนำ
        self._generate_recommendations()
        
        return self.analysis_results
    
    def _find_duplicate_systems(self):
        """หาระบบที่ซ้ำกัน"""
        logger.info("📋 ตรวจสอบระบบที่ซ้ำกัน...")
        
        # ไฟล์ AI Models ที่ซ้ำกัน
        ai_models = list(self.root.glob("ai_models/*.py"))
        if len(ai_models) > 2:  # มากกว่า __init__.py และหลักหนึ่งตัว
            for model in ai_models:
                if model.name != "__init__.py":
                    self.analysis_results["duplicates"].append({
                        "type": "AI Model",
                        "file": str(model),
                        "reason": "มีหลายไฟล์ AI model - ควรเหลือเฉพาะตัวหลัก"
                    })
        
        # ไฟล์ trained models ที่ซ้ำกัน
        trained_dirs = [d for d in self.root.iterdir() if d.is_dir() and "trained_model" in d.name]
        if len(trained_dirs) > 1:
            for trained_dir in trained_dirs:
                self.analysis_results["duplicates"].append({
                    "type": "Trained Model",
                    "file": str(trained_dir),
                    "reason": "มีหลาย trained model directories - ควรเหลือเฉพาะ current version"
                })
        
        # API ซ้ำกัน
        api_files = list(self.root.glob("backend/api/*.py"))
        api_systems = [f for f in api_files if "api" in f.name]
        if len(api_systems) > 1:
            for api in api_systems:
                self.analysis_results["duplicates"].append({
                    "type": "API System",
                    "file": str(api),
                    "reason": "มีหลาย API implementations"
                })
    
    def _find_secondary_systems(self):
        """หาระบบรองและระบบตัวเลือก"""
        logger.info("🔍 ตรวจสอบระบบรองและตัวเลือก...")
        
        secondary_patterns = [
            ("lean", "ระบบ Lean - ไม่จำเป็นถ้ามี Production system"),
            ("v3", "Version 3 - เก่าแล้วถ้ามี v4"),
            ("production_ready", "Production Ready - อาจซ้ำกับ Enhanced Production"),
            ("secondary", "ระบบรอง"),
            ("alternative", "ระบบทางเลือก"),
            ("backup", "สำรองข้อมูล"),
            ("temp", "ไฟล์ชั่วคราว"),
            ("test", "ไฟล์ทดสอบ (บางส่วน)"),
            ("old", "ไฟล์เก่า")
        ]
        
        for pattern, reason in secondary_patterns:
            matches = list(self.root.rglob(f"*{pattern}*"))
            for match in matches:
                if match.is_file() and not match.name.startswith('.'):
                    self.analysis_results["secondary_systems"].append({
                        "file": str(match),
                        "pattern": pattern,
                        "reason": reason
                    })
    
    def _find_unnecessary_files(self):
        """หาไฟล์ที่ไม่จำเป็น"""
        logger.info("🗑️ ตรวจสอบไฟล์ที่ไม่จำเป็น...")
        
        unnecessary_files = [
            "miniconda.exe",
            "*.pyc",
            "__pycache__",
            "*.log",
            ".DS_Store",
            "Thumbs.db",
            "desktop.ini",
            "*.tmp",
            "*.temp"
        ]
        
        doc_files = [
            "PHASE3_FINAL_REPORT.md",
            "ENHANCED_SYSTEM_FINAL_REPORT.md", 
            "PERSONA_TECHNICAL_SOLUTIONS.md",
            "PERSONA_SOLUTIONS_QUICK_REFERENCE.md",
            "persona_solutions_summary.json"
        ]
        
        # หาไฟล์ที่ไม่จำเป็น
        for pattern in unnecessary_files:
            matches = list(self.root.rglob(pattern))
            for match in matches:
                self.analysis_results["unnecessary_files"].append({
                    "file": str(match),
                    "type": "System file",
                    "reason": "ไฟล์ระบบหรือไฟล์ขยะ"
                })
        
        # เอกสารที่มากเกิน
        for doc in doc_files:
            doc_path = self.root / doc
            if doc_path.exists():
                self.analysis_results["unnecessary_files"].append({
                    "file": str(doc_path),
                    "type": "Documentation",
                    "reason": "เอกสารมากเกินไป - ควรรวมเป็น README เดียว"
                })
    
    def _analyze_file_connections(self):
        """วิเคราะห์การเชื่อมต่อระหว่างไฟล์"""
        logger.info("🔗 วิเคราะห์การเชื่อมต่อไฟล์...")
        
        # ตรวจสอบ import ที่เสียหาย
        python_files = list(self.root.rglob("*.py"))
        
        for py_file in python_files:
            if py_file.name.startswith('.'):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # หา import ที่อาจเสียหาย
                if "optimized_model" in content:
                    if not (self.root / "ai_models" / "optimized_model.py").exists():
                        self.analysis_results["missing_connections"].append({
                            "file": str(py_file),
                            "issue": "Import optimized_model แต่ไฟล์ไม่พบ",
                            "severity": "High"
                        })
                
                if "lean_model" in content:
                    if not (self.root / "ai_models" / "lean_model.py").exists():
                        self.analysis_results["missing_connections"].append({
                            "file": str(py_file),
                            "issue": "Import lean_model แต่ไฟล์ไม่พบ",
                            "severity": "Medium"
                        })
                        
            except Exception as e:
                logger.warning(f"ไม่สามารถอ่านไฟล์ {py_file}: {e}")
    
    def _identify_project_weaknesses(self):
        """ระบุจุดอ่อนของโปรเจค"""
        logger.info("⚠️ ระบุจุดอ่อนของโปรเจค...")
        
        weaknesses = []
        
        # 1. ความซับซ้อนเกินจำเป็น
        weaknesses.append({
            "category": "Architecture",
            "issue": "มีระบบหลายเวอร์ชันทำงานคู่กัน",
            "description": "มี v3, v4, enhanced, production ทำให้สับสน",
            "severity": "High",
            "impact": "ผู้พัฒนาสับสน, maintenance ยาก"
        })
        
        # 2. ข้อมูลมีน้อย
        dataset_path = self.root / "dataset_optimized"
        if dataset_path.exists():
            train_dirs = list((dataset_path / "train").glob("*"))
            if train_dirs:
                sample_count = len(list(train_dirs[0].glob("*.jpg")))
                if sample_count < 50:
                    weaknesses.append({
                        "category": "Data",
                        "issue": "ข้อมูลฝึกมีน้อย",
                        "description": f"มีเพียง {sample_count} ภาพต่อคลาส - น้อยเกินไปสำหรับ production",
                        "severity": "Critical",
                        "impact": "โมเดลอาจ overfit, ไม่ generalize ได้ดี"
                    })
        
        # 3. หลุดจุดประสงค์เดิม
        weaknesses.append({
            "category": "Project Focus",
            "issue": "หลุดจากจุดประสงค์เดิม",
            "description": "เดิมเป็นระบบจำแนกพระเครื่อง แต่กลายเป็น tech showcase ที่ซับซ้อน",
            "severity": "High",
            "impact": "ผู้ใช้งานจริงใช้งานยาก, ไม่ตรงความต้องการ"
        })
        
        # 4. Over-engineering
        weaknesses.append({
            "category": "Engineering",
            "issue": "Over-engineering",
            "description": "มี features มากเกินความจำเป็น เช่น multiple personas, complex monitoring",
            "severity": "Medium", 
            "impact": "ระบบซับซ้อน, พัฒนาและ maintain ยาก"
        })
        
        # 5. ไฟล์เอกสารมากเกิน
        doc_count = len(list(self.root.glob("*.md")))
        if doc_count > 5:
            weaknesses.append({
                "category": "Documentation",
                "issue": "เอกสารมากเกินไป",
                "description": f"มี {doc_count} ไฟล์เอกสาร - มากเกินไป",
                "severity": "Low",
                "impact": "ผู้ใช้สับสน, ไม่รู้ว่าควรอ่านอะไร"
            })
        
        # 6. Dataset ซ้ำซ้อน
        dataset_dirs = [d for d in self.root.iterdir() if d.is_dir() and "dataset" in d.name]
        if len(dataset_dirs) > 1:
            weaknesses.append({
                "category": "Data Management",
                "issue": "Dataset ซ้ำซ้อน",
                "description": f"มี {len(dataset_dirs)} datasets - สิ้นเปลืองพื้นที่",
                "severity": "Medium",
                "impact": "ใช้พื้นที่ดิสก์มาก, สับสน"
            })
        
        self.analysis_results["project_weaknesses"] = weaknesses
    
    def _generate_recommendations(self):
        """สร้างคำแนะนำ"""
        logger.info("💡 สร้างคำแนะนำ...")
        
        recommendations = [
            {
                "priority": "Critical",
                "action": "รวมระบบให้เหลือเพียงตัวเดียว",
                "description": "เลือก enhanced_production_system.py เป็นหลัก ลบที่เหลือ",
                "benefit": "ลดความสับสน, ง่ายต่อการ maintain"
            },
            {
                "priority": "High", 
                "action": "เพิ่มข้อมูลฝึก",
                "description": "ต้องมีอย่างน้อย 100-500 ภาพต่อคลาสสำหรับ production",
                "benefit": "โมเดลแม่นยำและเสถียรขึ้น"
            },
            {
                "priority": "High",
                "action": "ลดความซับซ้อนของ UI",
                "description": "ทำ UI ง่ายๆ เพื่อผู้ใช้ทั่วไป ไม่ใช่ tech experts",
                "benefit": "ใช้งานง่าย, ตรงจุดประสงค์"
            },
            {
                "priority": "Medium",
                "action": "รวมเอกสารเป็น README เดียว",
                "description": "ลบเอกสารส่วนเกิน เหลือแค่ README.md ที่สมบูรณ์",
                "benefit": "ผู้ใช้หาข้อมูลง่าย"
            },
            {
                "priority": "Medium",
                "action": "ลบ trained models เก่า",
                "description": "เหลือแค่ model ที่ใช้งานจริง",
                "benefit": "ประหยัดพื้นที่, ลดความสับสน"
            },
            {
                "priority": "Low",
                "action": "จัดระบบ dependencies",
                "description": "อัพเดท requirements.txt ให้ตรงกับที่ใช้งานจริง",
                "benefit": "ติดตั้งง่าย, ไม่มี dependency conflicts"
            }
        ]
        
        self.analysis_results["recommendations"] = recommendations
    
    def generate_cleanup_script(self):
        """สร้างสคริปต์สำหรับจัดระบบ"""
        
        cleanup_actions = []
        
        # ลบไฟล์ซ้ำ
        for dup in self.analysis_results["duplicates"]:
            cleanup_actions.append(f"# ลบ duplicate: {dup['file']}")
            cleanup_actions.append(f"rm -rf '{dup['file']}'")
        
        # ลบระบบรอง
        for sec in self.analysis_results["secondary_systems"]:
            cleanup_actions.append(f"# ลบ secondary system: {sec['file']}")
            cleanup_actions.append(f"rm -rf '{sec['file']}'")
        
        # ลบไฟล์ไม่จำเป็น
        for unnecessary in self.analysis_results["unnecessary_files"]:
            cleanup_actions.append(f"# ลบไฟล์ไม่จำเป็น: {unnecessary['file']}")
            cleanup_actions.append(f"rm -rf '{unnecessary['file']}'")
        
        return "\n".join(cleanup_actions)
    
    def save_analysis(self, output_path: str = "project_analysis_report.json"):
        """บันทึกผลการวิเคราะห์"""
        
        # เพิ่มสรุปสถิติ
        self.analysis_results["summary"] = {
            "total_duplicates": len(self.analysis_results["duplicates"]),
            "total_secondary_systems": len(self.analysis_results["secondary_systems"]),
            "total_unnecessary_files": len(self.analysis_results["unnecessary_files"]),
            "total_weaknesses": len(self.analysis_results["project_weaknesses"]),
            "critical_issues": len([w for w in self.analysis_results["project_weaknesses"] if w["severity"] == "Critical"]),
            "cleanup_script": self.generate_cleanup_script()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 บันทึกรายงานการวิเคราะห์ที่: {output_path}")

def main():
    """เริ่มการวิเคราะห์"""
    
    current_dir = Path(__file__).parent
    analyzer = ProjectAnalyzer(current_dir)
    
    # วิเคราะห์โปรเจค
    results = analyzer.analyze_project()
    
    # บันทึกรายงาน
    analyzer.save_analysis("project_analysis_report.json")
    
    # แสดงสรุป
    print("\n" + "="*60)
    print("🔍 PROJECT ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"📊 พบปัญหาทั้งหมด:")
    print(f"   • ไฟล์ซ้ำ: {len(results['duplicates'])} ไฟล์")
    print(f"   • ระบบรอง: {len(results['secondary_systems'])} ไฟล์")
    print(f"   • ไฟล์ไม่จำเป็น: {len(results['unnecessary_files'])} ไฟล์")
    print(f"   • จุดอ่อนโปรเจค: {len(results['project_weaknesses'])} จุด")
    
    critical_issues = [w for w in results['project_weaknesses'] if w['severity'] == 'Critical']
    if critical_issues:
        print(f"\n⚠️ ปัญหาวิกฤติ ({len(critical_issues)} จุด):")
        for issue in critical_issues:
            print(f"   • {issue['issue']}: {issue['description']}")
    
    print(f"\n💡 คำแนะนำสำคัญ:")
    for rec in results['recommendations'][:3]:
        print(f"   • [{rec['priority']}] {rec['action']}")
    
    print(f"\n📄 รายงานฉบับเต็ม: project_analysis_report.json")
    print("="*60)

if __name__ == "__main__":
    main()