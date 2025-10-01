#!/usr/bin/env python3
"""
🔬 Comprehensive System Analyzer
วิเคราะห์ระบบทุกส่วนแบบครอบคลุม หาจุดอ่อนและข้อบกพร่อง
"""
import os
import sys
import time
import json
import requests
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
import traceback
import psutil

class SystemAnalyzer:
    def __init__(self):
        self.project_root = Path("E:/Amulet-Ai")
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "issues": [],
            "recommendations": [],
            "performance_metrics": {},
            "security_assessment": {}
        }
        
    def analyze_ai_models(self):
        """วิเคราะห์ AI Models และ ML Components"""
        print("🤖 วิเคราะห์ AI Models...")
        
        model_analysis = {
            "status": "unknown",
            "issues": [],
            "performance": {},
            "recommendations": []
        }
        
        try:
            # ตรวจสอบไฟล์โมเดล
            model_files = {
                "classifier": self.project_root / "trained_model" / "classifier.joblib",
                "scaler": self.project_root / "trained_model" / "scaler.joblib", 
                "label_encoder": self.project_root / "trained_model" / "label_encoder.joblib",
                "model_info": self.project_root / "trained_model" / "model_info.json"
            }
            
            missing_files = []
            for name, path in model_files.items():
                if not path.exists():
                    missing_files.append(name)
                    model_analysis["issues"].append(f"Missing {name} file: {path}")
            
            if missing_files:
                model_analysis["status"] = "incomplete"
                model_analysis["recommendations"].append("Retrain model to generate missing components")
            else:
                model_analysis["status"] = "available"
            
            # ทดสอบการโหลดโมเดล
            try:
                import joblib
                classifier = joblib.load(model_files["classifier"])
                scaler = joblib.load(model_files["scaler"])
                
                model_analysis["performance"]["model_type"] = type(classifier).__name__
                model_analysis["performance"]["n_estimators"] = getattr(classifier, 'n_estimators', 'N/A')
                model_analysis["performance"]["n_features"] = getattr(scaler, 'n_features_in_', 'N/A')
                
                # ทดสอบ memory usage
                import numpy as np
                test_data = np.random.random((1, model_analysis["performance"]["n_features"]))
                
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                scaled_data = scaler.transform(test_data)
                prediction = classifier.predict(scaled_data)
                probabilities = classifier.predict_proba(scaled_data)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                model_analysis["performance"]["prediction_time"] = end_time - start_time
                model_analysis["performance"]["memory_usage_mb"] = (end_memory - start_memory) / 1024 / 1024
                
            except Exception as e:
                model_analysis["issues"].append(f"Model loading error: {str(e)}")
                model_analysis["status"] = "error"
            
            # ตรวจสอบ AI models code
            ai_models_dir = self.project_root / "ai_models"
            py_files = list(ai_models_dir.glob("*.py"))
            
            if not py_files:
                model_analysis["issues"].append("No Python files in ai_models directory")
            
            # ตรวจสอบ labels.json
            labels_file = ai_models_dir / "labels.json"
            if labels_file.exists():
                try:
                    with open(labels_file, 'r', encoding='utf-8') as f:
                        labels = json.load(f)
                    model_analysis["performance"]["n_classes"] = len(labels.get("current_classes", {}))
                except Exception as e:
                    model_analysis["issues"].append(f"Labels file error: {str(e)}")
            else:
                model_analysis["issues"].append("Labels file missing")
            
        except Exception as e:
            model_analysis["status"] = "error"
            model_analysis["issues"].append(f"Critical error: {str(e)}")
        
        self.analysis_results["components"]["ai_models"] = model_analysis
        return model_analysis
    
    def analyze_api_systems(self):
        """วิเคราะห์ API Systems"""
        print("🌐 วิเคราะห์ API Systems...")
        
        api_analysis = {
            "endpoints": {},
            "performance": {},
            "security": {},
            "issues": [],
            "recommendations": []
        }
        
        api_files = [
            "api/main_api.py",
            "api/main_api_fast.py"
        ]
        
        for api_file in api_files:
            api_path = self.project_root / api_file
            file_analysis = {
                "exists": api_path.exists(),
                "size_kb": api_path.stat().st_size / 1024 if api_path.exists() else 0,
                "issues": [],
                "features": []
            }
            
            if api_path.exists():
                try:
                    with open(api_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # ตรวจสอบฟีเจอร์
                    if "FastAPI" in content:
                        file_analysis["features"].append("FastAPI framework")
                    if "cors" in content.lower():
                        file_analysis["features"].append("CORS enabled")
                    if "security" in content.lower():
                        file_analysis["features"].append("Security measures")
                    if "rate" in content.lower():
                        file_analysis["features"].append("Rate limiting")
                    
                    # ตรวจสอบจุดอ่อน
                    if "debug=True" in content:
                        file_analysis["issues"].append("Debug mode enabled in production")
                    if "host=\"0.0.0.0\"" in content:
                        file_analysis["issues"].append("Listening on all interfaces - security risk")
                    if "allow_origins=[\"*\"]" in content:
                        file_analysis["issues"].append("CORS allows all origins - security risk")
                    
                except Exception as e:
                    file_analysis["issues"].append(f"Cannot read file: {str(e)}")
            else:
                file_analysis["issues"].append("File does not exist")
            
            api_analysis["endpoints"][api_file] = file_analysis
        
        # ทดสอบ API เมื่อเซิร์ฟเวอร์ทำงาน
        test_urls = [
            "http://localhost:8000/health",
            "http://localhost:8000/docs",
            "http://localhost:8000/"
        ]
        
        api_analysis["connectivity"] = {}
        for url in test_urls:
            try:
                response = requests.get(url, timeout=5)
                api_analysis["connectivity"][url] = {
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                    "accessible": True
                }
            except requests.exceptions.ConnectionError:
                api_analysis["connectivity"][url] = {
                    "accessible": False,
                    "error": "Connection refused - server not running"
                }
            except Exception as e:
                api_analysis["connectivity"][url] = {
                    "accessible": False,
                    "error": str(e)
                }
        
        # Security assessment
        security_issues = []
        for file_info in api_analysis["endpoints"].values():
            security_issues.extend(file_info["issues"])
        
        if security_issues:
            api_analysis["security"]["risk_level"] = "high"
            api_analysis["security"]["issues"] = security_issues
        else:
            api_analysis["security"]["risk_level"] = "medium"
        
        self.analysis_results["components"]["api_systems"] = api_analysis
        return api_analysis
    
    def analyze_frontend(self):
        """วิเคราะห์ Frontend Systems"""
        print("🖥️ วิเคราะห์ Frontend...")
        
        frontend_analysis = {
            "apps": {},
            "components": {},
            "performance": {},
            "usability": {},
            "issues": [],
            "recommendations": []
        }
        
        frontend_dir = self.project_root / "frontend"
        
        # ตรวจสอบไฟล์ Streamlit apps
        streamlit_files = list(frontend_dir.glob("main_streamlit_app*.py"))
        
        for app_file in streamlit_files:
            app_analysis = {
                "size_kb": app_file.stat().st_size / 1024,
                "features": [],
                "issues": [],
                "complexity": "unknown"
            }
            
            try:
                with open(app_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                app_analysis["lines_of_code"] = len(lines)
                
                # ตรวจสอบฟีเจอร์
                if "file_uploader" in content:
                    app_analysis["features"].append("File upload")
                if "image" in content.lower():
                    app_analysis["features"].append("Image display")
                if "plotly" in content or "matplotlib" in content:
                    app_analysis["features"].append("Data visualization")
                if "session_state" in content:
                    app_analysis["features"].append("Session management")
                
                # ตรวจสอบประสิทธิภาพ
                if content.count("st.") > 50:
                    app_analysis["complexity"] = "high"
                    app_analysis["issues"].append("High number of Streamlit components - may affect performance")
                elif content.count("st.") > 20:
                    app_analysis["complexity"] = "medium"
                else:
                    app_analysis["complexity"] = "low"
                
                # ตรวจสอบ error handling
                if "try:" not in content:
                    app_analysis["issues"].append("No error handling detected")
                if "except:" in content and "Exception" not in content:
                    app_analysis["issues"].append("Generic exception handling - should be more specific")
                
            except Exception as e:
                app_analysis["issues"].append(f"Cannot analyze file: {str(e)}")
            
            frontend_analysis["apps"][app_file.name] = app_analysis
        
        # ตรวจสอบ components
        components_dir = frontend_dir / "components"
        if components_dir.exists():
            component_files = list(components_dir.glob("*.py"))
            frontend_analysis["components"]["count"] = len(component_files)
            frontend_analysis["components"]["modular"] = len(component_files) > 0
        else:
            frontend_analysis["issues"].append("No components directory - code may not be modular")
        
        # ตรวจสอบ static files
        static_files = list(frontend_dir.glob("*.css")) + list(frontend_dir.glob("*.js"))
        if static_files:
            frontend_analysis["features"] = frontend_analysis.get("features", [])
            frontend_analysis["features"].append("Custom styling")
        
        self.analysis_results["components"]["frontend"] = frontend_analysis
        return frontend_analysis
    
    def analyze_data_management(self):
        """วิเคราะห์ Data Management"""
        print("📊 วิเคราะห์ Data Management...")
        
        data_analysis = {
            "datasets": {},
            "organization": {},
            "storage": {},
            "issues": [],
            "recommendations": []
        }
        
        # ตรวจสอบ organized_dataset
        dataset_dir = self.project_root / "organized_dataset"
        if dataset_dir.exists():
            subdirs = ["raw", "processed", "augmented", "splits", "metadata"]
            data_analysis["organization"]["structured"] = True
            
            for subdir in subdirs:
                subdir_path = dataset_dir / subdir
                if subdir_path.exists():
                    file_count = len(list(subdir_path.rglob("*.*")))
                    folder_count = len([d for d in subdir_path.rglob("*") if d.is_dir()])
                    
                    data_analysis["datasets"][subdir] = {
                        "exists": True,
                        "file_count": file_count,
                        "folder_count": folder_count,
                        "total_size_mb": sum(f.stat().st_size for f in subdir_path.rglob("*.*") if f.is_file()) / 1024 / 1024
                    }
                else:
                    data_analysis["datasets"][subdir] = {"exists": False}
                    data_analysis["issues"].append(f"Missing {subdir} directory")
        else:
            data_analysis["issues"].append("No organized_dataset directory")
            data_analysis["organization"]["structured"] = False
        
        # ตรวจสอบการกระจายของข้อมูล
        splits_dir = dataset_dir / "splits"
        if splits_dir.exists():
            train_dir = splits_dir / "train"
            val_dir = splits_dir / "validation" 
            test_dir = splits_dir / "test"
            
            split_balance = {}
            for split_name, split_path in [("train", train_dir), ("validation", val_dir), ("test", test_dir)]:
                if split_path.exists():
                    class_counts = {}
                    for class_dir in split_path.iterdir():
                        if class_dir.is_dir():
                            # นับไฟล์ใน front และ back
                            total_files = 0
                            for subdir in ["front", "back"]:
                                subdir_path = class_dir / subdir
                                if subdir_path.exists():
                                    total_files += len(list(subdir_path.glob("*.jpg"))) + len(list(subdir_path.glob("*.png")))
                            class_counts[class_dir.name] = total_files
                    split_balance[split_name] = class_counts
            
            data_analysis["datasets"]["split_balance"] = split_balance
            
            # ตรวจสอบความสมดุลของคลาส
            if split_balance:
                train_counts = split_balance.get("train", {})
                if train_counts:
                    min_samples = min(train_counts.values())
                    max_samples = max(train_counts.values())
                    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
                    
                    if imbalance_ratio > 3:
                        data_analysis["issues"].append(f"Class imbalance detected: ratio {imbalance_ratio:.2f}")
                        data_analysis["recommendations"].append("Consider data augmentation for minority classes")
        
        self.analysis_results["components"]["data_management"] = data_analysis
        return data_analysis
    
    def analyze_security(self):
        """วิเคราะห์ความปลอดภัย"""
        print("🔒 วิเคราะห์ Security...")
        
        security_analysis = {
            "vulnerabilities": [],
            "risk_level": "unknown",
            "recommendations": [],
            "file_permissions": {},
            "exposed_secrets": []
        }
        
        # ตรวจสอบไฟล์ที่อาจมี secrets
        sensitive_files = [
            ".env", ".env.local", ".env.prod",
            "config.json", "secrets.json",
            "api_keys.txt", "passwords.txt"
        ]
        
        for file_name in sensitive_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                security_analysis["exposed_secrets"].append(file_name)
                security_analysis["vulnerabilities"].append(f"Sensitive file exposed: {file_name}")
        
        # ตรวจสอบ API security
        api_files = ["api/main_api.py", "api/main_api_fast.py"]
        for api_file in api_files:
            api_path = self.project_root / api_file
            if api_path.exists():
                try:
                    with open(api_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # ตรวจสอบ security issues
                    if "debug=True" in content:
                        security_analysis["vulnerabilities"].append(f"Debug mode enabled in {api_file}")
                    if "allow_origins=[\"*\"]" in content:
                        security_analysis["vulnerabilities"].append(f"CORS allows all origins in {api_file}")
                    if "api_key" in content.lower() and "=" in content:
                        security_analysis["vulnerabilities"].append(f"Hardcoded API key suspected in {api_file}")
                        
                except Exception as e:
                    security_analysis["vulnerabilities"].append(f"Cannot check {api_file}: {str(e)}")
        
        # กำหนดระดับความเสี่ยง
        vuln_count = len(security_analysis["vulnerabilities"])
        if vuln_count >= 5:
            security_analysis["risk_level"] = "high"
        elif vuln_count >= 2:
            security_analysis["risk_level"] = "medium"
        else:
            security_analysis["risk_level"] = "low"
        
        # คำแนะนำ
        if security_analysis["exposed_secrets"]:
            security_analysis["recommendations"].append("Move sensitive files to .gitignore")
        if "high" in security_analysis["risk_level"]:
            security_analysis["recommendations"].append("Implement proper authentication and authorization")
            security_analysis["recommendations"].append("Review and fix CORS and debug settings")
        
        self.analysis_results["security_assessment"] = security_analysis
        return security_analysis
    
    def analyze_performance(self):
        """วิเคราะห์ประสิทธิภาพ"""
        print("⚡ วิเคราะห์ Performance...")
        
        performance_analysis = {
            "system_resources": {},
            "file_sizes": {},
            "bottlenecks": [],
            "recommendations": []
        }
        
        # System resources
        performance_analysis["system_resources"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
        
        # ตรวจสอบขนาดไฟล์ใหญ่
        large_files = []
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / 1024 / 1024
                if size_mb > 10:  # ไฟล์ใหญ่กว่า 10MB
                    large_files.append({
                        "path": str(file_path.relative_to(self.project_root)),
                        "size_mb": round(size_mb, 2)
                    })
        
        performance_analysis["file_sizes"]["large_files"] = large_files
        
        # ตรวจสอบ model size
        model_dir = self.project_root / "trained_model"
        if model_dir.exists():
            model_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1024 / 1024
            performance_analysis["file_sizes"]["model_size_mb"] = round(model_size, 2)
            
            if model_size > 100:
                performance_analysis["bottlenecks"].append("Large model size may affect loading time")
                performance_analysis["recommendations"].append("Consider model compression or optimization")
        
        # ตรวจสอบจำนวนไฟล์
        total_files = len(list(self.project_root.rglob("*.*")))
        performance_analysis["file_sizes"]["total_files"] = total_files
        
        if total_files > 10000:
            performance_analysis["bottlenecks"].append("High number of files may affect performance")
        
        self.analysis_results["performance_metrics"] = performance_analysis
        return performance_analysis
    
    def generate_comprehensive_report(self):
        """สร้างรายงานครอบคลุม"""
        print("\n📋 สร้างรายงานครอบคลุม...")
        
        # นับปัญหาทั้งหมด
        total_issues = 0
        critical_issues = 0
        
        for component, analysis in self.analysis_results["components"].items():
            issues = analysis.get("issues", [])
            total_issues += len(issues)
            
            # นับปัญหาร้าย serious
            for issue in issues:
                if any(keyword in issue.lower() for keyword in ["critical", "error", "missing", "fail"]):
                    critical_issues += 1
        
        # เพิ่ม security issues
        security_issues = len(self.analysis_results["security_assessment"].get("vulnerabilities", []))
        total_issues += security_issues
        
        # สรุปผล
        self.analysis_results["summary"] = {
            "total_components_analyzed": len(self.analysis_results["components"]),
            "total_issues_found": total_issues,
            "critical_issues": critical_issues,
            "security_risk_level": self.analysis_results["security_assessment"].get("risk_level", "unknown"),
            "overall_health": "good" if total_issues < 5 else "needs_attention" if total_issues < 15 else "critical"
        }
        
        # บันทึกรายงาน
        report_path = self.project_root / "documentation" / "analysis" / "comprehensive_system_analysis.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ รายงานบันทึกที่: {report_path}")
        
        return self.analysis_results
    
    def print_summary(self):
        """แสดงสรุปผลการวิเคราะห์"""
        print("\n" + "="*70)
        print("📊 สรุปผลการวิเคราะห์ระบบ")
        print("="*70)
        
        summary = self.analysis_results.get("summary", {})
        
        print(f"🔍 ส่วนประกอบที่วิเคราะห์: {summary.get('total_components_analyzed', 0)}")
        print(f"⚠️ ปัญหาที่พบทั้งหมด: {summary.get('total_issues_found', 0)}")
        print(f"🚨 ปัญหาร้ายแรง: {summary.get('critical_issues', 0)}")
        print(f"🔒 ระดับความเสี่ยงด้านความปลอดภัย: {summary.get('security_risk_level', 'unknown').upper()}")
        print(f"💊 สุขภาพระบบโดยรวม: {summary.get('overall_health', 'unknown').upper()}")
        
        print("\n📋 ปัญหาสำคัญที่พบ:")
        issue_count = 1
        for component, analysis in self.analysis_results["components"].items():
            issues = analysis.get("issues", [])
            for issue in issues[:3]:  # แสดงแค่ 3 ปัญหาแรกต่อส่วน
                print(f"   {issue_count}. [{component.upper()}] {issue}")
                issue_count += 1
        
        # Security issues
        security_vulns = self.analysis_results["security_assessment"].get("vulnerabilities", [])
        for vuln in security_vulns[:3]:
            print(f"   {issue_count}. [SECURITY] {vuln}")
            issue_count += 1
        
        print("\n✨ การวิเคราะห์เสร็จสิ้น!")
    
    def run_full_analysis(self):
        """รันการวิเคราะห์ทั้งหมด"""
        print("🔬 เริ่มการวิเคราะห์ระบบแบบครอบคลุม...")
        print("="*70)
        
        try:
            # วิเคราะห์แต่ละส่วน
            self.analyze_ai_models()
            self.analyze_api_systems()
            self.analyze_frontend()
            self.analyze_data_management()
            self.analyze_security()
            self.analyze_performance()
            
            # สร้างรายงาน
            self.generate_comprehensive_report()
            
            # แสดงสรุป
            self.print_summary()
            
        except Exception as e:
            print(f"\n❌ เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}")
            traceback.print_exc()

def main():
    analyzer = SystemAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()