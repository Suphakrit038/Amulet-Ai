"""
🎯 Amulet-AI Quick Start Guide
ระบบจดจำพระเครื่องไทยขั้นสูง - เริ่มใช้งานง่าย ๆ
"""
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load system configuration"""
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Warning: Could not load config.json: {e}")
        return {}

def check_python_version() -> bool:
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.9+")
        return False

def check_dependencies() -> bool:
    """Check if required packages are installed"""
    required_packages = [
        "fastapi", "uvicorn", "streamlit", "tensorflow", 
        "pillow", "numpy", "pandas", "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def show_system_info():
    """Display system information"""
    config = load_config()
    project_info = config.get("project", {})
    
    print("🏺 ยินดีต้อนรับสู่ Amulet-AI")
    print("=" * 60)
    print(f"📁 Project: {project_info.get('name', 'Amulet-AI')}")
    print(f"🏷️ Version: {project_info.get('version', '2.0.0-Optimized')}")
    print(f"📝 Description: {project_info.get('description', 'Advanced Thai Buddhist Amulet Recognition System')}")
    print("=" * 60)
    print()

def show_features():
    """Show system features"""
    config = load_config()
    classes = config.get("classes", {})
    
    print("✨ ฟีเจอร์หลัก:")
    print("🧠 AI Recognition Engine - จดจำพระเครื่องด้วยปัญญาประดิษฐ์")
    print("🎨 Web Interface - เว็บแอปพลิเคชันใช้งานง่าย") 
    print("🚀 REST API - API สำหรับนักพัฒนา")
    print("📊 Advanced Analytics - การวิเคราะห์ขั้นสูง")
    print("🔍 Similarity Search - ค้นหารูปคล้ายกัน")
    print("💰 Price Estimation - ประเมินราคาพระเครื่อง")
    print()
    
    print(f"🏷️ รองรับการจดจำ {len(classes)} ประเภท:")
    for class_id, class_name in classes.items():
        print(f"   {int(class_id) + 1}. {class_name}")
    print()

def show_quick_start_steps():
    """Show quick start steps"""
    print("🚀 เริ่มใช้งานใน 4 ขั้นตอน:")
    print()
    
    print("1️⃣ ติดตั้ง Dependencies:")
    print("   pip install -r requirements.txt")
    print()
    
    print("2️⃣ เริ่มระบบ Backend:")
    print("   python app.py --mode backend")
    print("   🌐 API: http://localhost:8000")
    print("   📚 Docs: http://localhost:8000/docs")
    print()
    
    print("3️⃣ เริ่ม Frontend (terminal ใหม่):")
    print("   python app.py --mode frontend")
    print("   🎨 Web UI: http://localhost:8501")
    print()
    
    print("4️⃣ หรือเริ่มทั้งระบบ:")
    print("   python app.py --mode full")
    print()

def show_usage_tips():
    """Show usage tips"""
    print("💡 เคล็ดลับการใช้งาน:")
    print("📷 อัปโหลดรูปที่ชัดเจน ไม่เบลอ")
    print("💡 แสงสว่างเพียงพอ หลีกเลี่ยงเงา")
    print("📐 ใช้รูปที่มีพระเครื่องเต็มเฟรม")
    print("🎯 ถ่ายรูปตรงด้านหน้าเพื่อผลลัพธ์ที่ดีที่สุด")
    print()

def show_project_structure():
    """Show basic project structure"""
    print("📁 โครงสร้างโปรเจค์:")
    print("├── 📄 app.py              # Main launcher")
    print("├── 📄 config.json         # System configuration")
    print("├── 📄 README.md           # Complete documentation")
    print("├── 📁 backend/            # API services")
    print("├── 📁 frontend/           # Web interface")
    print("├── 📁 development/        # Development tools")
    print("└── 📁 docs/               # Documentation")
    print()

def run_system_check():
    """Run complete system check"""
    print("🔍 ตรวจสอบระบบ:")
    print("-" * 30)
    
    # Check Python version
    if not check_python_version():
        return False
    
    print()
    
    # Check dependencies
    print("📦 ตรวจสอบ Dependencies:")
    deps_ok = check_dependencies()
    
    print()
    
    if deps_ok:
        print("✅ ระบบพร้อมใช้งาน!")
        return True
    else:
        print("❌ กรุณาติดตั้ง dependencies ที่ขาดหายไป")
        return False

def main():
    """Main quick start function"""
    show_system_info()
    show_features()
    show_quick_start_steps()
    show_usage_tips()
    show_project_structure()
    
    print("🔧 ต้องการตรวจสอบระบบ? (y/n): ", end="")
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes', 'ใช่', '1']:
            print()
            system_ok = run_system_check()
            
            if system_ok:
                print("\n🎉 พร้อมเริ่มใช้งาน Amulet-AI!")
                print("💻 เรียกใช้: python app.py")
            else:
                print("\n⚠️ กรุณาแก้ไขปัญหาก่อนใช้งาน")
        else:
            print("\n📖 อ่านเอกสารเพิ่มเติม: README.md")
            
    except KeyboardInterrupt:
        print("\n👋 ขอบคุณที่ใช้ Amulet-AI!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
