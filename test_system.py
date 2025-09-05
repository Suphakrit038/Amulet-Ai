"""
Quick Test Script for Amulet-AI System
ทดสอบระบบ Amulet-AI แบบง่าย
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """ทดสอบการ import โมดูลต่างๆ"""
    print("🔍 ทดสอบการ import โมดูล...")
    
    test_modules = [
        ("streamlit", "Streamlit web framework"),
        ("requests", "HTTP library"),
        ("PIL", "Pillow image library"),
        ("numpy", "NumPy scientific computing"),
        ("pathlib", "Path utilities")
    ]
    
    results = []
    
    for module_name, description in test_modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {module_name} ({description}) - version: {version}")
            results.append(True)
        except ImportError as e:
            print(f"❌ {module_name} - ไม่พบ: {e}")
            results.append(False)
    
    return all(results)

def test_file_structure():
    """ทดสอบโครงสร้างไฟล์"""
    print("\n📁 ทดสอบโครงสร้างไฟล์...")
    
    project_root = Path(__file__).parent
    required_files = [
        "frontend/config.py",
        "frontend/analytics.py", 
        "frontend/components/ui_components.py",
        "frontend/components/layout_manager.py",
        "frontend/app_modern.py",
        "ai_models/modern_model.py",
        "backend/api_with_real_model.py"
    ]
    
    results = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
            results.append(True)
        else:
            print(f"❌ {file_path} - ไม่พบไฟล์")
            results.append(False)
    
    return all(results)

def test_config():
    """ทดสอบการตั้งค่า"""
    print("\n⚙️ ทดสอบการตั้งค่า...")
    
    try:
        # Add frontend to path
        sys.path.append(str(Path(__file__).parent / "frontend"))
        
        from config import API_URL, IMAGE_SETTINGS, UI_SETTINGS
        print(f"✅ API_URL: {API_URL}")
        print(f"✅ IMAGE_SETTINGS: {len(IMAGE_SETTINGS)} settings")
        print(f"✅ UI_SETTINGS: {len(UI_SETTINGS)} settings")
        return True
    except Exception as e:
        print(f"❌ ไม่สามารถโหลด config: {e}")
        return False

def test_directories():
    """ทดสอบและสร้างโฟลเดอร์จำเป็น"""
    print("\n📂 ตรวจสอบโฟลเดอร์จำเป็น...")
    
    project_root = Path(__file__).parent
    required_dirs = [
        "logs",
        "uploads",
        "frontend/assets/css",
        "backend/logs",
        "ai_models/saved_models"
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 สร้างโฟลเดอร์: {dir_path}")
        else:
            print(f"✅ {dir_path}")
    
    return True

def main():
    """ฟังก์ชันหลัก"""
    print("""
    🔮 Amulet-AI System Test
    ========================
    
    """)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_file_structure),
        ("Configuration Test", test_config),
        ("Directory Test", test_directories)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append(result)
            
            if result:
                print(f"✅ {test_name} - สำเร็จ")
            else:
                print(f"❌ {test_name} - ไม่สำเร็จ")
                
        except Exception as e:
            print(f"❌ {test_name} - เกิดข้อผิดพลาด: {e}")
            results.append(False)
    
    print(f"\n{'='*50}")
    print("📊 สรุปผลการทดสอบ")
    print('='*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"ผ่าน: {passed}/{total} tests")
    
    if all(results):
        print("🎉 ระบบพร้อมใช้งาน!")
        print("\n🚀 วิธีการเริ่มระบบ:")
        print("1. รันคำสั่ง: python launch_amulet_ai.py")
        print("2. หรือใช้: start_amulet_ai.bat")
        print("3. หรือรันตรง: streamlit run frontend/app_modern.py")
    else:
        print("⚠️ ระบบยังไม่พร้อม กรุณาแก้ไขปัญหาที่พบ")
        
        print("\n🔧 การแก้ไขปัญหา:")
        print("1. ติดตั้ง dependencies: pip install -r requirements.txt")
        print("2. ตรวจสอบ Python version >= 3.8")
        print("3. ตรวจสอบไฟล์ที่ขาดหายไป")

if __name__ == "__main__":
    main()
