"""
Master Setup Script for Amulet-AI
สคริปต์หลักสำหรับตั้งค่าและเริ่มระบบทั้งหมด
"""
import os
import sys
import subprocess
import json
from pathlib import Path

def create_main_config():
    """Create main configuration file"""
    config = {
        "project": {
            "name": "Amulet-AI",
            "version": "2.0.0-Optimized",
            "description": "Advanced Thai Buddhist Amulet Recognition System"
        },
        "environments": {
            "development": {
                "debug": True,
                "api_host": "127.0.0.1",
                "api_port": 8000,
                "frontend_port": 8501
            },
            "production": {
                "debug": False,
                "api_host": "0.0.0.0", 
                "api_port": 8000,
                "frontend_port": 8501
            }
        },
        "ai": {
            "mode": "advanced_simulation",
            "supported_formats": ["JPEG", "PNG", "HEIC", "WebP", "BMP", "TIFF"],
            "max_file_size_mb": 10,
            "cache_enabled": True
        },
        "classes": {
            "0": "หลวงพ่อกวยแหวกม่าน",
            "1": "โพธิ์ฐานบัว",
            "2": "ฐานสิงห์",
            "3": "สีวลี"
        }
    }
    
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("✅ Created config.json")

def create_requirements():
    """Create consolidated requirements.txt"""
    requirements = [
        "# Core Framework",
        "fastapi>=0.116.0",
        "uvicorn[standard]>=0.30.0",
        "streamlit>=1.48.0",
        "",
        "# AI/ML Libraries", 
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "",
        "# Image Processing",
        "pillow>=10.0.0",
        "pillow-heif>=0.13.0",
        "",
        "# Data Processing",
        "pandas>=2.0.0",
        "requests>=2.31.0",
        "",
        "# Optional AI Extensions",
        "faiss-cpu>=1.7.4",
        "scrapy>=2.11.0",
        "",
        "# Development Tools",
        "pytest>=7.4.0",
        "black>=23.0.0",
        "flake8>=6.0.0"
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    print("✅ Updated requirements.txt")

def create_main_launcher():
    """Create main application launcher"""
    launcher_code = '''"""
Main Application Launcher for Amulet-AI
จุดเริ่มต้นหลักของระบบ
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="Amulet-AI System Launcher")
    parser.add_argument("--mode", choices=["backend", "frontend", "full"], 
                       default="full", help="Launch mode")
    parser.add_argument("--env", choices=["dev", "prod"], 
                       default="dev", help="Environment")
    
    args = parser.parse_args()
    
    if args.mode == "backend":
        from scripts.start_optimized_system import start_backend
        start_backend()
    elif args.mode == "frontend": 
        from scripts.start_optimized_system import start_frontend
        start_frontend()
    else:
        # Full system
        from scripts.start_optimized_system import main as start_system
        start_system()

if __name__ == "__main__":
    main()
'''
    
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(launcher_code)
    
    print("✅ Created app.py (main launcher)")

def cleanup_duplicates():
    """Remove duplicate files and organize"""
    
    # Files to remove (duplicates or obsolete)
    files_to_remove = [
        "README_OLD.md",
        "setup.py",  # Replace with this script
        "USAGE.md"   # Info moved to docs/
    ]
    
    for file in files_to_remove:
        if Path(file).exists():
            os.remove(file)
            print(f"🗑️ Removed {file}")
    
    # Check for empty directories
    empty_dirs = ["exports", "logs"]  # Will be recreated when needed
    for dir_name in empty_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and not any(dir_path.iterdir()):
            dir_path.rmdir()
            print(f"🗑️ Removed empty directory {dir_name}")

def create_quick_start_guide():
    """Create quick start guide"""
    guide = """# 🚀 Quick Start Guide

## Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Start system
python app.py
```

## Usage Options
```bash
# Full system (default)
python app.py --mode full

# Backend only
python app.py --mode backend

# Frontend only  
python app.py --mode frontend

# Production mode
python app.py --env prod
```

## Access Points
- 🎨 Web Interface: http://localhost:8501
- 🚀 API Server: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs

## Project Structure
See PROJECT_STRUCTURE.md for detailed architecture.
"""
    
    with open("QUICKSTART.md", "w", encoding="utf-8") as f:
        f.write(guide)
    
    print("✅ Created QUICKSTART.md")

def main():
    """Main setup function"""
    print("🏺 AMULET-AI MASTER SETUP")
    print("="*40)
    
    print("🔧 Setting up project configuration...")
    create_main_config()
    create_requirements()
    create_main_launcher()
    
    print("🧹 Cleaning up duplicate files...")
    cleanup_duplicates()
    
    print("📚 Creating documentation...")  
    create_quick_start_guide()
    
    print("\n" + "="*50)
    print("🎉 SETUP COMPLETE!")
    print("="*50)
    print("📁 Project organized and optimized")
    print("🚀 Ready to use: python app.py")
    print("📚 Quick guide: QUICKSTART.md")
    print("📖 Full docs: docs/")
    print("="*50)

if __name__ == "__main__":
    main()
