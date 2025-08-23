#!/usr/bin/env python3
"""
Setup script for Amulet-AI system
Initializes all components and prepares the system for use
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    logger.info(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8+ required")
        return False
    
    logger.info("✅ Python version compatible")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "data/scraped", 
        "temp",
        "logs",
        "exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")
    
    logger.info("✅ All directories created")

def install_requirements():
    """Install Python requirements"""
    if not os.path.exists("requirements.txt"):
        logger.warning("⚠️ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )

def initialize_components():
    """Initialize AI components"""
    logger.info("🤖 Initializing AI components")
    
    try:
        # Initialize FAISS similarity search
        logger.info("🔍 Initializing FAISS similarity search")
        from backend.similarity_search import initialize_similarity_search
        search_engine = initialize_similarity_search()
        logger.info(f"✅ FAISS index status: {search_engine.get_index_stats()}")
        
        # Initialize price estimator with mock data
        logger.info("💰 Initializing price estimator")
        from backend.price_estimator import create_mock_training_data, PriceEstimator
        mock_data = create_mock_training_data()
        estimator = PriceEstimator()
        estimator.train_model(mock_data)
        logger.info("✅ Price estimator trained with mock data")
        
        # Initialize market scraper
        logger.info("🕷️ Initializing market scraper")
        from backend.market_scraper import update_market_data
        update_market_data()
        logger.info("✅ Market data collected")
        
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Some components not available: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Component initialization failed: {e}")
        return False

def check_dataset():
    """Check if dataset is available"""
    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        logger.warning("⚠️ Dataset directory not found")
        logger.info("💡 Please organize your dataset as:")
        logger.info("dataset/")
        logger.info("  ├── หลวงพ่อกวยแหวกม่าน/")
        logger.info("  ├── โพธิ์ฐานบัว/") 
        logger.info("  ├── ฐานสิงห์/")
        logger.info("  └── สีวลี/")
        return False
    
    # Count images in each class
    classes = ["หลวงพ่อกวยแหวกม่าน", "โพธิ์ฐานบัว", "ฐานสิงห์", "สีวลี"]
    total_images = 0
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.exists(class_path):
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(image_files)
            total_images += count
            logger.info(f"📊 {class_name}: {count} images")
        else:
            logger.warning(f"⚠️ Class directory not found: {class_name}")
    
    logger.info(f"📊 Total images: {total_images}")
    
    if total_images < 20:
        logger.warning("⚠️ Very small dataset. Consider adding more images for better performance.")
    
    return total_images > 0

def create_startup_scripts():
    """Create convenient startup scripts"""
    
    # Backend startup script
    backend_script = """@echo off
echo Starting Amulet-AI Backend...
cd /d "%~dp0"
python -m uvicorn backend.api:app --reload --port 8000
pause
"""
    
    with open("start_backend.bat", "w", encoding="utf-8") as f:
        f.write(backend_script)
    
    # Frontend startup script
    frontend_script = """@echo off
echo Starting Amulet-AI Frontend...
cd /d "%~dp0"
python -m streamlit run frontend/app_streamlit.py
pause
"""
    
    with open("start_frontend.bat", "w", encoding="utf-8") as f:
        f.write(frontend_script)
    
    # Combined startup script
    combined_script = """@echo off
echo Starting Amulet-AI System...
cd /d "%~dp0"

echo Starting Backend...
start "Amulet-AI Backend" cmd /k "python -m uvicorn backend.api:app --reload --port 8000"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend...
start "Amulet-AI Frontend" cmd /k "python -m streamlit run frontend/app_streamlit.py"

echo System started! Check the opened windows.
pause
"""
    
    with open("start_system.bat", "w", encoding="utf-8") as f:
        f.write(combined_script)
    
    logger.info("✅ Startup scripts created")

def main():
    """Main setup function"""
    logger.info("🚀 Setting up Amulet-AI System")
    logger.info("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        logger.error("❌ Failed to install requirements")
        return
    
    # Check dataset
    has_dataset = check_dataset()
    
    # Initialize components
    components_ready = initialize_components()
    
    # Create startup scripts
    create_startup_scripts()
    
    # Summary
    logger.info("=" * 50)
    logger.info("📋 Setup Summary:")
    logger.info(f"✅ Python: Compatible")
    logger.info(f"✅ Requirements: Installed")
    logger.info(f"{'✅' if has_dataset else '⚠️'} Dataset: {'Available' if has_dataset else 'Not found'}")
    logger.info(f"{'✅' if components_ready else '⚠️'} AI Components: {'Ready' if components_ready else 'Partial'}")
    logger.info(f"✅ Startup Scripts: Created")
    
    if not has_dataset:
        logger.info("\n💡 Next Steps:")
        logger.info("1. Add your amulet images to the dataset directory")
        logger.info("2. Run 'python train_model.py' to train a real model")
        logger.info("3. Use 'start_system.bat' to launch the application")
    else:
        logger.info("\n🎉 System ready to use!")
        logger.info("Run 'start_system.bat' to launch the application")
        
        if components_ready:
            logger.info("💡 To train a real model: python train_model.py")
    
    logger.info("\n📚 Usage:")
    logger.info("- Backend API: http://localhost:8000")
    logger.info("- Frontend UI: http://localhost:8501")
    logger.info("- API Docs: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
