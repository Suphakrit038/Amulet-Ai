"""
Optimized System Startup Script
สคริปต์เริ่มต้นระบบที่ optimize แล้ว
"""
import os
import sys
import time
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required. Current version: %s", sys.version)
        return False
    logger.info("✅ Python version: %s", sys.version.split()[0])
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        ('fastapi', 'fastapi'),
        ('streamlit', 'streamlit'), 
        ('pillow', 'PIL'),
        ('numpy', 'numpy')
    ]
    missing_packages = []
    
    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            logger.info("✅ Package available: %s", display_name)
        except ImportError:
            missing_packages.append(display_name)
            logger.warning("⚠️ Missing package: %s", display_name)
    
    if missing_packages:
        logger.error("❌ Missing required packages: %s", ', '.join(missing_packages))
        logger.info("💡 Run: pip install -r requirements.txt")
        return False
    
    return True

def start_backend():
    """Start the backend API server"""
    logger.info("🚀 Starting Backend API Server...")
    
    # Try optimized API first
    api_files = [
        "backend/optimized_api.py",
        "backend/test_api.py",
        "backend/api.py"
    ]
    
    for api_file in api_files:
        if Path(api_file).exists():
            try:
                logger.info(f"🔄 Attempting to start: {api_file}")
                process = subprocess.Popen(
                    [sys.executable, api_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Give it a moment to start
                time.sleep(3)
                
                if process.poll() is None:  # Process is still running
                    logger.info(f"✅ Backend started successfully with: {api_file}")
                    return process
                else:
                    stdout, stderr = process.communicate()
                    logger.warning(f"⚠️ Failed to start {api_file}: {stderr[:200]}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error starting {api_file}: {e}")
                continue
    
    logger.error("❌ Failed to start backend API")
    return None

def start_frontend():
    """Start the frontend Streamlit app"""
    logger.info("🎨 Starting Frontend Streamlit App...")
    
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "frontend/app_streamlit.py", "--server.port", "8501"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        time.sleep(3)  # Give time to start
        
        if process.poll() is None:
            logger.info("✅ Frontend started successfully on http://localhost:8501")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ Failed to start frontend: {stderr[:200]}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error starting frontend: {e}")
        return None

def check_services():
    """Check if services are responding"""
    import requests
    import time
    
    logger.info("🔍 Checking service health...")
    
    # Check backend
    for attempt in range(10):  # Try for 10 seconds
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                logger.info("✅ Backend API is responding")
                break
        except:
            time.sleep(1)
    else:
        logger.warning("⚠️ Backend API health check failed")
    
    # Check frontend
    for attempt in range(10):  # Try for 10 seconds
        try:
            response = requests.get("http://localhost:8501", timeout=2)
            if response.status_code == 200:
                logger.info("✅ Frontend is responding")
                break
        except:
            time.sleep(1)
    else:
        logger.warning("⚠️ Frontend health check failed")

def display_summary():
    """Display system summary"""
    print("\n" + "="*60)
    print("🏺 AMULET-AI SYSTEM STARTED SUCCESSFULLY")
    print("="*60)
    print()
    print("📊 System Status:")
    print("├── 🚀 Backend API:  http://localhost:8000")
    print("├── 🎨 Frontend UI:  http://localhost:8501") 
    print("├── 📚 API Docs:     http://localhost:8000/docs")
    print("└── ❤️ Health:       http://localhost:8000/health")
    print()
    print("🤖 AI Features:")
    print("├── ✅ Advanced Image Analysis")
    print("├── ✅ Price Valuation")
    print("├── ✅ Market Recommendations")
    print("└── ✅ Multi-format Support (JPEG, PNG, HEIC)")
    print()
    print("🎯 Usage:")
    print("1. Open http://localhost:8501 in your browser")
    print("2. Upload amulet images for recognition")
    print("3. View predictions and price estimates")
    print()
    print("⭐ Press Ctrl+C to stop all services")
    print("="*60)

def main():
    """Main system startup function"""
    print("🏺 AMULET-AI SYSTEM STARTUP")
    print("="*40)
    
    # Pre-flight checks
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Change to project directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    logger.info(f"📁 Working directory: {os.getcwd()}")
    
    # Start services
    backend_process = start_backend()
    if not backend_process:
        logger.error("❌ Cannot start system without backend")
        sys.exit(1)
    
    frontend_process = start_frontend()
    
    # Health checks
    time.sleep(5)  # Wait for services to fully start
    try:
        check_services()
    except ImportError:
        logger.warning("⚠️ requests package not available, skipping health checks")
    
    # Display summary
    display_summary()
    
    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🔄 Shutting down system...")
        
        if backend_process:
            backend_process.terminate()
            logger.info("🛑 Backend stopped")
        
        if frontend_process:
            frontend_process.terminate()
            logger.info("🛑 Frontend stopped")
        
        logger.info("👋 Goodbye!")

if __name__ == "__main__":
    main()
