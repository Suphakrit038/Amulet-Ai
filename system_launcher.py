"""
🚀 System Integration Launcher
สคริปต์เปิดระบบทั้งหมด - Backend API + Frontend + AI Model
"""

import subprocess
import time
import os
import sys
import signal
import logging
from pathlib import Path
import requests
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemLauncher:
    """System Integration Launcher"""
    
    def __init__(self):
        self.processes = []
        self.base_dir = Path(__file__).parent
        self.running = True
        
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        required_packages = [
            ('fastapi', 'fastapi'), 
            ('uvicorn', 'uvicorn'), 
            ('streamlit', 'streamlit'), 
            ('requests', 'requests'), 
            ('torch', 'torch'), 
            ('PIL', 'pillow'), 
            ('numpy', 'numpy')
        ]
        
        missing_packages = []
        
        for import_name, package_name in required_packages:
            try:
                __import__(import_name)
                logger.info(f"✅ {package_name} - OK")
            except ImportError:
                missing_packages.append(package_name)
                logger.error(f"❌ {package_name} - Missing")
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Install with: pip install " + " ".join(missing_packages))
            return False
        
        return True
    
    def check_ai_model(self):
        """Check if AI model files exist"""
        model_files = [
            "ai_models/training_output/step5_final_model.pth",
            "ai_models/training_output/PRODUCTION_MODEL_INFO.json"
        ]
        
        all_exist = True
        for model_file in model_files:
            full_path = self.base_dir / model_file
            if full_path.exists():
                logger.info(f"✅ {model_file} - Found")
            else:
                logger.warning(f"⚠️ {model_file} - Not found (will use default)")
                all_exist = False
        
        return all_exist
    
    def start_backend_api(self):
        """Start the Backend API server"""
        try:
            logger.info("🚀 Starting Backend API...")
            
            # Change to project directory
            os.chdir(self.base_dir)
            
            # Start FastAPI server
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "backend.api:app", 
                "--host", "127.0.0.1",
                "--port", "8000",
                "--reload",
                "--log-level", "info"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append(('Backend API', process))
            logger.info("✅ Backend API process started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Backend API: {e}")
            return False
    
    def wait_for_api(self, max_attempts=30):
        """Wait for API to be ready"""
        logger.info("⏳ Waiting for API to be ready...")
        
        for attempt in range(max_attempts):
            try:
                response = requests.get("http://127.0.0.1:8000/health", timeout=2)
                if response.status_code == 200:
                    logger.info("✅ API is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
            logger.info(f"⏳ Attempt {attempt + 1}/{max_attempts}...")
        
        logger.error("❌ API failed to start within timeout")
        return False
    
    def start_frontend(self):
        """Start the Streamlit frontend"""
        try:
            logger.info("🎨 Starting Frontend...")
            
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(self.base_dir / "frontend" / "enhanced_streamlit_app.py"),
                "--server.port", "8501",
                "--server.address", "127.0.0.1",
                "--browser.gatherUsageStats", "false",
                "--server.headless", "false"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes.append(('Frontend', process))
            logger.info("✅ Frontend process started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Frontend: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor running processes"""
        while self.running:
            try:
                for name, process in self.processes:
                    if process.poll() is not None:
                        logger.error(f"❌ {name} process has stopped!")
                        return False
                
                time.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal")
                break
        
        return True
    
    def stop_all_processes(self):
        """Stop all running processes"""
        logger.info("🛑 Stopping all processes...")
        
        self.running = False
        
        for name, process in self.processes:
            try:
                logger.info(f"Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                    logger.info(f"✅ {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ Force killing {name}")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
        
        logger.info("✅ All processes stopped")
    
    def print_system_info(self):
        """Print system information"""
        print("\n" + "="*60)
        print("🏺 AMULET-AI SYSTEM INTEGRATION")
        print("="*60)
        print(f"📅 Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Base Directory: {self.base_dir}")
        print(f"🐍 Python: {sys.version}")
        print("="*60)
        print("🌐 URLs:")
        print("  Backend API:  http://127.0.0.1:8000")
        print("  API Docs:     http://127.0.0.1:8000/docs")
        print("  Frontend:     http://127.0.0.1:8501")
        print("="*60)
        print("⌨️ Controls:")
        print("  Ctrl+C:       Stop all services")
        print("="*60)
    
    def run(self):
        """Run the complete system"""
        try:
            # Print system info
            self.print_system_info()
            
            # Check dependencies
            logger.info("🔍 Checking dependencies...")
            if not self.check_dependencies():
                logger.error("❌ Missing dependencies. Please install required packages.")
                return False
            
            # Check AI model
            logger.info("🧠 Checking AI model...")
            self.check_ai_model()
            
            # Start backend API
            if not self.start_backend_api():
                logger.error("❌ Failed to start backend")
                return False
            
            # Wait for API to be ready
            if not self.wait_for_api():
                logger.error("❌ API not responding")
                self.stop_all_processes()
                return False
            
            # Start frontend
            if not self.start_frontend():
                logger.error("❌ Failed to start frontend")
                self.stop_all_processes()
                return False
            
            # Wait a bit for frontend to start
            time.sleep(3)
            
            # Print success message
            print("\n🎉 SYSTEM READY!")
            print("✅ Backend API: http://127.0.0.1:8000")
            print("✅ Frontend:    http://127.0.0.1:8501")
            print("📖 API Docs:   http://127.0.0.1:8000/docs")
            print("\n💡 The system is now running. Access the frontend to use Amulet-AI!")
            print("🛑 Press Ctrl+C to stop all services\n")
            
            # Monitor processes
            self.monitor_processes()
            
        except KeyboardInterrupt:
            logger.info("🛑 Received shutdown signal")
        except Exception as e:
            logger.error(f"❌ System error: {e}")
        finally:
            self.stop_all_processes()
    
    def test_api_connection(self):
        """Test API connection and AI model"""
        try:
            logger.info("🧪 Testing API connection...")
            
            # Health check
            response = requests.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ API Health: {data}")
            else:
                logger.error(f"❌ API Health check failed: {response.status_code}")
            
            # AI model info
            response = requests.get("http://127.0.0.1:8000/ai-info", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"🧠 AI Model: {data.get('model_name', 'Unknown')}")
                logger.info(f"📊 Categories: {data.get('categories_count', 0)}")
                logger.info(f"⚙️ Parameters: {data.get('model_parameters', 0):,}")
            else:
                logger.warning(f"⚠️ AI info not available: {response.status_code}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ API test failed: {e}")
            return False

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    print("\n🛑 Received interrupt signal. Shutting down...")
    sys.exit(0)

def main():
    """Main function"""
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    
    launcher = SystemLauncher()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            # Test mode - just check if everything is working
            if launcher.check_dependencies():
                launcher.check_ai_model()
                logger.info("✅ System check complete")
            return
        elif sys.argv[1] == '--api-only':
            # Start only API
            launcher.start_backend_api()
            launcher.wait_for_api()
            launcher.test_api_connection()
            launcher.monitor_processes()
            return
    
    # Start complete system
    launcher.run()

if __name__ == "__main__":
    main()
