"""
Production Deployment Script for Amulet-AI
Handles production setup, health checks, and monitoring
"""

import subprocess
import sys
import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProductionDeployer:
    """Handles production deployment and monitoring"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.frontend_port = 8501
        self.backend_port = 8001
        self.processes = {}
        
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        logger.info("🔍 Checking dependencies...")
        
        required_packages = [
            'streamlit',
            'requests',
            'pillow',
            'numpy',
            'torch',
            'torchvision',
            'fastapi',
            'uvicorn'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
            logger.info("Installing missing packages...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages, check=True)
            logger.info("✅ Packages installed successfully")
        else:
            logger.info("✅ All dependencies are satisfied")
        
        return True
    
    def setup_environment(self):
        """Setup production environment"""
        logger.info("🛠️ Setting up production environment...")
        
        # Create necessary directories
        directories = [
            'logs',
            'uploads',
            'frontend/assets/css',
            'backend/logs',
            'ai_models/saved_models'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")
        
        # Set environment variables
        os.environ.update({
            'STREAMLIT_SERVER_PORT': str(self.frontend_port),
            'STREAMLIT_SERVER_ADDRESS': '0.0.0.0',
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
            'STREAMLIT_SERVER_ENABLE_CORS': 'true',
            'AMULET_API_URL': f'http://localhost:{self.backend_port}',
            'PYTHONPATH': str(self.project_root)
        })
        
        logger.info("✅ Environment setup complete")
        return True
    
    def start_backend(self):
        """Start FastAPI backend server"""
        logger.info("🚀 Starting backend server...")
        
        backend_script = self.project_root / 'backend' / 'api_with_real_model.py'
        
        if not backend_script.exists():
            logger.error(f"❌ Backend script not found: {backend_script}")
            return False
        
        try:
            # Start backend process
            cmd = [
                sys.executable, '-m', 'uvicorn',
                'backend.api_with_real_model:app',
                '--host', '0.0.0.0',
                '--port', str(self.backend_port),
                '--reload'
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.processes['backend'] = process
            
            # Wait for backend to start
            time.sleep(5)
            
            # Check if backend is running
            if self.check_backend_health():
                logger.info(f"✅ Backend server started on port {self.backend_port}")
                return True
            else:
                logger.error("❌ Backend failed to start properly")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {str(e)}")
            return False
    
    def start_frontend(self):
        """Start Streamlit frontend"""
        logger.info("🎨 Starting frontend server...")
        
        frontend_script = self.project_root / 'frontend' / 'app_modern.py'
        
        if not frontend_script.exists():
            logger.error(f"❌ Frontend script not found: {frontend_script}")
            return False
        
        try:
            # Start frontend process
            cmd = [
                sys.executable, '-m', 'streamlit', 'run',
                str(frontend_script),
                '--server.port', str(self.frontend_port),
                '--server.address', '0.0.0.0',
                '--browser.gatherUsageStats', 'false',
                '--server.enableCORS', 'true',
                '--server.enableXsrfProtection', 'true'
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.processes['frontend'] = process
            
            # Wait for frontend to start
            time.sleep(10)
            
            # Check if frontend is running
            if self.check_frontend_health():
                logger.info(f"✅ Frontend server started on port {self.frontend_port}")
                return True
            else:
                logger.error("❌ Frontend failed to start properly")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {str(e)}")
            return False
    
    def check_backend_health(self):
        """Check backend health"""
        try:
            response = requests.get(
                f'http://localhost:{self.backend_port}/health',
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def check_frontend_health(self):
        """Check frontend health"""
        try:
            response = requests.get(
                f'http://localhost:{self.frontend_port}/_stcore/health',
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def monitor_services(self):
        """Monitor running services"""
        logger.info("👀 Starting service monitoring...")
        
        try:
            while True:
                # Check backend
                if not self.check_backend_health():
                    logger.warning("⚠️ Backend health check failed")
                
                # Check frontend
                if not self.check_frontend_health():
                    logger.warning("⚠️ Frontend health check failed")
                
                # Check processes
                for service, process in self.processes.items():
                    if process.poll() is not None:
                        logger.error(f"❌ {service} process terminated unexpectedly")
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
            self.shutdown_services()
    
    def shutdown_services(self):
        """Gracefully shutdown all services"""
        logger.info("🛑 Shutting down services...")
        
        for service, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ {service} stopped gracefully")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"⚠️ {service} force killed")
            except Exception as e:
                logger.error(f"❌ Error stopping {service}: {str(e)}")
    
    def create_deployment_report(self):
        """Create deployment report"""
        report = {
            "deployment_time": datetime.now().isoformat(),
            "backend_port": self.backend_port,
            "frontend_port": self.frontend_port,
            "backend_health": self.check_backend_health(),
            "frontend_health": self.check_frontend_health(),
            "processes": {
                service: process.pid for service, process in self.processes.items()
            }
        }
        
        report_file = self.project_root / 'logs' / 'deployment_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Deployment report saved: {report_file}")
        return report
    
    def deploy(self):
        """Full deployment process"""
        logger.info("🚀 Starting Amulet-AI deployment...")
        
        try:
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Setup environment
            if not self.setup_environment():
                return False
            
            # Start backend
            if not self.start_backend():
                return False
            
            # Start frontend
            if not self.start_frontend():
                return False
            
            # Create deployment report
            report = self.create_deployment_report()
            
            logger.info("🎉 Deployment completed successfully!")
            logger.info(f"🌐 Frontend: http://localhost:{self.frontend_port}")
            logger.info(f"🔧 Backend API: http://localhost:{self.backend_port}")
            
            # Start monitoring
            self.monitor_services()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {str(e)}")
            self.shutdown_services()
            return False


def main():
    """Main deployment function"""
    deployer = ProductionDeployer()
    
    try:
        deployer.deploy()
    except KeyboardInterrupt:
        logger.info("🛑 Deployment interrupted by user")
        deployer.shutdown_services()
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        deployer.shutdown_services()


if __name__ == "__main__":
    main()
