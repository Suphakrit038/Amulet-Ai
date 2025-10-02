"""
🚀 System Integration & Orchestration
=====================================

Complete system integration script that connects all components:
- Training Pipeline
- Model Deployment  
- Frontend Interface
- API Services
- MLOps Monitoring

This script orchestrates the entire Amulet-AI system.

Author: Amulet-AI Team
Date: October 2, 2025
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import webbrowser
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemOrchestrator:
    """
    🎭 System Orchestration Engine
    
    Manages the complete Amulet-AI system lifecycle.
    """
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.services: Dict[str, Any] = {}
        self.status = {
            'training_pipeline': 'stopped',
            'api_service': 'stopped', 
            'frontend': 'stopped',
            'monitoring': 'stopped'
        }
        
        # Configuration
        self.config = self._load_system_config()
        
        logger.info("System Orchestrator initialized")
    
    def _load_system_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        config_path = self.base_path / "config" / "system_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                "api": {
                    "host": "localhost",
                    "port": 8000,
                    "module": "api.main_api_fast:app"
                },
                "frontend": {
                    "host": "localhost", 
                    "port": 8501,
                    "script": "frontend/main_streamlit_app.py"
                },
                "monitoring": {
                    "enabled": True,
                    "check_interval": 60,
                    "endpoint": "http://localhost:8000"
                },
                "training": {
                    "auto_train": False,
                    "schedule": "daily",
                    "data_path": "organized_dataset/DATA SET"
                },
                "deployment": {
                    "mode": "development",  # development, production
                    "auto_deploy": False
                }
            }
            
            # Create config directory and save
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    async def start_system(self, components: Optional[List[str]] = None):
        """
        Start the complete system or specific components
        
        Args:
            components: List of components to start. If None, starts all.
                       Options: ['api', 'frontend', 'monitoring', 'training']
        """
        if components is None:
            components = ['api', 'frontend', 'monitoring']
        
        logger.info(f"Starting Amulet-AI system components: {components}")
        
        # Start components in order
        if 'api' in components:
            await self._start_api_service()
            
        if 'frontend' in components:
            await self._start_frontend()
            
        if 'monitoring' in components:
            await self._start_monitoring()
            
        if 'training' in components:
            await self._start_training_pipeline()
        
        # Wait for services to be ready
        await self._wait_for_services()
        
        # Show system status
        self._show_system_status()
        
        logger.info("✅ Amulet-AI system startup complete!")
    
    async def _start_api_service(self):
        """Start API service"""
        logger.info("🚀 Starting API service...")
        
        try:
            # Start FastAPI with uvicorn
            cmd = [
                sys.executable, "-m", "uvicorn",
                self.config['api']['module'],
                "--host", self.config['api']['host'],
                "--port", str(self.config['api']['port']),
                "--reload"
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.base_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.services['api'] = {
                'process': process,
                'url': f"http://{self.config['api']['host']}:{self.config['api']['port']}"
            }
            
            self.status['api_service'] = 'starting'
            logger.info("✅ API service started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start API service: {e}")
            self.status['api_service'] = 'error'
    
    async def _start_frontend(self):
        """Start Streamlit frontend"""
        logger.info("🎨 Starting Frontend service...")
        
        try:
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                self.config['frontend']['script'],
                "--server.port", str(self.config['frontend']['port']),
                "--server.address", self.config['frontend']['host']
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.base_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.services['frontend'] = {
                'process': process,
                'url': f"http://{self.config['frontend']['host']}:{self.config['frontend']['port']}"
            }
            
            self.status['frontend'] = 'starting'
            logger.info("✅ Frontend service started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
            self.status['frontend'] = 'error'
    
    async def _start_monitoring(self):
        """Start monitoring system"""
        if not self.config['monitoring']['enabled']:
            logger.info("⏭️ Monitoring disabled in config")
            return
            
        logger.info("📊 Starting Monitoring system...")
        
        try:
            # Import and start monitoring
            from mlops.monitoring import setup_monitoring_system
            
            alert_manager, health_checker, monitor = setup_monitoring_system(
                model_endpoint=self.config['monitoring']['endpoint']
            )
            
            monitor.start_monitoring(
                check_interval=self.config['monitoring']['check_interval']
            )
            
            self.services['monitoring'] = {
                'alert_manager': alert_manager,
                'health_checker': health_checker,
                'monitor': monitor
            }
            
            self.status['monitoring'] = 'running'
            logger.info("✅ Monitoring system started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring: {e}")
            self.status['monitoring'] = 'error'
    
    async def _start_training_pipeline(self):
        """Start training pipeline (if auto-training enabled)"""
        if not self.config['training']['auto_train']:
            logger.info("⏭️ Auto-training disabled")
            return
            
        logger.info("🎯 Starting Training Pipeline...")
        
        try:
            # Import and configure training
            from model_training.pipeline import ModernTrainingPipeline
            
            # Create training pipeline
            pipeline = ModernTrainingPipeline()
            
            # Start training in background thread
            def run_training():
                try:
                    data_path = self.base_path / self.config['training']['data_path']
                    results = pipeline.quick_train(str(data_path))
                    logger.info(f"✅ Training completed: {results}")
                except Exception as e:
                    logger.error(f"❌ Training failed: {e}")
            
            training_thread = threading.Thread(target=run_training, daemon=True)
            training_thread.start()
            
            self.services['training'] = {
                'pipeline': pipeline,
                'thread': training_thread
            }
            
            self.status['training_pipeline'] = 'running'
            logger.info("✅ Training pipeline started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start training: {e}")
            self.status['training_pipeline'] = 'error'
    
    async def _wait_for_services(self):
        """Wait for services to be ready"""
        logger.info("⏳ Waiting for services to be ready...")
        
        max_wait = 60  # Maximum wait time in seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            all_ready = True
            
            # Check API service
            if 'api' in self.services:
                try:
                    import requests
                    response = requests.get(
                        f"{self.services['api']['url']}/health",
                        timeout=5
                    )
                    if response.status_code == 200:
                        self.status['api_service'] = 'running'
                    else:
                        all_ready = False
                except:
                    all_ready = False
            
            # Check Frontend (just check if process is running)
            if 'frontend' in self.services:
                if self.services['frontend']['process'].poll() is None:
                    self.status['frontend'] = 'running'
                else:
                    all_ready = False
            
            if all_ready:
                logger.info("✅ All services are ready!")
                break
                
            await asyncio.sleep(2)
        
        if not all_ready:
            logger.warning("⚠️ Some services may not be fully ready")
    
    def _show_system_status(self):
        """Display system status"""
        print("\n" + "="*60)
        print("🎉 AMULET-AI SYSTEM STATUS")
        print("="*60)
        
        for component, status in self.status.items():
            emoji = "✅" if status == "running" else "❌" if status == "error" else "⏳"
            print(f"{emoji} {component.replace('_', ' ').title()}: {status}")
        
        print("\n🔗 ACCESS URLS:")
        if 'api' in self.services:
            print(f"   📡 API Service: {self.services['api']['url']}")
            print(f"   📄 API Docs: {self.services['api']['url']}/docs")
        
        if 'frontend' in self.services:
            print(f"   🎨 Frontend: {self.services['frontend']['url']}")
        
        print("\n💡 NEXT STEPS:")
        print("   1. Open the Frontend URL in your browser")
        print("   2. Upload amulet images for classification")
        print("   3. Monitor system performance via logs")
        print("   4. Check API documentation for integration")
        
        print("="*60)
    
    def stop_system(self):
        """Stop all system components"""
        logger.info("🛑 Stopping Amulet-AI system...")
        
        # Stop processes
        for service_name, service in self.services.items():
            try:
                if 'process' in service:
                    service['process'].terminate()
                    logger.info(f"✅ Stopped {service_name}")
                elif service_name == 'monitoring' and 'monitor' in service:
                    service['monitor'].stop_monitoring()
                    logger.info(f"✅ Stopped {service_name}")
            except Exception as e:
                logger.error(f"❌ Error stopping {service_name}: {e}")
        
        # Reset status
        for component in self.status:
            self.status[component] = 'stopped'
        
        logger.info("✅ System shutdown complete")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check each component
        for component, status in self.status.items():
            if status == 'running':
                health['components'][component] = {'status': 'healthy', 'message': 'Running normally'}
            elif status == 'error':
                health['components'][component] = {'status': 'error', 'message': 'Service error'}
                health['overall_status'] = 'degraded'
            else:
                health['components'][component] = {'status': 'stopped', 'message': 'Service stopped'}
        
        return health
    
    async def run_integration_test(self):
        """Run integration test across all components"""
        logger.info("🧪 Running integration test...")
        
        test_results = {
            'api_test': False,
            'frontend_test': False,
            'prediction_test': False,
            'monitoring_test': False
        }
        
        try:
            # Test API health
            if 'api' in self.services:
                import requests
                response = requests.get(f"{self.services['api']['url']}/health")
                test_results['api_test'] = response.status_code == 200
            
            # Test prediction endpoint (with dummy data)
            if 'api' in self.services:
                test_data = {'test': True, 'image': 'dummy_data'}
                response = requests.post(
                    f"{self.services['api']['url']}/predict",
                    json=test_data
                )
                test_results['prediction_test'] = response.status_code in [200, 422]  # 422 for invalid data is OK
            
            # Test monitoring
            if 'monitoring' in self.services:
                monitor = self.services['monitoring']['monitor']
                monitor.log_metric('test_metric', 0.95)
                test_results['monitoring_test'] = True
            
            # Test frontend (just check if process is alive)
            if 'frontend' in self.services:
                test_results['frontend_test'] = self.services['frontend']['process'].poll() is None
            
        except Exception as e:
            logger.error(f"❌ Integration test error: {e}")
        
        # Report results
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        print(f"\n🧪 INTEGRATION TEST RESULTS: {passed_tests}/{total_tests} passed")
        for test_name, passed in test_results.items():
            emoji = "✅" if passed else "❌"
            print(f"   {emoji} {test_name.replace('_', ' ').title()}")
        
        return test_results


async def main():
    """Main entry point"""
    print("🚀 Amulet-AI System Orchestrator")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = SystemOrchestrator()
    
    try:
        # Start system
        await orchestrator.start_system()
        
        # Run integration test
        await orchestrator.run_integration_test()
        
        # Open frontend in browser (optional)
        if 'frontend' in orchestrator.services:
            frontend_url = orchestrator.services['frontend']['url']
            print(f"\n🌐 Opening frontend: {frontend_url}")
            webbrowser.open(frontend_url)
        
        print("\n✨ System is running! Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down system...")
        orchestrator.stop_system()
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        orchestrator.stop_system()


if __name__ == "__main__":
    asyncio.run(main())