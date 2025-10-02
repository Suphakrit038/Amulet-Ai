#!/usr/bin/env python3
"""
🚀 Amulet-AI Production Launcher
Quick start script for production deployment
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import streamlit
        import numpy
        import pandas
        import requests
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def start_api():
    """Start FastAPI server"""
    print("🚀 Starting API server...")
    api_dir = Path(__file__).parent / "api"
    os.chdir(api_dir)
    subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "main_api:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ])
    print("✅ API server started on http://localhost:8000")

def start_frontend():
    """Start Streamlit frontend"""
    print("🎨 Starting frontend...")
    frontend_dir = Path(__file__).parent / "frontend" 
    os.chdir(frontend_dir)
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        "main_app.py",
        "--server.port", "8501"
    ])
    print("✅ Frontend started on http://localhost:8501")

def main():
    """Main launcher function"""
    print("🔮 Amulet-AI Production Launcher")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Start services
    start_api()
    time.sleep(3)  # Wait for API to start
    start_frontend()
    
    print("\n🎉 Amulet-AI is now running!")
    print("📱 Web App: http://localhost:8501")
    print("🔌 API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all services")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Amulet-AI...")

if __name__ == "__main__":
    main()