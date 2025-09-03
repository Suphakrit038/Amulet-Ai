#!/usr/bin/env python3
"""
🚀 Complete Amulet-AI System Launcher
เปิดระบบ AI ครบครัน - Backend + Frontend พร้อมกัน
"""

import subprocess
import time
import os
import sys
import threading
import socket
from pathlib import Path

def check_port(host, port):
    """Check if a port is available"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def wait_for_service(host, port, service_name, max_wait=30):
    """Wait for a service to be available"""
    print(f"⏳ Waiting for {service_name} to start on {host}:{port}...")
    
    for i in range(max_wait):
        if check_port(host, port):
            print(f"✅ {service_name} is ready!")
            return True
        time.sleep(1)
        print(f"   Waiting... ({i+1}/{max_wait})")
    
    print(f"❌ {service_name} failed to start within {max_wait} seconds")
    return False

def run_backend():
    """Run the FastAPI backend server"""
    try:
        print("🔥 Starting Backend API Server...")
        
        # Try using the venv python directly
        venv_python = Path(".venv/Scripts/python.exe")
        if venv_python.exists():
            python_cmd = str(venv_python)
        else:
            python_cmd = "python"
        
        # Start the backend using uvicorn
        cmd = [
            python_cmd, "-m", "uvicorn", 
            "backend.api:app", 
            "--host", "127.0.0.1", 
            "--port", "8000", 
            "--reload"
        ]
        
        process = subprocess.Popen(cmd, cwd=os.getcwd())
        return process
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def run_frontend():
    """Run the Streamlit frontend"""
    try:
        print("🌐 Starting Frontend Web Interface...")
        
        # Try using the venv python directly
        venv_python = Path(".venv/Scripts/python.exe")
        if venv_python.exists():
            python_cmd = str(venv_python)
        else:
            python_cmd = "python"
        
        # Start the frontend using streamlit
        cmd = [
            python_cmd, "-m", "streamlit", "run", 
            "frontend/app_streamlit.py",
            "--server.port", "8501",
            "--server.address", "127.0.0.1"
        ]
        
        process = subprocess.Popen(cmd, cwd=os.getcwd())
        return process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def main():
    """Main launcher function"""
    print("🏺 Amulet-AI Complete System Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("backend/api.py").exists():
        print("❌ Error: Please run this script from the Amulet-AI project root directory")
        sys.exit(1)
    
    # Check for required files
    required_files = [
        "backend/api.py",
        "frontend/app_streamlit.py",
        "requirements.txt"
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        sys.exit(1)
    
    # Start backend in a separate thread
    print("🚀 Starting backend server...")
    backend_process = run_backend()
    
    if backend_process is None:
        print("❌ Failed to start backend process")
        sys.exit(1)
    
    # Wait for backend to be ready
    if not wait_for_service("127.0.0.1", 8000, "Backend API", max_wait=30):
        print("❌ Backend failed to start properly")
        backend_process.terminate()
        sys.exit(1)
    
    # Start frontend
    print("🚀 Starting frontend server...")
    frontend_process = run_frontend()
    
    if frontend_process is None:
        print("❌ Failed to start frontend process")
        backend_process.terminate()
        sys.exit(1)
    
    # Wait for frontend to be ready
    if not wait_for_service("127.0.0.1", 8501, "Frontend Web App", max_wait=30):
        print("❌ Frontend failed to start properly")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(1)
    
    # System ready
    print("\n" + "=" * 60)
    print("🎉 Amulet-AI System is Ready!")
    print("=" * 60)
    print(f"🔗 Backend API:     http://127.0.0.1:8000")
    print(f"📚 API Docs:        http://127.0.0.1:8000/docs")
    print(f"🌐 Web Interface:   http://127.0.0.1:8501")
    print("=" * 60)
    print("\n💡 Tips:")
    print("   • Upload both front and back images for best results")
    print("   • Supported formats: JPG, PNG, HEIC, WebP, BMP, TIFF")
    print("   • Maximum file size: 10MB per image")
    print("   • The AI model is trained on 10 Thai amulet categories")
    print("\n⚠️  Press Ctrl+C to stop both servers")
    
    try:
        # Keep the launcher running and monitor processes
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("\n❌ Backend process has stopped unexpectedly")
                break
            
            if frontend_process.poll() is not None:
                print("\n❌ Frontend process has stopped unexpectedly") 
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Amulet-AI system...")
        
        # Graceful shutdown
        try:
            backend_process.terminate()
            frontend_process.terminate()
            
            # Wait for processes to terminate
            backend_process.wait(timeout=5)
            frontend_process.wait(timeout=5)
            
            print("✅ System shutdown complete")
            
        except subprocess.TimeoutExpired:
            print("⚠️ Forcefully killing processes...")
            backend_process.kill()
            frontend_process.kill()
            print("✅ System stopped")

if __name__ == "__main__":
    main()
