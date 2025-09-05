"""
🏺 Amulet-AI Complete System Launcher
Launches both backend API and frontend Streamlit app in separate processes.
"""

import os
import subprocess
import sys
import time
import threading
from pathlib import Path

# Define paths
BACKEND_PATH = Path(__file__).parent / "backend" / "api" / "api_with_real_model.py"
FRONTEND_PATH = Path(__file__).parent / "frontend" / "app_streamlit.py"

def run_backend():
    """Run the backend API in a separate process"""
    print("🚀 Starting Amulet-AI Backend API...")
    try:
        subprocess.Popen([sys.executable, str(BACKEND_PATH)], 
                        creationflags=subprocess.CREATE_NEW_CONSOLE)
        print("✅ Backend API started!")
    except Exception as e:
        print(f"❌ Error starting backend: {str(e)}")

def run_frontend():
    """Run the frontend Streamlit app in a separate process"""
    print("🚀 Starting Amulet-AI Frontend...")
    try:
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(FRONTEND_PATH)],
                        creationflags=subprocess.CREATE_NEW_CONSOLE)
        print("✅ Frontend Streamlit app started!")
    except Exception as e:
        print(f"❌ Error starting frontend: {str(e)}")

def main():
    """Main launcher function"""
    print("\n" + "=" * 50)
    print("🏺 Amulet-AI Complete System Launcher 🏺")
    print("=" * 50 + "\n")
    
    # Check if files exist
    if not BACKEND_PATH.exists():
        print(f"❌ Backend file not found: {BACKEND_PATH}")
        return
    
    if not FRONTEND_PATH.exists():
        print(f"❌ Frontend file not found: {FRONTEND_PATH}")
        return
    
    # Start backend first
    run_backend()
    
    # Wait a moment for backend to initialize
    print("⏳ Waiting for backend to initialize...")
    time.sleep(5)
    
    # Then start frontend
    run_frontend()
    
    print("\n" + "=" * 50)
    print("✨ Amulet-AI System Launched Successfully! ✨")
    print("=" * 50)
    print("\n🔹 Backend API running at: http://127.0.0.1:8001")
    print("🔹 Frontend app should open in your browser")
    print("🔹 If frontend doesn't open, go to: http://localhost:8501")
    print("\n🛑 Close the console windows to stop the applications")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
