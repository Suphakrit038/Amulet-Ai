"""
Amulet AI System Launcher (Simplified)
--------------------------------------
This script launches the complete Amulet AI system with both backend and frontend.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print banner message when starting the system."""
    print("\n" + "="*70)
    print("                     AMULET AI SYSTEM LAUNCHER")
    print("="*70)
    print("Starting the complete Amulet AI system...")
    print("="*70 + "\n")

def start_backend():
    """Start the FastAPI backend server."""
    print("[1/2] Starting Backend API server...")
    
    # Get the path to the backend module
    backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
    
    # Start backend process (non-blocking)
    backend_cmd = [sys.executable, "-m", "uvicorn", "backend.api.api:app", "--host", "0.0.0.0", "--port", "8001"]
    backend_process = subprocess.Popen(
        backend_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give the backend time to start
    time.sleep(2)
    
    # Check if backend started successfully
    if backend_process.poll() is not None:
        print("Error: Backend failed to start.")
        _, stderr = backend_process.communicate()
        print(f"Error details: {stderr}")
        sys.exit(1)
    
    print("Backend API server running at http://localhost:8001")
    return backend_process

def start_frontend():
    """Start the Streamlit frontend."""
    print("[2/2] Starting Streamlit frontend...")
    
    # Get the path to the frontend module
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    streamlit_app = os.path.join(frontend_path, "app_streamlit.py")
    
    # Start frontend process (non-blocking)
    frontend_cmd = [sys.executable, "-m", "streamlit", "run", streamlit_app, "--server.port", "8501"]
    frontend_process = subprocess.Popen(
        frontend_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give the frontend time to start
    time.sleep(3)
    
    # Check if frontend started successfully
    if frontend_process.poll() is not None:
        print("Error: Frontend failed to start.")
        _, stderr = frontend_process.communicate()
        print(f"Error details: {stderr}")
        sys.exit(1)
    
    print("Streamlit frontend running at http://localhost:8501")
    return frontend_process

def open_browser():
    """Open web browser to the Streamlit app."""
    print("\nOpening web browser to Streamlit app...")
    webbrowser.open("http://localhost:8501")

def main():
    """Main function to start the complete system."""
    print_banner()
    
    # Start backend and frontend
    backend_process = start_backend()
    frontend_process = start_frontend()
    
    # Open browser after a short delay
    time.sleep(2)
    open_browser()
    
    print("\n" + "="*70)
    print("✅ Amulet AI system is now running!")
    print("• Frontend: http://localhost:8501")
    print("• Backend API: http://localhost:8001")
    print("• API Documentation: http://localhost:8001/docs")
    print("="*70)
    print("\nPress Ctrl+C to stop the system...\n")
    
    try:
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Amulet AI system...")
        
        # Terminate processes
        backend_process.terminate()
        frontend_process.terminate()
        
        print("System shutdown complete.")

if __name__ == "__main__":
    main()
