"""
🏺 Amulet-AI Simple Test Script  
ทดสอบระบบอย่างง่าย
"""

import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def test_imports():
    """Test basic imports"""
    print("🧪 Testing imports...")
    try:
        import streamlit
        print("  ✅ Streamlit")
        import fastapi
        print("  ✅ FastAPI") 
        import uvicorn
        print("  ✅ Uvicorn")
        import PIL
        print("  ✅ PIL/Pillow")
        import numpy
        print("  ✅ NumPy")
        import pandas
        print("  ✅ Pandas")
        import requests
        print("  ✅ Requests")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def start_api_server():
    """Start the mock API server"""
    print("\n🚀 Starting API server...")
    
    try:
        # Check if mock API exists
        if Path("backend/mock_api.py").exists():
            api_process = subprocess.Popen([
                sys.executable, "backend/mock_api.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            print("  ❌ backend/mock_api.py not found")
            return None
        
        # Wait for startup
        print("  ⏳ Waiting for API to start...")
        time.sleep(3)
        
        # Test connection
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                print("  ✅ API server started successfully")
                return api_process
            else:
                print(f"  ❌ API returned status {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"  ❌ API connection failed: {e}")
            return None
            
    except Exception as e:
        print(f"  ❌ Failed to start API: {e}")
        return None

def start_streamlit():
    """Start Streamlit app"""
    print("\n🌐 Starting Streamlit...")
    
    try:
        if Path("frontend/app_streamlit.py").exists():
            streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "frontend/app_streamlit.py",
                "--server.port", "8501",
                "--server.address", "127.0.0.1"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            print("  ❌ frontend/app_streamlit.py not found")
            return None
        
        print("  ⏳ Waiting for Streamlit to start...")
        time.sleep(5)
        print("  ✅ Streamlit started")
        return streamlit_process
        
    except Exception as e:
        print(f"  ❌ Failed to start Streamlit: {e}")
        return None

def open_browser():
    """Open browser to the app"""
    print("\n🌐 Opening browser...")
    try:
        import webbrowser
        webbrowser.open("http://127.0.0.1:8501")
        print("  ✅ Browser opened")
    except Exception as e:
        print(f"  ⚠️ Could not open browser: {e}")
        print("  🔗 Please open: http://127.0.0.1:8501")

def main():
    """Main test function"""
    print("=" * 50)
    print("  🏺 Amulet-AI Quick Test & Launch")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Run setup first:")
        print("   python setup_complete_system.py")
        return
    
    # Start API
    api_process = start_api_server()
    if not api_process:
        print("\n❌ Failed to start API server")
        return
    
    # Start Streamlit
    streamlit_process = start_streamlit()
    if not streamlit_process:
        print("\n❌ Failed to start Streamlit")
        if api_process:
            api_process.terminate()
        return
    
    # Open browser
    open_browser()
    
    # Success message
    print("\n" + "=" * 50)
    print("  🎉 Amulet-AI is Running!")
    print("=" * 50)
    print("  🔗 Web App: http://127.0.0.1:8501")
    print("  🔗 API:     http://127.0.0.1:8000")  
    print("  📚 Docs:    http://127.0.0.1:8000/docs")
    print("  ⚠️  Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        if api_process:
            api_process.terminate()
        if streamlit_process:
            streamlit_process.terminate()
        print("✅ Services stopped")

if __name__ == "__main__":
    main()
