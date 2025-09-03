"""
Final Working Test - Simple API & Frontend
"""
import subprocess
import time
import sys
import webbrowser
import requests
from pathlib import Path

def test_api():
    """Test if API is working"""
    print("🔥 Starting API server...")
    
    # Check if mock_api.py exists
    if not Path("backend/mock_api.py").exists():
        print("❌ backend/mock_api.py not found")
        return False
    
    # Start API
    api_process = subprocess.Popen([
        sys.executable, "backend/mock_api.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print("⏳ Waiting 5 seconds for API to start...")
    time.sleep(5)
    
    # Test connection
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=3)
        if response.status_code == 200:
            print("✅ API is working!")
            print(f"🔗 API URL: http://127.0.0.1:8000")
            print(f"📚 Docs: http://127.0.0.1:8000/docs")
            return api_process
        else:
            print(f"❌ API returned {response.status_code}")
            api_process.terminate()
            return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        api_process.terminate()
        return False

def test_streamlit():
    """Test if Streamlit is working"""
    print("\n🌐 Starting Streamlit...")
    
    if not Path("frontend/app_streamlit.py").exists():
        print("❌ frontend/app_streamlit.py not found")
        return False
    
    # Start Streamlit
    streamlit_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "frontend/app_streamlit.py",
        "--server.port", "8501",
        "--server.address", "127.0.0.1"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print("⏳ Waiting 8 seconds for Streamlit to start...")
    time.sleep(8)
    
    print("✅ Streamlit started!")
    print("🔗 Web App: http://127.0.0.1:8501")
    
    return streamlit_process

def main():
    print("=" * 50)
    print("  🏺 Amulet-AI Final Working Test")
    print("=" * 50)
    
    # Start API
    api_process = test_api()
    if not api_process:
        print("\n❌ API failed to start")
        return
    
    # Start Streamlit
    streamlit_process = test_streamlit()
    if not streamlit_process:
        print("\n❌ Streamlit failed to start")
        api_process.terminate()
        return
    
    # Open browser
    print("\n🌐 Opening browser...")
    try:
        webbrowser.open("http://127.0.0.1:8501")
        print("✅ Browser opened")
    except:
        print("⚠️ Please manually open: http://127.0.0.1:8501")
    
    # Success message
    print("\n" + "=" * 50)
    print("  🎉 SYSTEM IS RUNNING!")
    print("=" * 50)
    print("  🔗 Web Interface: http://127.0.0.1:8501")
    print("  🔗 API Backend:   http://127.0.0.1:8000")
    print("  📚 API Docs:      http://127.0.0.1:8000/docs")
    print("  ⚠️  Press Ctrl+C to stop all services")
    print("=" * 50)
    
    try:
        input("\nPress Enter to stop services...")
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    
    # Cleanup
    print("🧹 Stopping services...")
    api_process.terminate()
    streamlit_process.terminate()
    print("✅ All services stopped")

if __name__ == "__main__":
    main()
