"""
Amulet-AI System Starter Script
Launches both backend API and Streamlit frontend
"""
import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def find_python_executable():
    """Find the Python executable in the virtual environment"""
    base_dir = Path(__file__).parent
    venv_dir = base_dir / '.venv'
    
    if sys.platform == 'win32':
        python_path = venv_dir / 'Scripts' / 'python.exe'
        streamlit_path = venv_dir / 'Scripts' / 'streamlit.exe'
    else:
        python_path = venv_dir / 'bin' / 'python'
        streamlit_path = venv_dir / 'bin' / 'streamlit'
    
    if not python_path.exists():
        python_path = Path(sys.executable)
    
    if not streamlit_path.exists():
        print("Warning: Streamlit not found in virtual environment. Using system Python.")
        streamlit_path = 'streamlit'
    
    return str(python_path), str(streamlit_path)

def start_backend_api():
    """Start the backend API server"""
    python_exe, _ = find_python_executable()
    
    print("🚀 Starting backend API server...")
    api_command = [python_exe, 'backend/mock_api.py']
    
    # Use subprocess.Popen to start the process
    api_process = subprocess.Popen(
        api_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Wait for API to start
    print("⏳ Waiting for API server to start...")
    time.sleep(3)
    
    return api_process

def start_streamlit_frontend():
    """Start the Streamlit frontend application"""
    _, streamlit_exe = find_python_executable()
    
    print("🚀 Starting Streamlit frontend...")
    streamlit_command = [streamlit_exe, 'run', 'frontend/app_streamlit.py', '--server.port', '8501']
    
    # Use subprocess.Popen to start the process
    streamlit_process = subprocess.Popen(
        streamlit_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Wait for Streamlit to start
    print("⏳ Waiting for Streamlit to start...")
    time.sleep(5)
    
    # Open browser
    webbrowser.open('http://localhost:8501')
    
    return streamlit_process

def main():
    """Main function to start both servers"""
    print("🏺 Amulet-AI System Starter")
    print("===========================")
    
    try:
        # Start backend API
        api_process = start_backend_api()
        
        # Start Streamlit frontend
        streamlit_process = start_streamlit_frontend()
        
        print("✅ Both services started successfully!")
        print("📊 Streamlit frontend available at: http://localhost:8501")
        print("🔌 Backend API available at: http://127.0.0.1:8000")
        print("\nPress Ctrl+C to stop all services")
        
        # Keep the script running to capture output
        while True:
            try:
                # Read and print output from API process
                api_output = api_process.stdout.readline()
                if api_output:
                    print(f"[API] {api_output.strip()}")
                
                # Read and print output from Streamlit process
                streamlit_output = streamlit_process.stdout.readline()
                if streamlit_output:
                    print(f"[Streamlit] {streamlit_output.strip()}")
                
                # Check if processes are still running
                if api_process.poll() is not None:
                    print("⚠️ API process has terminated")
                    break
                
                if streamlit_process.poll() is not None:
                    print("⚠️ Streamlit process has terminated")
                    break
                
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping all services...")
                api_process.terminate()
                streamlit_process.terminate()
                break
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
