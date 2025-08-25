"""
🏺 Amulet-AI System Launcher
Advanced Thai Buddhist Amulet Recognition System
จุดเริ่มต้นหลักของระบบจดจำพระเครื่องไทย
"""
import sys
import argparse
import asyncio
import subprocess
from pathlib import Path
import json
from typing import Optional

# Enhanced imports with fallbacks
try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: uvicorn not installed. Install with: pip install uvicorn")
    uvicorn = None
    UVICORN_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_config() -> dict:
    """Load system configuration"""
    config_path = PROJECT_ROOT / "config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Warning: Could not load config.json: {e}")
        return {}

def start_backend(env: str = "development") -> None:
    """Start FastAPI backend server"""
    print("🚀 Starting Amulet-AI Backend...")
    
    if not UVICORN_AVAILABLE:
        print("❌ Backend dependencies not found. Please install: pip install fastapi uvicorn")
        return
    
    config = load_config()
    env_config = config.get("environments", {}).get(env, {})
    
    host = env_config.get("api_host", "127.0.0.1")
    port = env_config.get("api_port", 8000)
    
    try:
        uvicorn.run(
            "backend.api:app",
            host=host,
            port=port,
            reload=(env == "development"),
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Backend startup failed: {e}")

def start_frontend(env: str = "development") -> None:
    """Start Streamlit frontend"""
    print("🎨 Starting Amulet-AI Frontend...")
    
    config = load_config()
    env_config = config.get("environments", {}).get(env, {})
    
    port = env_config.get("frontend_port", 8501)
    
    try:
        # Check if frontend file exists
        frontend_file = PROJECT_ROOT / "frontend" / "app_streamlit.py"
        if not frontend_file.exists():
            print(f"❌ Frontend file not found: {frontend_file}")
            return
            
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(frontend_file),
            "--server.port", str(port),
            "--server.address", "localhost"
        ])
    except ImportError:
        print("❌ Streamlit not found. Please install: pip install streamlit")
    except Exception as e:
        print(f"❌ Frontend startup failed: {e}")

async def start_full_system(env: str = "development") -> None:
    """Start both backend and frontend"""
    print("🏺 Starting Full Amulet-AI System...")
    print("=" * 50)
    
    config = load_config()
    env_config = config.get("environments", {}).get(env, {})
    
    api_port = env_config.get("api_port", 8000)
    frontend_port = env_config.get("frontend_port", 8501)
    
    print(f"🔧 Environment: {env}")
    print(f"🚀 Backend API: http://localhost:{api_port}")
    print(f"🎨 Frontend UI: http://localhost:{frontend_port}")
    print(f"📚 API Docs: http://localhost:{api_port}/docs")
    print("=" * 50)
    
    # For full system, start backend first, then suggest frontend command
    print("Starting backend server...")
    print("💡 To start frontend, run in another terminal:")
    print(f"   python app.py --mode frontend --env {env}")
    
    start_backend(env)

def show_system_info() -> None:
    """Show system information"""
    config = load_config()
    
    print("🏺 Amulet-AI System Information")
    print("=" * 40)
    print(f"📁 Project: {config.get('project', {}).get('name', 'Amulet-AI')}")
    print(f"🏷️ Version: {config.get('project', {}).get('version', '2.0.0')}")
    print(f"📝 Description: {config.get('project', {}).get('description', 'Thai Buddhist Amulet Recognition')}")
    print("=" * 40)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="🏺 Amulet-AI System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                    # Start full system (development)
  python app.py --mode backend     # Backend only
  python app.py --mode frontend    # Frontend only  
  python app.py --env prod         # Production mode
  python app.py --info             # Show system info
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["backend", "frontend", "full"], 
        default="full", 
        help="Launch mode (default: full)"
    )
    
    parser.add_argument(
        "--env", 
        choices=["development", "production"], 
        default="development", 
        help="Environment mode (default: development)"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show system information"
    )
    
    args = parser.parse_args()
    
    if args.info:
        show_system_info()
        return
    
    try:
        if args.mode == "backend":
            start_backend(args.env)
        elif args.mode == "frontend":
            start_frontend(args.env)
        else:
            asyncio.run(start_full_system(args.env))
            
    except KeyboardInterrupt:
        print("\n👋 Amulet-AI system stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
