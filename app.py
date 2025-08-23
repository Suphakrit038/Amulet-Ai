"""
Main Application Launcher for Amulet-AI
จุดเริ่มต้นหลักของระบบ
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="Amulet-AI System Launcher")
    parser.add_argument("--mode", choices=["backend", "frontend", "full"], 
                       default="full", help="Launch mode")
    parser.add_argument("--env", choices=["dev", "prod"], 
                       default="dev", help="Environment")
    
    args = parser.parse_args()
    
    if args.mode == "backend":
        from scripts.start_optimized_system import start_backend
        start_backend()
    elif args.mode == "frontend": 
        from scripts.start_optimized_system import start_frontend
        start_frontend()
    else:
        # Full system
        from scripts.start_optimized_system import main as start_system
        start_system()

if __name__ == "__main__":
    main()
