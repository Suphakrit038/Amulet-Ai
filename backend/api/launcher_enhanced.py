"""
Launcher for Enhanced API with Reference Images
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Import the API module
from backend.api.api_with_reference_images import app

if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Launch the Amulet-AI Enhanced API with Reference Images")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the API server to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind the API server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting Amulet-AI Enhanced API on {args.host}:{args.port}")
    
    uvicorn.run(
        "backend.api.api_with_reference_images:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
