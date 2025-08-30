"""Small helper to launch the project's Streamlit app with one command.

Usage
-----
From the `frontend` folder run:

    python master_setup.py

The script finds a free port (starting at 8501), switches to the script folder, and
invokes `python -m streamlit run app_Testnew_streamlit.py --server.port <port>`.
"""
from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path


APP_FILENAME = "app_Testnew_streamlit.py"
START_PORT = 8501
MAX_PORT = 8600


def find_free_port(start: int = START_PORT, end: int = MAX_PORT) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port found in range {start}-{end}")


def main() -> int:
    repo_dir = Path(__file__).resolve().parent
    app_path = repo_dir / APP_FILENAME
    if not app_path.exists():
        print(f"Error: {APP_FILENAME} not found in {repo_dir}")
        return 2

    # Find an available port
    try:
        port = find_free_port()
    except RuntimeError as e:
        print(str(e))
        return 3

    print(f"Starting Streamlit app: {APP_FILENAME}")
    print(f"Working dir: {repo_dir}")
    print(f"Using port: {port}")

    # Use the same Python executable running this script
    python_exe = sys.executable

    cmd = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        APP_FILENAME,
        "--server.port",
        str(port),
    ]

    # Launch Streamlit in the current terminal so logs are visible. This call blocks
    # until the user stops the server (Ctrl+C).
    try:
        subprocess.run(cmd, cwd=str(repo_dir))
        return 0
    except KeyboardInterrupt:
        print("Interrupted, stopping")
        return 0
    except Exception as e:
        print("Failed to start Streamlit:", e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
