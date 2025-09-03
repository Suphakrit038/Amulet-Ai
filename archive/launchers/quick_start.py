#!/usr/bin/env python3
"""
Amulet-AI Quick Start
เริ่มใช้งานระบบ AI จริงแบบง่ายๆ
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("Amulet-AI Quick Start")
    print("   เริ่มใช้งานระบบ AI จริงง่ายๆ")
    print("=" * 60)
    print()

def main():
    print_banner()
    
    print("เลือกวิธีเริ่มระบบ:")
    print("1. เริ่มระบบครบ (Backend + Frontend)")
    print("2. เริ่มแค่ Backend")
    print("3. เริ่มแค่ Frontend")
    print("4. ทดสอบระบบ")
    print("0. ออก")
    print()

    choice = input("เลือกตัวเลข (1-4, 0=ออก): ").strip()

    if choice == "1":
        print("\nเริ่มระบบครบ...")
        os.system("python scripts/launch_real_ai_system.py")
    elif choice == "2":
        print("\nเริ่ม Backend...")
        os.system("python backend/api_with_real_model.py")
    elif choice == "3":
        print("\nเริ่ม Frontend...")
        os.system("streamlit run frontend/app_streamlit.py --server.port 8501")
    elif choice == "4":
        print("\nทดสอบระบบ...")
        os.system("python scripts/test_real_ai_system.py")
    elif choice == "0":
        print("ออกจากโปรแกรม")
    else:
        print("เลือกไม่ถูกต้อง")

if __name__ == "__main__":
    main()
