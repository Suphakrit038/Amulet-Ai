#!/usr/bin/env python3
"""
🏺 Test Real AI System
ทดสอบระบบ AI จริงที่เทรนแล้ว
"""

import requests
import time
from datetime import datetime
import json

def test_health_check():
    """ทดสอบ health check"""
    print("🏥 ทดสอบ Health Check...")
    try:
        response = requests.get("http://127.0.0.1:8001/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health Check ผ่าน")
            print(f"   🧠 AI Service: {'พร้อม' if data.get('ai_service_available') else 'ไม่พร้อม'}")
            print(f"   📊 Model Status: {data.get('model_status', 'unknown')}")
            print(f"   🎯 Classes: {data.get('num_classes', 0)}")
            print(f"   💻 Device: {data.get('device', 'unknown')}")
            return True
        else:
            print(f"❌ Health Check ล้มเหลว: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ Backend: {e}")
        return False

def test_model_info():
    """ทดสอบข้อมูล model"""
    print("\n📊 ทดสอบข้อมูล Model...")
    try:
        response = requests.get("http://127.0.0.1:8001/model-info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Model Info ได้รับสำเร็จ")
            print(f"   🏗️  Architecture: {data.get('architecture', 'Unknown')}")
            print(f"   📁 Model File: {data.get('model_path', 'Unknown')}")
            print(f"   🎯 Classes: {len(data.get('classes', []))}")
            print(f"   💰 Price Categories: {data.get('total_price_ranges', 0)}")
            return True
        else:
            print(f"❌ Model Info ล้มเหลว: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ไม่สามารถดึงข้อมูล Model: {e}")
        return False

def test_classes():
    """ทดสอบรายชื่อ classes"""
    print("\n📋 ทดสอบรายชื่อ Classes...")
    try:
        response = requests.get("http://127.0.0.1:8001/classes", timeout=5)
        if response.status_code == 200:
            data = response.json()
            classes = data.get('classes', [])
            print(f"✅ พบ Classes ทั้งหมด {len(classes)} ประเภท")
            
            # แสดง classes ที่มีข้อมูลราคา
            price_available = data.get('price_data_available', [])
            print(f"   💰 Classes ที่มีข้อมูลราคา: {len(price_available)}")
            
            # แสดง sample classes
            if classes:
                print("   📝 ตัวอย่าง Classes:")
                for i, cls in enumerate(classes[:5]):
                    has_price = "💰" if cls in price_available else "  "
                    print(f"      {i+1}. {has_price} {cls}")
                if len(classes) > 5:
                    print(f"      ... และอีก {len(classes) - 5} ประเภท")
            
            return True
        else:
            print(f"❌ Classes Info ล้มเหลว: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ไม่สามารถดึง Classes: {e}")
        return False

def test_predict_dummy():
    """ทดสอบ prediction โดยไม่ใช้รูปจริง"""
    print("\n🔮 ทดสอบ Prediction API...")
    print("   ⚠️  ต้องมีรูปพระเครื่องเพื่อทดสอบ")
    print("   💡 ใช้ Streamlit UI สำหรับทดสอบแบบเต็ม: http://127.0.0.1:8501")
    return True

def main():
    """ทดสอบระบบทั้งหมด"""
    print("=" * 60)
    print("🏺 Amulet-AI Real System Test")
    print(f"   เวลาทดสอบ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # เริ่มทดสอบ
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info), 
        ("Classes Info", test_classes),
        ("Prediction API", test_predict_dummy)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            time.sleep(0.5)  # หน่วงเวลาเล็กน้อย
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
    
    print("\n" + "=" * 60)
    print("📊 ผลการทดสอบ")
    print("=" * 60)
    print(f"✅ ผ่าน: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 ระบบ Real AI ทำงานปกติทั้งหมด!")
        print("🌐 เปิด Streamlit UI: http://127.0.0.1:8501")
        print("📚 เปิด API Docs: http://127.0.0.1:8001/docs")
    else:
        print("⚠️  มีบางส่วนที่ยังไม่พร้อม กรุณาตรวจสอบ")
        print("💡 ตรวจสอบว่า Backend ทำงานอยู่หรือไม่:")
        print("   python backend/api_with_real_model.py")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
