"""
ไฟล์ทดสอบการรวมฟีเจอร์วิเคราะห์และเปรียบเทียบเข้าด้วยกัน
"""
import requests
import json
from PIL import Image
import io
import base64
from io import BytesIO

def test_api_connection():
    """ทดสอบการเชื่อมต่อกับ API"""
    print("ทดสอบการเชื่อมต่อกับ API...")
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        print(f"สถานะการเชื่อมต่อ: {response.status_code}")
        print(f"ข้อมูลตอบกลับ: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: {e}")
        return False

def test_frontend_integration():
    """ทดสอบการรวมฟีเจอร์วิเคราะห์และเปรียบเทียบของ frontend"""
    print("\nทดสอบการรวมฟีเจอร์ frontend...")
    
    try:
        # ✅ ใช้โมดูลใหม่ที่รวมแล้ว
        from frontend.amulet_unified import (
            get_unified_result, 
            format_comparison_results, 
            get_dataset_info,
            find_reference_images,
            load_reference_images_for_comparison
        )
        print("- สามารถนำเข้าโมดูล amulet_unified ได้")
        
        # ทดสอบการทำงานของฟังก์ชัน get_dataset_info
        dataset_info = get_dataset_info("dataset_organized")
        print(f"- มีข้อมูลคลาสทั้งหมด: {dataset_info.get('class_count', 0)} คลาส")
        print(f"- มีรูปภาพทั้งหมด: {dataset_info.get('image_count', 0)} รูป")
        
        # ทดสอบการค้นหารูปภาพอ้างอิง
        test_class = "somdej_fatherguay"
        reference_images = find_reference_images(test_class, "dataset_organized")
        print(f"- ค้นหารูปภาพอ้างอิงสำหรับ {test_class}: พบ {len(reference_images)} รูป")
        
        # จำลองผลลัพธ์ API เพื่อทดสอบการทำงานของ get_unified_result
        mock_api_result = {
            "top1": {
                "class_name": test_class,
                "confidence": 0.95
            },
            "topk": [
                {
                    "class_name": test_class,
                    "confidence": 0.95
                }
            ]
        }
        
        # ทดสอบการรวมผลลัพธ์
        unified_result = get_unified_result(mock_api_result, "dataset_organized")
        print(f"- รวมผลลัพธ์สำเร็จ: มีรูปภาพอ้างอิง {len(unified_result.get('reference_images', {}))} รูป")
        print(f"- มีข้อมูลเปรียบเทียบ {len(unified_result.get('comparison_data', []))} รายการ")
        
        # ทดสอบการจัดรูปแบบผลลัพธ์
        formatted_result = format_comparison_results(unified_result)
        print("- จัดรูปแบบผลลัพธ์สำเร็จ")
        
        # แสดงสรุปผลการเปรียบเทียบ
        summary = formatted_result.get("summary", {})
        print(f"\nสรุปผลการเปรียบเทียบ:")
        print(f"- คลาส: {summary.get('top_class', '')}")
        print(f"- ความเชื่อมั่น: {summary.get('confidence', 0):.1f}%")
        print(f"- ความเหมือนโดยรวม: {summary.get('overall_similarity_percent', '')}")
        print(f"- ลักษณะที่เหมือน: {', '.join(summary.get('similar_features', []))}")
        print(f"- ลักษณะที่แตกต่าง: {', '.join(summary.get('different_features', []))}")
        
        return True
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการทดสอบการรวมฟีเจอร์ frontend: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("ทดสอบการรวมฟีเจอร์วิเคราะห์และเปรียบเทียบ")
    print("=" * 50)
    
    # ทดสอบการเชื่อมต่อกับ API
    api_ok = test_api_connection()
    
    # ทดสอบการรวมฟีเจอร์ frontend
    frontend_ok = test_frontend_integration()
    
    print("\n" + "=" * 50)
    print(f"ผลการทดสอบ API: {'✅ ผ่าน' if api_ok else '❌ ไม่ผ่าน'}")
    print(f"ผลการทดสอบ Frontend: {'✅ ผ่าน' if frontend_ok else '❌ ไม่ผ่าน'}")
    print("=" * 50)
