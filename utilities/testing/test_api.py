#!/usr/bin/env python3
"""
Quick API Test - ทดสอบ API ด่วน
"""

import requests
import io
from PIL import Image
import numpy as np

# Create a test image
def create_test_image():
    """สร้างรูปภาพทดสอบ"""
    # Create a simple test image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

def test_api():
    """ทดสอบ API"""
    try:
        # Test health
        print("🏥 Testing health endpoint...")
        health_response = requests.get("http://localhost:8000/health")
        print(f"Health Status: {health_response.status_code}")
        print(f"Health Data: {health_response.json()}")
        
        # Test prediction
        print("\n🔮 Testing prediction endpoint...")
        test_image = create_test_image()
        
        files = {'file': ('test.png', test_image.getvalue(), 'image/png')}
        predict_response = requests.post("http://localhost:8000/predict", files=files)
        
        print(f"Prediction Status: {predict_response.status_code}")
        
        if predict_response.status_code == 200:
            result = predict_response.json()
            print("✅ Prediction successful!")
            print(f"Predicted class: {result['prediction']['class']}")
            print(f"Confidence: {result['prediction']['confidence']:.3f}")
            print("Top predictions:")
            for cls, prob in sorted(result['prediction']['probabilities'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]:
                print(f"  - {cls}: {prob:.3f}")
        else:
            print(f"❌ Prediction failed: {predict_response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()