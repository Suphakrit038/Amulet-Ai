"""
Unit Tests for Amulet-AI API
Basic tests to ensure API functionality
"""
import pytest
import requests
import io
from PIL import Image

API_URL = "http://localhost:8000"

def create_test_image():
    """สร้างภาพทดสอบ"""
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_api_health():
    """ทดสอบ health check endpoint"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        assert response.status_code == 200
        print("✅ Health check passed")
    except requests.exceptions.ConnectionError:
        print("⚠️ API server not running - skipping test")
        pytest.skip("API server not available")

def test_predict_endpoint():
    """ทดสอบ predict endpoint"""
    try:
        test_image = create_test_image()
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        assert response.status_code == 200
        
        data = response.json()
        assert "top1" in data
        assert "topk" in data
        assert "valuation" in data
        assert "recommendations" in data
        
        print("✅ Predict endpoint test passed")
    except requests.exceptions.ConnectionError:
        print("⚠️ API server not running - skipping test")
        pytest.skip("API server not available")

if __name__ == "__main__":
    test_api_health()
    test_predict_endpoint()
    print("🎉 All tests completed!")
