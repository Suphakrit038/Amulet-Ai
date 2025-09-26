#!/usr/bin/env python3
"""
🧪 E2E Test Suite for Amulet-AI API
End-to-End testing สำหรับ API endpoints

Usage: pytest tests/e2e/test_api_complete.py -v
"""

import pytest
import requests
import time
import os
import json
import tempfile
from pathlib import Path
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Test configuration
API_BASE_URL = "http://localhost:8000"
FRONTEND_BASE_URL = "http://localhost:8501"
TEST_TIMEOUT = 30  # seconds

class TestAPIHealth:
    """Test API health and basic functionality"""
    
    def test_api_is_running(self):
        """Test API server is running and responding"""
        try:
            response = requests.get(f"{API_BASE_URL}/", timeout=5)
            assert response.status_code == 200
            
            data = response.json()
            assert "service" in data
            assert "version" in data
            assert data["version"] == "4.0.0"
            
        except requests.exceptions.ConnectionError:
            pytest.fail("API server is not running. Start with: python backend/api/main_api.py")
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert health_data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "model_status" in health_data
        assert "timestamp" in health_data
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=10)
        assert response.status_code == 200
        
        info = response.json()
        assert info["model_version"] == "4.0.0"
        assert "supported_classes" in info
        assert len(info["supported_classes"]) == 3
        assert "phra_somdej" in info["supported_classes"]
        assert "phra_rod" in info["supported_classes"] 
        assert "phra_nang_phya" in info["supported_classes"]
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint for monitoring"""
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        assert response.status_code == 200
        
        metrics = response.json()
        assert "requests_total" in metrics
        assert "uptime_minutes" in metrics
        assert isinstance(metrics["requests_total"], int)

class TestPredictionAPI:
    """Test prediction functionality"""
    
    @pytest.fixture
    def sample_images(self):
        """Provide sample images for testing"""
        fixtures_dir = Path("tests/fixtures")
        
        # Check if fixtures exist
        test_images = {
            "phra_somdej": fixtures_dir / "phra_somdej_front.jpg",
            "phra_rod": fixtures_dir / "phra_rod_front.jpg", 
            "phra_nang_phya": fixtures_dir / "phra_nang_phya_front.jpg"
        }
        
        # If fixtures don't exist, use dataset samples
        for class_name, path in test_images.items():
            if not path.exists():
                dataset_samples = list(Path(f"dataset/test/{class_name}").glob("*.jpg"))
                if dataset_samples:
                    test_images[class_name] = dataset_samples[0]
        
        return test_images
    
    def test_predict_happy_path(self, sample_images):
        """Test successful prediction with valid images"""
        for class_name, image_path in sample_images.items():
            if not image_path.exists():
                pytest.skip(f"Test image not found: {image_path}")
            
            with open(image_path, 'rb') as f:
                files = {
                    'front_image': (f'{class_name}_front.jpg', f, 'image/jpeg'),
                    'back_image': (f'{class_name}_back.jpg', f, 'image/jpeg')
                }
                
                start_time = time.time()
                response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=TEST_TIMEOUT)
                latency = time.time() - start_time
                
                # Check response
                assert response.status_code == 200, f"Failed for {class_name}: {response.text}"
                
                result = response.json()
                
                # Check response structure
                assert result["status"] == "success"
                assert "is_supported" in result
                assert "predicted_class" in result
                assert "confidence" in result
                assert "model_version" in result
                assert "timestamp" in result
                
                # Check prediction quality
                if result["is_supported"]:
                    assert result["predicted_class"] in ["phra_somdej", "phra_rod", "phra_nang_phya"]
                    assert 0.0 <= result["confidence"] <= 1.0
                    assert result["model_version"] == "4.0.0"
                
                # Check performance
                assert latency < 2.0, f"Latency {latency:.2f}s exceeds 2s limit for {class_name}"
                
                print(f"✅ {class_name}: {result.get('predicted_class', 'N/A')} "
                      f"(confidence: {result.get('confidence', 0):.3f}, latency: {latency:.3f}s)")
    
    def test_file_size_validation(self):
        """Test file size limits"""
        # Create a large fake image (>10MB)
        large_content = b"fake_image_header\xFF\xD8\xFF\xE0" + b"x" * (11 * 1024 * 1024)
        
        files = {
            'front_image': ('large.jpg', large_content, 'image/jpeg'),
            'back_image': ('large.jpg', large_content, 'image/jpeg')
        }
        
        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=TEST_TIMEOUT)
        
        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()
    
    def test_invalid_file_format(self):
        """Test rejection of invalid file formats"""
        # Text file disguised as image
        files = {
            'front_image': ('fake.jpg', b'This is not an image', 'image/jpeg'),
            'back_image': ('fake.jpg', b'This is not an image', 'image/jpeg')
        }
        
        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=TEST_TIMEOUT)
        
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()
    
    def test_missing_files(self):
        """Test handling of missing files"""
        # Only front image provided
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            tmp.write(b"fake_image_content")
            tmp.flush()
            
            files = {'front_image': open(tmp.name, 'rb')}
            
            response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=TEST_TIMEOUT)
            
            # Should require both front and back images
            assert response.status_code == 422  # Validation error
    
    def test_ood_detection(self):
        """Test out-of-domain detection"""
        # Create a simple synthetic image that should be detected as OOD
        import numpy as np
        from PIL import Image
        import io
        
        # Create random noise image
        noise_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pil_image = Image.fromarray(noise_image)
        
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {
            'front_image': ('noise.jpg', img_bytes.getvalue(), 'image/jpeg'),
            'back_image': ('noise.jpg', img_bytes.getvalue(), 'image/jpeg')
        }
        
        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=TEST_TIMEOUT)
        
        assert response.status_code == 200
        result = response.json()
        
        # Should be detected as OOD or have very low confidence
        assert (result["is_supported"] is False or 
                result.get("confidence", 1.0) < 0.3), "OOD detection failed for noise image"

class TestPerformanceAndLoad:
    """Test performance and load handling"""
    
    def test_concurrent_requests(self, sample_images):
        """Test handling multiple concurrent requests"""
        if not sample_images:
            pytest.skip("No sample images available")
        
        # Use first available image
        test_image = next(iter(sample_images.values()))
        if not test_image.exists():
            pytest.skip("Test image not available")
        
        async def make_request(session, request_id):
            """Make a single async request"""
            with open(test_image, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('front_image', f, filename=f'test_{request_id}.jpg')
                data.add_field('back_image', f, filename=f'test_{request_id}.jpg')
                
                start_time = time.time()
                try:
                    async with session.post(f"{API_BASE_URL}/predict", data=data, timeout=TEST_TIMEOUT) as response:
                        await response.json()
                        return {
                            'request_id': request_id,
                            'status_code': response.status,
                            'latency': time.time() - start_time,
                            'success': response.status == 200
                        }
                except Exception as e:
                    return {
                        'request_id': request_id,
                        'error': str(e),
                        'latency': time.time() - start_time,
                        'success': False
                    }
        
        async def run_concurrent_test():
            """Run concurrent requests"""
            concurrent_requests = 10
            
            async with aiohttp.ClientSession() as session:
                tasks = [make_request(session, i) for i in range(concurrent_requests)]
                results = await asyncio.gather(*tasks)
            
            return results
        
        # Run the test
        results = asyncio.run(run_concurrent_test())
        
        # Analyze results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        success_rate = len(successful_requests) / len(results)
        
        if successful_requests:
            avg_latency = sum(r['latency'] for r in successful_requests) / len(successful_requests)
            max_latency = max(r['latency'] for r in successful_requests)
            
            print(f"📊 Concurrent test results:")
            print(f"   Success rate: {success_rate:.1%}")
            print(f"   Average latency: {avg_latency:.3f}s")
            print(f"   Max latency: {max_latency:.3f}s")
            print(f"   Failed requests: {len(failed_requests)}")
            
            # Assertions
            assert success_rate >= 0.9, f"Success rate {success_rate:.1%} below 90%"
            assert avg_latency < 3.0, f"Average latency {avg_latency:.3f}s exceeds 3s"
            assert max_latency < 5.0, f"Max latency {max_latency:.3f}s exceeds 5s"
        else:
            pytest.fail("No successful requests in concurrent test")
    
    def test_memory_stability(self, sample_images):
        """Test memory usage doesn't grow excessively"""
        import psutil
        
        if not sample_images:
            pytest.skip("No sample images available")
        
        test_image = next(iter(sample_images.values()))
        if not test_image.exists():
            pytest.skip("Test image not available")
        
        # Get initial memory usage
        initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # Make multiple requests
        num_requests = 20
        latencies = []
        
        for i in range(num_requests):
            with open(test_image, 'rb') as f:
                files = {
                    'front_image': f,
                    'back_image': f
                }
                
                start_time = time.time()
                response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=TEST_TIMEOUT)
                latency = time.time() - start_time
                latencies.append(latency)
                
                if response.status_code != 200:
                    print(f"Warning: Request {i} failed with status {response.status_code}")
        
        # Check final memory usage
        final_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Check latency stability
        avg_latency = sum(latencies) / len(latencies)
        latency_std = (sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5
        
        print(f"🧠 Memory stability test:")
        print(f"   Memory growth: {memory_growth:.1f} MB")
        print(f"   Average latency: {avg_latency:.3f}s ± {latency_std:.3f}s")
        
        # Assertions (generous limits for CI environment)
        assert memory_growth < 200, f"Memory growth {memory_growth:.1f}MB suggests memory leak"
        assert avg_latency < 2.0, f"Average latency {avg_latency:.3f}s exceeds 2s"

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_malformed_requests(self):
        """Test handling of malformed requests"""
        # Empty request
        response = requests.post(f"{API_BASE_URL}/predict", timeout=TEST_TIMEOUT)
        assert response.status_code == 422  # Validation error
        
        # Invalid JSON in non-file request
        response = requests.post(f"{API_BASE_URL}/predict", 
                               json={"invalid": "data"}, 
                               timeout=TEST_TIMEOUT)
        assert response.status_code == 422
    
    def test_nonexistent_endpoint(self):
        """Test 404 handling"""
        response = requests.get(f"{API_BASE_URL}/nonexistent", timeout=TEST_TIMEOUT)
        assert response.status_code == 404
        
        error_data = response.json()
        assert "error" in error_data
        assert "available_endpoints" in error_data
    
    def test_method_not_allowed(self):
        """Test wrong HTTP method handling"""
        response = requests.get(f"{API_BASE_URL}/predict", timeout=TEST_TIMEOUT)
        assert response.status_code == 405  # Method not allowed
    
    def test_request_timeout(self):
        """Test request timeout handling"""
        # This test simulates a scenario where the server might be slow
        # We'll use a very short timeout to test timeout handling
        
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=0.001)  # 1ms timeout
            pytest.fail("Request should have timed out")
        except requests.exceptions.Timeout:
            # Expected behavior
            pass

class TestSecurityBasics:
    """Basic security tests"""
    
    def test_rate_limiting(self):
        """Test rate limiting (if implemented)"""
        # Make rapid requests to test rate limiting
        rapid_requests = 50
        responses = []
        
        with open("dataset/test/phra_somdej/phra_somdej_001.jpg", 'rb') as test_file:
            files = {'front_image': test_file, 'back_image': test_file}
            
            for i in range(rapid_requests):
                try:
                    response = requests.post(f"{API_BASE_URL}/predict", 
                                           files=files, 
                                           timeout=1)
                    responses.append(response.status_code)
                except requests.exceptions.Timeout:
                    responses.append(408)  # Timeout
                except Exception:
                    responses.append(500)  # Server error
        
        # Check if any requests were rate limited (429 status)
        rate_limited_count = responses.count(429)
        
        if rate_limited_count > 0:
            print(f"✅ Rate limiting active: {rate_limited_count} requests limited")
        else:
            print("⚠️ No rate limiting detected")
        
        # At minimum, server should handle the load gracefully
        server_errors = responses.count(500)
        assert server_errors < rapid_requests * 0.1, f"Too many server errors: {server_errors}"
    
    def test_malicious_payloads(self):
        """Test protection against malicious payloads"""
        # SQL injection-like payload in filename
        malicious_filename = "'; DROP TABLE users; --"
        
        files = {
            'front_image': (malicious_filename, b'fake_image_content', 'image/jpeg'),  
            'back_image': (malicious_filename, b'fake_image_content', 'image/jpeg')
        }
        
        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=TEST_TIMEOUT)
        
        # Should not crash or expose internal errors
        assert response.status_code in [400, 422], "Server should handle malicious filenames gracefully"
        
        # Response should not contain sensitive information
        response_text = response.text.lower()
        sensitive_terms = ['traceback', 'exception', 'error:', 'stack trace', 'internal server error']
        
        for term in sensitive_terms:
            assert term not in response_text, f"Response contains sensitive term: {term}"

# Test fixtures setup/teardown
@pytest.fixture(scope="session", autouse=True)
def check_test_environment():
    """Check if test environment is properly set up"""
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            pytest.exit("API health check failed. Is the server running?")
    except requests.exceptions.ConnectionError:
        pytest.exit(f"Cannot connect to API at {API_BASE_URL}. Please start the server first.")
    
    # Check if test data exists
    test_data_dirs = ["dataset/test", "tests/fixtures"]
    available_dirs = [d for d in test_data_dirs if Path(d).exists()]
    
    if not available_dirs:
        pytest.exit("No test data directories found. Please ensure dataset/test or tests/fixtures exists.")
    
    print(f"✅ Test environment ready. API: {API_BASE_URL}")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])