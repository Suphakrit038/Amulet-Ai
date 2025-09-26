# 🧪 Comprehensive Test Criteria for Amulet-AI v4.0
## เกณฑ์การทดสอบแบบครอบคลุมสำหรับ Amulet-AI v4.0

**เวอร์ชันเอกสาร**: 1.0  
**วันที่**: 27 กันยายน 2025  
**สำหรับ**: CalibratedClassifierCV + RandomForest, PCA 30 dims, OOD ensemble

---

## 📋 Executive Summary

ระบบปัจจุบันแสดงผล metrics สมบูรณ์แบบ (100%) — **สัญญาณชัดเจนของ overfitting** เนื่องจาก:
- ข้อมูลฝึกเพียง 35 รูป/คลาส (รวม 105 รูปทั้งหมด)
- ขาด infrastructure สำคัญ (Docker, DB, Torch, ONNX, Prometheus, logs directory)

**วัตถุประสงค์ของชุดทดสอบนี้:**
1. ✅ ยืนยันคุณภาพข้อมูลและป้องกัน data leakage
2. ✅ ยืนยันว่า metrics จริง ๆ มาจาก evaluation ที่เหมาะสม
3. ✅ ตรวจสอบ OOD ensemble และ pair-handling (front/back)
4. ✅ ตรวจสอบ API contract, latency, memory, error handling
5. ✅ ตรวจสอบ readiness ก่อน deploy
6. ✅ ตั้ง CI gates สำหรับ regression protection

---

## 🎯 Acceptance Criteria (เกณฑ์การยอมรับ)

### 📊 Model Performance
```yaml
per_class_f1: ≥ 0.75 (baseline), goal ≥ 0.85
balanced_accuracy: ≥ 0.80
ood_auroc: ≥ 0.90
ood_recall: ≥ 0.90
calibration_ece: ≤ 0.08 (adjusted for small dataset)
overfit_gap: < 0.15 (train_f1 - val_f1)
```

### ⚡ Performance
```yaml
p95_latency_single_predict: < 2.0s (CPU)
memory_footprint: < 500MB (target < 250MB)
error_rate: < 0.5%
cold_start_time: < 10s (goal < 5s)
```

### 🏗️ System
```yaml
user_task_success_rate: ≥ 90%
model_versioning: present (/model/info returns version)
docker_build: success
logging_structured: present (request_id, model_version, latency)
```

---

## 🧪 Test Categories & Detailed Tests

### A. Data Quality & Leakage Tests

#### A1. Data Corruption Check
**วัตถุประสงค์**: ตรวจสอบความถูกต้องของไฟล์รูปภาพ

```bash
# สร้างและรันสคริปต์ตรวจสอบ
python tests/data/check_images_readable.py --data-dir dataset/
```

**เกณฑ์ผ่าน**: 0 corrupted files  
**เกณฑ์ล้มเหลว**: มี corrupted files > 0

#### A2. Duplicate Detection
**วัตถุประสงค์**: หา duplicates ที่อาจทำให้เกิด data leakage

```bash
python tests/data/find_duplicates.py --data-dir dataset/ --threshold 5
```

**เกณฑ์ผ่าน**: Duplicate ratio < 5%  
**เกณฑ์ล้มเหลว**: Duplicate ratio ≥ 5%

#### A3. Class Balance Check
**วัตถุประสงค์**: ตรวจสอบจำนวนข้อมูลต่อคลาส

```python
# Expected current state
phra_nang_phya: 35 images (20 train + 10 test + 5 val)
phra_rod: 35 images (20 train + 10 test + 5 val)  
phra_somdej: 35 images (20 train + 10 test + 5 val)
```

**เกณฑ์เตือน**: < 50 รูป/คลาส (แนะนำ data augmentation)  
**เกณฑ์วิกฤติ**: < 20 รูป/คลาส

#### A4. Train/Test Split Sanity
**วัตถุประสงค์**: ป้องกัน data leakage

```bash
python tests/data/validate_split.py --train dataset/train --test dataset/test --val dataset/validation
```

**เกณฑ์ผ่าน**: Zero overlap between splits

### B. Unit Tests (Code Logic)

#### B1. AdvancedFeatureExtractor Tests
```python
def test_feature_extractor_output_shape():
    """Test feature extraction produces correct dimensions"""
    extractor = AdvancedFeatureExtractor()
    front = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    back = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    features = extractor.extract_dual_features(front, back)
    
    # Expected: statistical(8*2) + edge(4*2) + lbp(8*2) + color(9*2) + hu(7*2) + pair(9) = 81
    assert features.shape == (81,), f"Expected 81 features, got {features.shape}"
    assert np.all(np.isfinite(features)), "All features must be finite"
```

#### B2. Model Save/Load Roundtrip
```python
def test_model_save_load_consistency():
    """Test model predictions remain consistent after save/load"""
    # Fixed seed input
    np.random.seed(42)
    test_features = np.random.randn(1, 30)  # After PCA
    
    # Load original model
    classifier = EnhancedProductionClassifier()
    classifier.load_model("trained_model")
    
    pred_original = classifier.classifier.predict_proba(test_features)
    
    # Save and reload
    temp_path = "temp_model_test"
    classifier.save_model(temp_path)
    
    classifier_new = EnhancedProductionClassifier()
    classifier_new.load_model(temp_path)
    
    pred_reloaded = classifier_new.classifier.predict_proba(test_features)
    
    np.testing.assert_array_almost_equal(pred_original, pred_reloaded, decimal=6)
```

#### B3. API Input Validation
```python
def test_api_file_size_limit():
    """Test API rejects files > 10MB"""
    # Create large fake image
    large_data = b"fake_image_data" * (11 * 1024 * 1024)  # 11MB
    
    response = requests.post(
        "http://localhost:8000/predict",
        files={'front_image': large_data, 'back_image': large_data}
    )
    
    assert response.status_code == 400
    assert "too large" in response.json()["detail"].lower()
```

### C. Integration / End-to-End Tests

#### C1. Happy Path Test
```python
def test_e2e_happy_path():
    """Test complete prediction workflow"""
    files = {
        'front_image': open('tests/fixtures/phra_somdej_front.jpg', 'rb'),
        'back_image': open('tests/fixtures/phra_somdej_back.jpg', 'rb')
    }
    
    response = requests.post('http://localhost:8000/predict', files=files, timeout=10)
    
    assert response.status_code == 200
    
    result = response.json()
    assert result['status'] == 'success'
    assert result['is_supported'] is True
    assert result['predicted_class'] in ['phra_nang_phya', 'phra_rod', 'phra_somdej']
    assert result['confidence'] >= 0.7
    assert 'model_version' in result
    assert result['model_version'] == '4.0.0'
```

#### C2. OOD Detection Test
```python
def test_ood_detection():
    """Test out-of-domain detection works"""
    files = {
        'front_image': open('tests/fixtures/coin.jpg', 'rb'),  # Not an amulet
        'back_image': open('tests/fixtures/coin_back.jpg', 'rb')
    }
    
    response = requests.post('http://localhost:8000/predict', files=files)
    
    assert response.status_code == 200
    result = response.json()
    
    # Should be rejected as OOD
    assert result['is_supported'] is False
    assert 'out-of-domain' in result['reason'].lower()
```

#### C3. Model Info Endpoint
```python
def test_model_info_endpoint():
    """Test /model/info returns correct information"""
    response = requests.get('http://localhost:8000/model/info')
    
    assert response.status_code == 200
    info = response.json()
    
    assert info['model_version'] == '4.0.0'
    assert info['supported_classes'] == ['phra_nang_phya', 'phra_rod', 'phra_somdej']
    assert 'dual_view_processing' in info['features']
    assert info['features']['dual_view_processing'] is True
```

### D. Model Evaluation Tests (Offline)

#### D1. K-Fold Cross Validation
```bash
python eval/run_cv_evaluation.py --model-config unified_config.json --k 5 --output eval/cv_results.json
```

**เกณฑ์ผ่าน**: 
- mean_f1 ≥ 0.75 
- std_f1 < 0.05
- No class with F1 < 0.70

#### D2. Overfit Detection
```python
def test_overfit_detection():
    """Check for overfitting by comparing train vs validation metrics"""
    
    # Load model and compute metrics on train set
    train_f1 = evaluate_on_dataset("dataset/train")
    val_f1 = evaluate_on_dataset("dataset/validation")
    
    overfit_gap = train_f1 - val_f1
    
    assert overfit_gap < 0.15, f"Potential overfitting: gap={overfit_gap:.3f}"
    
    # Current state warning (ปัจจุบันจะ fail เพราะ perfect metrics)
    if train_f1 >= 0.99 and val_f1 >= 0.99:
        warnings.warn("Perfect metrics suggest possible data leakage or overfitting")
```

#### D3. Calibration Test
```bash
python eval/calibration_analysis.py --predictions preds.npy --labels labels.npy --output calibration_report.json
```

**เกณฑ์ผ่าน**: ECE ≤ 0.08 (ผ่อนปรนเนื่องจาก dataset เล็ก)

#### D4. PCA Variance Check
```python
def test_pca_variance_retention():
    """Test PCA retains sufficient variance"""
    pca = joblib.load('trained_model/pca.joblib')
    
    explained_variance_ratio = pca.explained_variance_ratio_.sum()
    
    assert explained_variance_ratio >= 0.80, f"PCA retains only {explained_variance_ratio:.2%} variance"
    print(f"PCA with 30 components retains {explained_variance_ratio:.2%} of variance")
```

### E. OOD & Out-of-Scope Tests

#### E1. OOD Dataset Evaluation
```bash
python eval/ood_evaluation.py \
  --in_dist_embeddings embeddings/amulets.npy \
  --ood_embeddings embeddings/coins_portraits.npy \
  --output ood_results.json
```

**เกณฑ์ผ่าน**:
- AUROC ≥ 0.90
- OOD recall ≥ 0.90 (ที่ FAR = 5%)

#### E2. Ensemble Voting Consistency
```python
def test_ood_ensemble_consistency():
    """Test OOD ensemble components agree on clear cases"""
    
    # Test with clearly OOD sample
    ood_features = generate_ood_features()  # Synthetic outlier
    
    classifier = EnhancedProductionClassifier()
    classifier.load_model("trained_model")
    
    is_outlier, confidence, reason = classifier.ood_detector.is_outlier(ood_features)
    
    assert is_outlier is True, "Clear OOD case should be detected"
    assert confidence > 0.7, f"OOD confidence too low: {confidence}"
    assert "isolation forest" in reason.lower() or "svm" in reason.lower()
```

### F. Performance & Load Tests

#### F1. Single Request Latency
```bash
# Using k6 for load testing
k6 run -e HOST=http://localhost:8000 tests/load/single_request_latency.js
```

**สคริปต์ตัวอย่าง** (`tests/load/single_request_latency.js`):
```javascript
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  vus: 1,
  duration: '30s',
};

export default function () {
  let formData = {
    front_image: http.file(open('tests/fixtures/sample_front.jpg', 'b')),
    back_image: http.file(open('tests/fixtures/sample_back.jpg', 'b')),
  };
  
  let response = http.post(`${__ENV.HOST}/predict`, formData);
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 2s': (r) => r.timings.duration < 2000,
    'is_supported returned': (r) => JSON.parse(r.body).hasOwnProperty('is_supported'),
  });
}
```

**เกณฑ์ผ่าน**: p95 < 2.0s, p99 < 3.0s

#### F2. Concurrent Load Test
```bash
k6 run -e HOST=http://localhost:8000 tests/load/concurrent_load.js
```

**เกณฑ์ผ่าน**: 
- Error rate < 0.5% ที่ 20 concurrent users
- p95 latency < 3s under load

#### F3. Memory Test
```python
def test_memory_usage():
    """Test memory usage stays within limits"""
    import psutil
    import requests
    
    # Get baseline memory
    process = psutil.Process()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Make 100 requests
    for i in range(100):
        files = {
            'front_image': open('tests/fixtures/sample_front.jpg', 'rb'),
            'back_image': open('tests/fixtures/sample_back.jpg', 'rb')
        }
        requests.post('http://localhost:8000/predict', files=files)
    
    # Check final memory
    final_memory = process.memory_info().rss / 1024 / 1024
    
    assert final_memory < 500, f"Memory usage {final_memory:.1f}MB exceeds 500MB limit"
    
    memory_growth = final_memory - baseline_memory
    assert memory_growth < 100, f"Memory growth {memory_growth:.1f}MB suggests memory leak"
```

#### F4. Cold Start Test
```python
def test_cold_start_time():
    """Test model loading time from cold start"""
    import time
    
    start_time = time.time()
    
    classifier = EnhancedProductionClassifier()
    classifier.load_model("trained_model")
    
    load_time = time.time() - start_time
    
    assert load_time < 10, f"Cold start time {load_time:.2f}s exceeds 10s limit"
    print(f"Model loaded in {load_time:.2f}s")
```

### G. Reliability & Resilience Tests

#### G1. Missing Model Files
```python
def test_missing_model_graceful_failure():
    """Test API handles missing model files gracefully"""
    # Temporarily rename model file
    os.rename("trained_model/classifier.joblib", "trained_model/classifier.joblib.bak")
    
    try:
        response = requests.get("http://localhost:8000/health")
        assert response.status_code == 503
        assert "model not loaded" in response.json()["model_status"]["error"].lower()
    finally:
        # Restore model file
        os.rename("trained_model/classifier.joblib.bak", "trained_model/classifier.joblib")
```

#### G2. Malformed Input Handling
```python
def test_malformed_input_handling():
    """Test API handles malformed inputs gracefully"""
    # Non-image file
    files = {
        'front_image': ('test.txt', b'not an image', 'image/jpeg'),
        'back_image': ('test.txt', b'not an image', 'image/jpeg')
    }
    
    response = requests.post('http://localhost:8000/predict', files=files)
    
    assert response.status_code == 400
    assert "invalid image" in response.json()["detail"].lower()
```

### H. Security Tests

#### H1. Input Validation & Malicious Payloads
```python
def test_malicious_file_upload():
    """Test protection against malicious file uploads"""
    # Fake image header with malicious payload
    malicious_content = b'\xFF\xD8\xFF\xE0' + b'<script>alert("xss")</script>' * 1000
    
    files = {
        'front_image': ('malicious.jpg', malicious_content, 'image/jpeg'),
        'back_image': ('malicious.jpg', malicious_content, 'image/jpeg')
    }
    
    response = requests.post('http://localhost:8000/predict', files=files)
    
    # Should return 400 (bad request) not 500 (server error)
    assert response.status_code == 400
```

#### H2. Rate Limiting Test
```python
def test_rate_limiting():
    """Test rate limiting prevents abuse"""
    # Current limit: 100 requests/minute
    
    responses = []
    for i in range(105):  # Exceed limit
        files = {
            'front_image': open('tests/fixtures/sample_front.jpg', 'rb'),
            'back_image': open('tests/fixtures/sample_back.jpg', 'rb')
        }
        response = requests.post('http://localhost:8000/predict', files=files)
        responses.append(response.status_code)
    
    # Should see some 429 (Too Many Requests) responses
    rate_limited_count = responses.count(429)
    assert rate_limited_count > 0, "Rate limiting not working"
```

### I. Logging, Monitoring & Observability

#### I1. Structured Logging Test
```python
def test_structured_logging():
    """Test that requests generate structured logs"""
    import tempfile
    import json
    
    # Make a request
    files = {
        'front_image': open('tests/fixtures/sample_front.jpg', 'rb'),
        'back_image': open('tests/fixtures/sample_back.jpg', 'rb')
    }
    response = requests.post('http://localhost:8000/predict', files=files)
    
    # Check if logs directory exists and has recent entries
    assert os.path.exists("logs"), "Logs directory should exist"
    
    # Check log file for structured content
    log_files = glob.glob("logs/*.log")
    assert len(log_files) > 0, "No log files found"
    
    # Read latest log entries
    with open(log_files[0], 'r') as f:
        lines = f.readlines()
    
    # Look for request ID and model version in logs
    recent_logs = lines[-10:]  # Last 10 lines
    
    found_request_id = any('request_id' in line for line in recent_logs)
    found_model_version = any('model_version' in line for line in recent_logs)
    
    assert found_request_id, "request_id not found in logs"
    assert found_model_version, "model_version not found in logs"
```

#### I2. Metrics Endpoint Test
```python
def test_prometheus_metrics():
    """Test /metrics endpoint exposes required metrics"""
    response = requests.get('http://localhost:8000/metrics')
    
    assert response.status_code == 200
    
    metrics_text = response.text
    
    # Check for required metrics
    required_metrics = [
        'requests_total',
        'latency_percentiles',
        'memory_usage_mb',
        'error_rate'
    ]
    
    for metric in required_metrics:
        assert metric in metrics_text, f"Missing metric: {metric}"
```

### J. UX / Frontend Tests

#### J1. Responsive UI Test
```python
def test_responsive_ui():
    """Test UI works on different viewport sizes"""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get("http://localhost:8501")
        
        # Test different viewport sizes
        viewports = [(320, 568), (375, 667), (412, 732), (768, 1024)]
        
        for width, height in viewports:
            driver.set_window_size(width, height)
            
            # Check if upload button is visible and clickable
            upload_button = driver.find_element(By.CSS_SELECTOR, "[data-testid='stFileUploader']")
            assert upload_button.is_displayed(), f"Upload not visible at {width}x{height}"
            
    finally:
        driver.quit()
```

#### J2. User Flow Test
```python
def test_complete_user_flow():
    """Test complete user interaction flow"""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    
    driver = webdriver.Chrome()
    
    try:
        driver.get("http://localhost:8501")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stFileUploader']"))
        )
        
        # Upload file (simulate)
        file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
        file_input.send_keys(os.path.abspath("tests/fixtures/sample_front.jpg"))
        
        # Check for predict button and click
        predict_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'จำแนก')]"))
        )
        predict_button.click()
        
        # Wait for results
        result_element = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'ผลการทำนาย')]"))
        )
        
        assert result_element.is_displayed(), "Results not displayed"
        
    finally:
        driver.quit()
```

---

## 🔧 Implementation Scripts

### 1. Full Offline Evaluation
```bash
#!/bin/bash
# eval/run_full_evaluation.sh

echo "🧪 Running Full Model Evaluation..."

python eval/run_cv_evaluation.py \
  --model trained_model/classifier.joblib \
  --pca trained_model/pca.joblib \
  --scaler trained_model/scaler.joblib \
  --test_dir dataset/test/ \
  --output eval/full_evaluation_report.json

python eval/calibration_analysis.py \
  --model trained_model/ \
  --output eval/calibration_report.json

python eval/ood_evaluation.py \
  --model trained_model/ \
  --ood_dir tests/fixtures/ood_samples/ \
  --output eval/ood_report.json

echo "✅ Evaluation complete. Check eval/ directory for reports."
```

### 2. E2E Test Suite
```python
# tests/e2e/test_complete_workflow.py

import pytest
import requests
import os
import time

class TestCompleteWorkflow:
    
    @pytest.fixture(scope="class")
    def api_base_url(self):
        return "http://localhost:8000"
    
    def test_api_health_check(self, api_base_url):
        """Test API is healthy before running other tests"""
        response = requests.get(f"{api_base_url}/health", timeout=10)
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] in ["healthy", "degraded"]
    
    def test_model_info(self, api_base_url):
        """Test model info endpoint"""
        response = requests.get(f"{api_base_url}/model/info")
        assert response.status_code == 200
        
        info = response.json()
        assert info["model_version"] == "4.0.0"
        assert len(info["supported_classes"]) == 3
    
    @pytest.mark.parametrize("class_name", ["phra_somdej", "phra_rod", "phra_nang_phya"])
    def test_prediction_per_class(self, api_base_url, class_name):
        """Test prediction works for each class"""
        front_path = f"tests/fixtures/{class_name}_front.jpg"
        back_path = f"tests/fixtures/{class_name}_back.jpg"
        
        if not os.path.exists(front_path) or not os.path.exists(back_path):
            pytest.skip(f"Test fixtures not found for {class_name}")
        
        files = {
            'front_image': open(front_path, 'rb'),
            'back_image': open(back_path, 'rb')
        }
        
        start_time = time.time()
        response = requests.post(f"{api_base_url}/predict", files=files, timeout=15)
        latency = time.time() - start_time
        
        assert response.status_code == 200
        assert latency < 2.0, f"Latency {latency:.2f}s exceeds 2s limit"
        
        result = response.json()
        assert result["status"] == "success"
        assert result["is_supported"] is True
        assert result["predicted_class"] in ["phra_somdej", "phra_rod", "phra_nang_phya"]
        assert result["confidence"] >= 0.7
    
    def test_ood_detection(self, api_base_url):
        """Test OOD detection with non-amulet images"""
        ood_files = {
            'front_image': open('tests/fixtures/coin.jpg', 'rb'),
            'back_image': open('tests/fixtures/portrait.jpg', 'rb')
        }
        
        response = requests.post(f"{api_base_url}/predict", files=ood_files)
        
        assert response.status_code == 200
        result = response.json()
        
        # Should be rejected or have very low confidence
        assert result["is_supported"] is False or result["confidence"] < 0.3
```

### 3. Performance Test Script
```python
# tests/performance/load_test.py

import asyncio
import aiohttp
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

class PerformanceTest:
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def single_request(self, session):
        """Make a single prediction request"""
        
        with open('tests/fixtures/sample_front.jpg', 'rb') as f1, \
             open('tests/fixtures/sample_back.jpg', 'rb') as f2:
            
            data = aiohttp.FormData()
            data.add_field('front_image', f1, filename='front.jpg')
            data.add_field('back_image', f2, filename='back.jpg')
            
            start_time = time.time()
            
            try:
                async with session.post(f"{self.base_url}/predict", data=data) as response:
                    await response.json()
                    latency = time.time() - start_time
                    success = response.status == 200
                    
                    self.results.append({
                        'latency': latency,
                        'success': success,
                        'status_code': response.status
                    })
                    
            except Exception as e:
                latency = time.time() - start_time
                self.results.append({
                    'latency': latency,
                    'success': False,
                    'error': str(e)
                })
    
    async def run_concurrent_test(self, concurrent_users=10, requests_per_user=10):
        """Run concurrent load test"""
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for _ in range(concurrent_users):
                for _ in range(requests_per_user):
                    tasks.append(self.single_request(session))
            
            await asyncio.gather(*tasks)
    
    def analyze_results(self):
        """Analyze performance test results"""
        
        if not self.results:
            return {"error": "No results to analyze"}
        
        latencies = [r['latency'] for r in self.results if r['success']]
        success_count = sum(1 for r in self.results if r['success'])
        error_rate = 1 - (success_count / len(self.results))
        
        if not latencies:
            return {"error": "No successful requests"}
        
        return {
            "total_requests": len(self.results),
            "successful_requests": success_count,
            "error_rate": error_rate,
            "latency_stats": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p95": statistics.quantiles(latencies, n=20)[18],  # 95th percentile
                "p99": statistics.quantiles(latencies, n=100)[98], # 99th percentile
                "min": min(latencies),
                "max": max(latencies)
            }
        }

# Usage
if __name__ == "__main__":
    test = PerformanceTest()
    
    print("🚀 Running performance test...")
    asyncio.run(test.run_concurrent_test(concurrent_users=20, requests_per_user=5))
    
    results = test.analyze_results()
    print(f"📊 Results: {results}")
    
    # Assert performance criteria
    assert results["error_rate"] < 0.005, f"Error rate {results['error_rate']:.3f} exceeds 0.5%"
    assert results["latency_stats"]["p95"] < 2.0, f"P95 latency {results['latency_stats']['p95']:.2f}s exceeds 2s"
    
    print("✅ Performance test passed!")
```

---

## 🚨 Missing Infrastructure & Quick Fixes

### 1. Create Missing Directories & Files

```bash
# สร้าง logs directory
mkdir -p logs
touch logs/.gitkeep

# สร้างโฟลเดอร์ tests
mkdir -p tests/{data,e2e,performance,fixtures,load}
mkdir -p eval

# สร้าง Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements_production.txt .
RUN pip install -r requirements_production.txt

COPY . .

RUN mkdir -p logs

EXPOSE 8000 8501

CMD ["python", "backend/api/main_api.py"]
EOF

# สร้าง docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  amulet-ai-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./trained_model:/app/trained_model
    environment:
      - PYTHONPATH=/app
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  amulet-ai-frontend:
    build: .
    command: streamlit run frontend/production_app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    depends_on:
      - amulet-ai-api
    environment:
      - API_BASE_URL=http://amulet-ai-api:8000
EOF
```

### 2. Test Data Fixtures Setup

```bash
# สร้าง test fixtures
mkdir -p tests/fixtures/ood_samples

# คัดลอกตัวอย่างจาก dataset มาเป็น fixtures
cp dataset/test/phra_somdej/phra_somdej_001.jpg tests/fixtures/phra_somdej_front.jpg
cp dataset/test/phra_somdej/phra_somdej_002.jpg tests/fixtures/phra_somdej_back.jpg

cp dataset/test/phra_rod/phra_rod_001.jpg tests/fixtures/phra_rod_front.jpg
cp dataset/test/phra_rod/phra_rod_002.jpg tests/fixtures/phra_rod_back.jpg

cp dataset/test/phra_nang_phya/phra_nang_phya_001.jpg tests/fixtures/phra_nang_phya_front.jpg
cp dataset/test/phra_nang_phya/phra_nang_phya_002.jpg tests/fixtures/phra_nang_phya_back.jpg

# สำหรับ OOD testing (ใช้รูปอื่นที่ไม่ใช่พระเครื่อง)
echo "⚠️ Need to add coin.jpg, portrait.jpg to tests/fixtures/ for OOD testing"
```

### 3. Pytest Configuration

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers = 
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
    e2e: marks tests as end-to-end tests
```

---

## 🎯 CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/test.yml
name: Comprehensive Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements_production.txt
        pip install pytest pytest-asyncio aiohttp selenium
    
    - name: Create logs directory
      run: mkdir -p logs
    
    - name: Run unit tests
      run: pytest tests/ -m "not (integration or performance or e2e)" -v
    
    - name: Build Docker image
      run: docker build -t amulet-ai:test .
    
    - name: Start services for integration tests
      run: |
        docker-compose up -d
        sleep 30  # Wait for services to start
    
    - name: Run integration tests
      run: pytest tests/ -m integration -v
    
    - name: Run E2E tests
      run: pytest tests/e2e/ -v
    
    - name: Run performance tests
      run: pytest tests/performance/ -v
    
    - name: Model evaluation gate
      run: |
        python eval/run_quick_eval.py --threshold 0.75
        
    - name: Check metrics regression
      run: |
        python ci/check_metrics_regression.py \
          --current eval/current_metrics.json \
          --baseline eval/baseline_metrics.json \
          --max_f1_drop 0.02
    
    - name: Cleanup
      run: docker-compose down
```

---

## 📊 Quick Acceptance Checklist

**ใช้สำหรับการตรวจสอบก่อน release:**

### Data & Model Quality
- [ ] ✅ Data readable, duplicates < 5%
- [ ] ✅ Train/test split by item confirmed (no leakage)
- [ ] ✅ CV mean_f1 ≥ 0.75 and std < 0.05
- [ ] ⚠️ Overfit check (current: likely overfitting due to perfect metrics)
- [ ] ✅ OOD AUROC ≥ 0.90 & OOD recall ≥ 0.90

### Performance & System
- [ ] ✅ p95 latency < 2s and memory < 500MB
- [ ] ✅ API contract validated by E2E tests
- [ ] ✅ Model versioning present (/model/info returns version)
- [ ] ⚠️ Logging present & structured (logs directory needs creation)

### Infrastructure
- [ ] ❌ Dockerfile + docker-compose exists and builds image
- [ ] ❌ Prometheus metrics exposed (partially implemented)
- [ ] ✅ Error handling graceful (API returns proper HTTP codes)
- [ ] ✅ Security basics (file size limits, input validation)

### CI/CD
- [ ] ❌ CI pipeline enforces model & code gates (needs setup)
- [ ] ❌ Automated testing on PR (needs GitHub Actions)
- [ ] ✅ Test fixtures available
- [ ] ⚠️ Performance regression detection (needs baseline)

---

## 🔄 Next Steps & Recommendations

### Immediate Actions (Within 1 week)
1. **สร้าง missing infrastructure files** (Dockerfile, logs directory)
2. **สร้าง test fixtures** สำหรับแต่ละคลาสและ OOD samples
3. **เซ็ตอัพ basic CI pipeline** ด้วย GitHub Actions
4. **รัน comprehensive evaluation** เพื่อหา baseline metrics

### Short-term (Within 1 month)
1. **เพิ่มข้อมูลฝึก** เพื่อลด overfitting (target: 100+ รูป/คลาส)
2. **เซ็ตอัพ monitoring** ด้วย Prometheus + Grafana
3. **Performance optimization** (target: p95 < 1s)
4. **Security hardening** (authentication, rate limiting)

### Long-term (Within 3 months)
1. **Production deployment** พร้อม auto-scaling
2. **A/B testing framework** สำหรับ model updates
3. **User feedback loop** integration
4. **Advanced monitoring** (drift detection, model performance tracking)

---

**⚡ การใช้งาน**: บันทึกไฟล์นี้เป็น `COMPREHENSIVE_TEST_CRITERIA.md` และใช้เป็น reference สำหรับการพัฒนา test suite และ CI/CD pipeline

**🎯 เป้าหมาย**: ระบบที่ผ่านเกณฑ์ทั้งหมดนี้จะพร้อมสำหรับ production deployment และมี confidence สูงในการทำงานภายใต้ real-world conditions