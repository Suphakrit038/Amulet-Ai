# ✅ Quick Acceptance Checklist
## สำหรับการตรวจสอบก่อน Production Release

**วันที่**: ___________  
**เวอร์ชัน**: Amulet-AI v4.0  
**ผู้ตรวจสอบ**: ___________

---

## 📊 Data & Model Quality

### Data Quality
- [ ] **Data readable**: ไม่มีไฟล์เสียหาย (0 corrupted files)
  - คำสั่ง: `python tests/data/check_data_quality.py --data-dir dataset`
  - เกณฑ์: ผ่าน = 0 corrupted files

- [ ] **Duplicates check**: Duplicate ratio < 5%
  - เกณฑ์: Exact duplicates < 1%, Perceptual duplicates < 5%

- [ ] **Train/test split integrity**: ไม่มี data leakage
  - เกณฑ์: Zero overlap between train/test/validation splits

- [ ] **Class balance**: เพียงพอสำหรับ training
  - ปัจจุบัน: 35 รูป/คลาส (20 train + 10 test + 5 val)
  - เกณฑ์: ≥ 20 รูป/คลาส (แนะนำ ≥ 50)

### Model Performance
- [ ] **CV mean F1** ≥ 0.75 และ std < 0.05
  - คำสั่ง: `python eval/run_quick_eval.py --threshold 0.75`
  - เกณฑ์ผ่าน: F1 ≥ 0.75, std < 0.05

- [ ] **Balanced accuracy** ≥ 0.80
  - ปัจจุบัน: _____ (กรอก)
  - เกณฑ์: ≥ 0.80

- [ ] **Per-class F1**: ทุกคลาส ≥ 0.70
  - phra_somdej: _____ / 0.70
  - phra_rod: _____ / 0.70  
  - phra_nang_phya: _____ / 0.70

- [ ] **Overfit check**: train-val gap < 0.15
  - เกณฑ์: |train_f1 - val_f1| < 0.15
  - ⚠️ ปัจจุบันอาจมี overfitting (perfect metrics = 1.00)

### OOD Detection
- [ ] **OOD AUROC** ≥ 0.90
  - คำสั่ง: `python eval/ood_evaluation.py`
  - เกณฑ์: AUROC ≥ 0.90

- [ ] **OOD recall** ≥ 0.90 (ที่ FAR = 5%)
  - เกณฑ์: OOD recall ≥ 0.90

---

## ⚡ Performance & System

### Latency & Throughput
- [ ] **P95 latency** < 2.0s (single prediction, CPU)
  - วิธีทดสอบ: ใช้ tests/e2e/test_api_complete.py
  - เกณฑ์: p95 < 2.0s

- [ ] **P99 latency** < 3.0s
  - เกณฑ์: p99 < 3.0s

- [ ] **Concurrent load**: ทนได้ 20 concurrent users
  - Error rate < 0.5% ที่ 20 concurrent requests
  - คำสั่ง: `pytest tests/e2e/test_api_complete.py::TestPerformanceAndLoad::test_concurrent_requests`

### Memory & Resources
- [ ] **Memory footprint** < 500MB (target < 250MB)
  - วิธีตรวจ: `python -c "import psutil; print(f'Memory: {psutil.Process().memory_info().rss/1024/1024:.1f}MB')"`
  - เกณฑ์: < 500MB

- [ ] **Cold start time** < 10s (goal < 5s)
  - วิธีทดสอบ: วัดเวลาจาก start ถึง ready
  - เกณฑ์: < 10s

- [ ] **Memory stability**: ไม่มี memory leak
  - เกณฑ์: Memory growth < 100MB หลังจาก 100 requests

---

## 🔗 API & Integration

### API Contract
- [ ] **API endpoints ทำงานได้**:
  - [ ] GET `/` → ส่งข้อมูล service
  - [ ] GET `/health` → status healthy/degraded
  - [ ] GET `/model/info` → model_version "4.0.0"
  - [ ] GET `/metrics` → prometheus metrics
  - [ ] POST `/predict` → prediction result

- [ ] **E2E happy path**: ทุกคลาสทำนายได้
  - คำสั่ง: `pytest tests/e2e/test_api_complete.py::TestPredictionAPI::test_predict_happy_path`

- [ ] **Error handling graceful**:
  - [ ] File size > 10MB → 400 Bad Request
  - [ ] Invalid image → 400 Bad Request  
  - [ ] Missing model → 503 Service Unavailable
  - [ ] Malformed request → 422 Validation Error

### Model Info Consistency
- [ ] **Model versioning present**: /model/info returns version
  - เกณฑ์: model_version = "4.0.0"

- [ ] **Supported classes correct**: 3 classes returned
  - เกณฑ์: ["phra_nang_phya", "phra_rod", "phra_somdej"]

- [ ] **Feature descriptions**: dual_view_processing = true

---

## 🏗️ Infrastructure

### Containerization
- [ ] **Dockerfile exists และ build ได้**
  - คำสั่ง: `docker build -t amulet-ai:test .`
  - เกณฑ์: Build สำเร็จ

- [ ] **Docker container runs**
  - คำสั่ง: `docker run -d -p 8000:8000 amulet-ai:test`
  - เกณฑ์: Container starts และ health check ผ่าน

- [ ] **docker-compose.yml** exists และใช้งานได้
  - คำสั่ง: `docker-compose up -d`
  - เกณฑ์: Services start successfully

### Logging & Monitoring
- [ ] **Logs directory** exists และ writable
  - เกณฑ์: `logs/` directory พร้อมใช้งาน

- [ ] **Structured logging**: request_id, model_version, latency
  - วิธีตรวจ: ตรวจสอบไฟล์ log หลัง API call
  - เกณฑ์: Log entries มี structured fields

- [ ] **Prometheus metrics** exposed และ scraping ได้
  - คำสั่ง: `curl http://localhost:8000/metrics`
  - เกณฑ์: Metrics endpoint returns data

---

## 🔒 Security & Reliability

### Input Validation
- [ ] **File size limits**: >10MB ถูก reject
  - เกณฑ์: ส่ง large file → 400/413 error

- [ ] **File type validation**: non-image ถูก reject  
  - เกณฑ์: ส่งไฟล์ text → 400 error

- [ ] **Malicious payload protection**: ไม่ crash
  - วิธีทดสอบ: ส่ง malformed data
  - เกณฑ์: Return error gracefully (ไม่ใช่ 500)

### Rate Limiting (if implemented)
- [ ] **Rate limiting active**: 100 requests/minute limit
  - วิธีทดสอบ: ส่ง rapid requests
  - เกณฑ์: บาง requests ได้ 429 status

### Reliability
- [ ] **Missing model handling**: graceful degradation
  - เกณฑ์: API ส่ง 503 พร้อมข้อความอธิบาย

- [ ] **Partial failure resilience**: บางส่วน fail ไม่กระทบทั้งระบบ

---

## 🧪 CI/CD & Testing

### Test Coverage
- [ ] **Unit tests pass**: core logic tested
  - คำสั่ง: `pytest tests/unit/ -v`
  - เกณฑ์: All tests pass

- [ ] **Integration tests pass**: full pipeline works
  - คำสั่ง: `pytest tests/e2e/ -v`
  - เกณฑ์: Core workflows pass

- [ ] **Performance tests pass**: latency within limits
  - คำสั่ง: `pytest tests/performance/ -v`

### CI Pipeline
- [ ] **GitHub Actions configured**: .github/workflows/test.yml
  - เกณฑ์: Workflow file exists

- [ ] **Quality gates active**: fail on regression
  - เกณฑ์: Pipeline fails เมื่อ metrics drop > threshold

- [ ] **Artifacts collected**: test reports, metrics
  - เกณฑ์: CI produces downloadable reports

### Regression Protection
- [ ] **Baseline metrics saved**: eval/baseline_metrics.json
  - เกณฑ์: Baseline file exists

- [ ] **Regression check works**: detects F1 drops
  - คำสั่ง: `python ci/check_metrics_regression.py --current <current> --baseline <baseline>`

---

## 📋 Manual Verification

### User Experience
- [ ] **Frontend accessible**: http://localhost:8501 loads
- [ ] **Upload flow works**: user can upload and get results
- [ ] **Results display**: clear prediction + confidence
- [ ] **Error messages helpful**: user-friendly error text

### API Documentation
- [ ] **Swagger/OpenAPI**: http://localhost:8000/docs accessible
- [ ] **API examples work**: copy-paste examples function
- [ ] **Response schemas documented**: correct field descriptions

---

## 🎯 Final Checklist Summary

**Critical (Must Pass)**:
- [ ] No corrupted data files
- [ ] Model F1 ≥ 0.75 per class
- [ ] P95 latency < 2s  
- [ ] API endpoints functional
- [ ] Docker builds successfully
- [ ] No data leakage between splits

**Important (Should Pass)**:
- [ ] OOD detection AUROC ≥ 0.90
- [ ] Memory usage < 500MB
- [ ] Structured logging working
- [ ] Error handling graceful
- [ ] CI pipeline configured

**Nice to Have**:
- [ ] Rate limiting implemented
- [ ] Prometheus metrics
- [ ] Performance regression detection
- [ ] Security scan passed

---

## 📊 Sign-off

**Overall Status**: ⬜ PASS / ⬜ CONDITIONAL PASS / ⬜ FAIL

**Critical Issues Found**: _________________

**Conditional Pass Requirements**: _________________

**Signed by**: _________________ **Date**: _________________

**Notes**: 
```
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
```

---

**🔄 Next Steps if FAIL**:
1. Address critical issues first
2. Re-run failed checks  
3. Update this checklist
4. Get re-approval before release

**🚀 Next Steps if PASS**:  
1. Deploy to staging environment
2. Run smoke tests in staging
3. Schedule production deployment
4. Update baseline metrics for future releases