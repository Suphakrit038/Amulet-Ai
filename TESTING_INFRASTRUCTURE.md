# 🧪 Testing Infrastructure Setup
# รายการไฟล์และโฟลเดอร์ที่จำเป็นสำหรับ comprehensive testing

tests/
├── __init__.py
├── conftest.py                    # pytest configuration
├── data/
│   ├── __init__.py
│   ├── check_data_quality.py      # ✅ Created
│   └── test_data_validation.py
├── unit/
│   ├── __init__.py
│   ├── test_feature_extractor.py
│   ├── test_model_components.py
│   └── test_api_utils.py
├── integration/
│   ├── __init__.py
│   └── test_model_pipeline.py
├── e2e/
│   ├── __init__.py
│   └── test_api_complete.py       # ✅ Created
├── performance/
│   ├── __init__.py
│   └── test_load_performance.py
└── fixtures/
    ├── phra_somdej_front.jpg      # Sample images for testing
    ├── phra_rod_front.jpg
    ├── phra_nang_phya_front.jpg
    ├── coin.jpg                   # OOD sample
    └── ood_samples/

eval/
├── __init__.py
├── run_quick_eval.py              # ✅ Created
├── run_cv_evaluation.py
├── calibration_analysis.py
├── ood_evaluation.py
└── baseline_metrics.json         # Baseline for regression testing

ci/
├── __init__.py
├── check_metrics_regression.py    # ✅ Created
└── generate_baseline.py

.github/
└── workflows/
    └── test.yml                   # ✅ Created

# Additional files needed:
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Local development
├── pytest.ini                     # pytest configuration
├── .dockerignore                  # Docker ignore file
└── requirements_dev.txt           # Development dependencies