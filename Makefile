# Amulet-AI Makefile
# ระบบจัดการคำสั่งสำหรับ development และ deployment

.PHONY: help install dev test lint format clean deploy docs

# Default target
help: ## แสดงคำสั่งที่ใช้ได้
	@echo "🔮 Amulet-AI Development Commands"
	@echo "=================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation and Setup
install: ## ติดตั้ง dependencies ทั้งหมด
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Installation complete!"

install-dev: ## ติดตั้ง dev dependencies
	@echo "🔧 Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest black flake8 mypy
	@echo "✅ Dev installation complete!"

# Development
dev: ## เริ่ม development server
	@echo "🚀 Starting Amulet-AI development server..."
	streamlit run frontend/main_streamlit_app.py --server.port 8501

api: ## เริ่ม API server
	@echo "🔌 Starting API server..."
	python -m uvicorn api.main_api:app --reload --port 8000

dev-full: ## เริ่ม full development stack (API + Frontend)
	@echo "🚀 Starting full development stack..."
	@powershell -Command "Start-Process powershell -ArgumentList '-NoExit', '-Command', 'python -m uvicorn api.main_api:app --reload --port 8000'"
	@timeout /t 3 /nobreak > NUL
	@streamlit run frontend/main_streamlit_app.py --server.port 8501

# Testing
test: ## รัน unit tests
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

test-coverage: ## รัน tests พร้อม coverage report
	@echo "📊 Running tests with coverage..."
	python -m pytest tests/ -v --cov=. --cov-report=html

test-api: ## ทดสอบ API endpoints
	@echo "🔌 Testing API endpoints..."
	python -m pytest tests/test_api.py -v

test-frontend: ## ทดสอบ frontend components
	@echo "🖥️ Testing frontend..."
	python -m pytest tests/test_frontend.py -v

# Code Quality
lint: ## ตรวจสอบ code quality
	@echo "🔍 Linting code..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format: ## จัดรูปแบบ code
	@echo "✨ Formatting code..."
	black . --line-length 127
	@echo "✅ Code formatted!"

type-check: ## ตรวจสอบ type hints
	@echo "🔍 Type checking..."
	mypy . --ignore-missing-imports

# AI Models
train: ## เทรน AI model
	@echo "🤖 Training AI model..."
	python ai_models/enhanced_training.py

inference: ## ทดสอบ inference
	@echo "🔮 Running inference test..."
	python ai_models/inference.py

model-info: ## แสดงข้อมูล model
	@echo "📊 Model information..."
	python scripts/show_model_info.py

# Database and Data
prepare-data: ## เตรียมข้อมูลสำหรับ training
	@echo "📊 Preparing training data..."
	python scripts/prepare_data.py

backup-models: ## สำรองข้อมูล models
	@echo "💾 Backing up models..."
	@if not exist "backups" mkdir backups
	xcopy /E /I "trained_model" "backups\trained_model_%date:~6,4%%date:~3,2%%date:~0,2%"

# Documentation
docs: ## สร้าง documentation
	@echo "📚 Generating documentation..."
	python scripts/generate_docs.py

docs-serve: ## เปิด documentation server
	@echo "📖 Serving documentation..."
	@echo "Documentation will be available at http://localhost:8080"
	python -m http.server 8080 --directory docs/

api-docs: ## สร้าง API documentation
	@echo "📋 Generating API docs..."
	python scripts/generate_api_docs.py

# Deployment
deploy-dev: ## Deploy to development environment
	@echo "🚀 Deploying to development..."
	docker-compose -f deployment/docker-compose.dev.yml up -d

deploy-staging: ## Deploy to staging environment
	@echo "🚀 Deploying to staging..."
	docker-compose -f deployment/docker-compose.staging.yml up -d

deploy-prod: ## Deploy to production environment
	@echo "🚀 Deploying to production..."
	docker-compose -f deployment/docker-compose.prod.yml up -d

# Docker
docker-build: ## สร้าง Docker images
	@echo "🐳 Building Docker images..."
	docker-compose -f deployment/docker-compose.yml build

docker-up: ## เริ่ม Docker containers
	@echo "🐳 Starting Docker containers..."
	docker-compose -f deployment/docker-compose.yml up -d

docker-down: ## หยุด Docker containers
	@echo "🐳 Stopping Docker containers..."
	docker-compose -f deployment/docker-compose.yml down

docker-logs: ## ดู Docker logs
	@echo "📋 Showing Docker logs..."
	docker-compose -f deployment/docker-compose.yml logs -f

# Cleanup
clean: ## ล้างไฟล์ temporary
	@echo "🧹 Cleaning temporary files..."
	@if exist "__pycache__" rmdir /s /q __pycache__
	@if exist ".pytest_cache" rmdir /s /q .pytest_cache
	@if exist ".coverage" del .coverage
	@if exist "htmlcov" rmdir /s /q htmlcov
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@echo "✅ Cleanup complete!"

clean-models: ## ล้าง temporary model files
	@echo "🧹 Cleaning model cache..."
	@if exist "ai_models\__pycache__" rmdir /s /q "ai_models\__pycache__"
	@if exist "trained_model\*.tmp" del "trained_model\*.tmp"

# Environment Management
env-info: ## แสดงข้อมูล environment
	@echo "🔍 Environment Information:"
	@echo "=================================="
	@python --version
	@pip --version
	@echo "Streamlit version:"
	@streamlit version
	@echo "=================================="

check-deps: ## ตรวจสอบ dependencies
	@echo "📋 Checking dependencies..."
	pip check

freeze: ## สร้าง requirements.txt จากสภาพแวดล้อมปัจจุบัน
	@echo "❄️ Freezing current environment..."
	pip freeze > requirements-freeze.txt
	@echo "✅ Requirements frozen to requirements-freeze.txt"

# Security
security-check: ## ตรวจสอบ security vulnerabilities
	@echo "🔒 Security check..."
	pip-audit

# Performance
benchmark: ## ทดสอบประสิทธิภาพ
	@echo "⚡ Running performance benchmarks..."
	python scripts/benchmark.py

profile: ## วิเคราะห์ performance profile
	@echo "📊 Profiling application..."
	python scripts/profile_app.py

# Database
db-init: ## เริ่มต้น database
	@echo "🗄️ Initializing database..."
	python scripts/init_db.py

db-migrate: ## รัน database migrations
	@echo "🗄️ Running database migrations..."
	python scripts/migrate_db.py

db-backup: ## สำรองข้อมูล database
	@echo "💾 Backing up database..."
	python scripts/backup_db.py

# Quick Commands
quick-setup: install ## Setup เริ่มต้นแบบรวดเร็ว
	@echo "🚀 Quick setup complete! Run 'make dev' to start."

quick-test: lint test ## ทดสอบแบบรวดเร็ว
	@echo "✅ Quick tests passed!"

quick-deploy: test deploy-dev ## Deploy แบบรวดเร็วหลังทดสอบ
	@echo "🚀 Quick deployment complete!"

# Status
status: ## แสดงสถานะ services
	@echo "📊 Service Status:"
	@echo "=================="
	@powershell -Command "Get-Process | Where-Object {$_.ProcessName -like '*streamlit*' -or $_.ProcessName -like '*uvicorn*' -or $_.ProcessName -like '*python*'} | Select-Object ProcessName, Id, CPU | Format-Table"