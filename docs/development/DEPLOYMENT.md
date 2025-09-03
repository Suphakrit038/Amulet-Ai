# 🚀 Deployment Guide - Amulet-AI

## Overview
This guide provides step-by-step instructions for deploying the Amulet-AI system in different environments.

## 🏠 Local Development

### Prerequisites
```bash
# Python 3.9+
python --version

# Required packages
pip install -r requirements.txt
```

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd Amulet-Ai

# Install dependencies
pip install -r requirements.txt

# Start the system
python scripts/start_system.bat
```

### Individual Components
```bash
# Backend API only
python backend/optimized_api.py

# Frontend UI only  
streamlit run frontend/app_streamlit.py --server.port 8501

# Test API (minimal)
python backend/test_api.py
```

## 🧪 Testing Environment

### Environment Setup
```bash
# Set testing environment
set AMULET_ENV=testing

# Run with testing config
python backend/optimized_api.py
```

### Testing Checklist
- [ ] API health check: `GET /health`
- [ ] Image upload: `POST /predict`
- [ ] System status: `GET /system-status`
- [ ] Performance stats: `GET /stats`
- [ ] Error handling: Invalid file uploads

## 🌐 Production Deployment

### Server Requirements
- **OS**: Windows Server 2019+ or Linux Ubuntu 20.04+
- **Python**: 3.9 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 10GB available space
- **CPU**: 2 cores minimum, 4 cores recommended

### Production Setup

#### 1. Environment Configuration
```bash
# Set production environment
set AMULET_ENV=production

# Disable debugging
# In config.py, set debug=False
```

#### 2. Install Production Dependencies
```bash
# Install with production optimizations
pip install -r requirements.txt --no-dev

# Install additional production tools
pip install gunicorn uvicorn[standard]
```

#### 3. Start Production Server
```bash
# Using Uvicorn with workers
uvicorn backend.optimized_api:app --host 0.0.0.0 --port 8000 --workers 4

# Using Gunicorn (Linux)
gunicorn backend.optimized_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Production Checklist
- [ ] Environment variables set correctly
- [ ] Debug mode disabled
- [ ] Logging configured for production
- [ ] File size limits enforced
- [ ] Error handling comprehensive
- [ ] Performance monitoring enabled

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV AMULET_ENV=production
ENV PYTHONPATH=/app

EXPOSE 8000
EXPOSE 8501

CMD ["python", "scripts/start_system.py"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  amulet-api:
    build: .
    ports:
      - "8000:8000"
      - "8501:8501"
    environment:
      - AMULET_ENV=production
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - amulet-api
    restart: unless-stopped
```

### Build and Run
```bash
# Build image
docker build -t amulet-ai:latest .

# Run container
docker run -d --name amulet-ai -p 8000:8000 -p 8501:8501 amulet-ai:latest

# Using Docker Compose
docker-compose up -d
```

## ☁️ Cloud Deployment

### AWS EC2

#### 1. Launch Instance
- **Instance Type**: t3.medium or larger
- **OS**: Amazon Linux 2 or Ubuntu 20.04
- **Storage**: 20GB EBS volume
- **Security Groups**: Allow ports 22, 80, 443, 8000, 8501

#### 2. Setup Script
```bash
#!/bin/bash
# Install Python and dependencies
sudo yum update -y
sudo amazon-linux-extras install python3.8 -y
sudo pip3 install --upgrade pip

# Clone and setup application
git clone <repository-url> /home/ec2-user/amulet-ai
cd /home/ec2-user/amulet-ai
pip3 install -r requirements.txt

# Create systemd service
sudo cp deployment/amulet-ai.service /etc/systemd/system/
sudo systemctl enable amulet-ai
sudo systemctl start amulet-ai
```

#### 3. Systemd Service
```ini
[Unit]
Description=Amulet-AI API Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/amulet-ai
Environment=AMULET_ENV=production
ExecStart=/usr/bin/python3 backend/optimized_api.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Google Cloud Platform

#### 1. App Engine Deployment
```yaml
# app.yaml
runtime: python39
service: amulet-ai

env_variables:
  AMULET_ENV: production

handlers:
- url: /.*
  script: auto

automatic_scaling:
  min_instances: 1
  max_instances: 10
```

#### 2. Deploy Command
```bash
# Initialize gcloud
gcloud init

# Deploy application
gcloud app deploy app.yaml

# View logs
gcloud app logs tail -s amulet-ai
```

### Microsoft Azure

#### 1. Web App Deployment
```bash
# Create resource group
az group create --name amulet-ai-rg --location "Southeast Asia"

# Create app service plan
az appservice plan create --name amulet-ai-plan --resource-group amulet-ai-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group amulet-ai-rg --plan amulet-ai-plan --name amulet-ai-app --runtime "PYTHON|3.9"

# Deploy code
az webapp deployment source config-zip --resource-group amulet-ai-rg --name amulet-ai-app --src deployment.zip
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Production settings
AMULET_ENV=production
DEBUG=false
LOG_LEVEL=WARNING
MAX_WORKERS=4
```

### Configuration Files

#### `production.env`
```
AMULET_ENV=production
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
CACHE_TTL=3600
MAX_CONCURRENT_REQUESTS=50
```

#### Load Configuration
```python
# In production startup script
import os
from dotenv import load_dotenv

load_dotenv('production.env')
```

## 📊 Monitoring and Logging

### Application Logging
```python
# Configure logging for production
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/amulet-ai.log'),
        logging.StreamHandler()
    ]
)
```

### Health Monitoring
```bash
# Add health check script
#!/bin/bash
# health_check.sh
curl -f http://localhost:8000/health || exit 1
```

### Performance Monitoring
- Use `/stats` endpoint for application metrics
- Monitor system resources (CPU, memory, disk)
- Set up alerts for high error rates
- Track response times and throughput

## 🔐 Security Hardening

### Production Security
```python
# In production config
security_settings = {
    "cors_origins": ["https://yourdomain.com"],
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "rate_limiting": True,
    "https_only": True
}
```

### Firewall Rules
```bash
# Allow only necessary ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

### SSL/TLS Configuration
```nginx
# nginx.conf for SSL
server {
    listen 443 ssl;
    server_name yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🚨 Troubleshooting

### Common Deployment Issues

#### Memory Issues
```bash
# Monitor memory usage
free -m
htop

# Optimize Python memory
export PYTHONOPTIMIZE=1
```

#### Port Conflicts
```bash
# Check port usage
netstat -tlnp | grep :8000

# Kill conflicting process
sudo fuser -k 8000/tcp
```

#### Permission Issues
```bash
# Fix file permissions
chmod +x scripts/*.bat
chown -R www-data:www-data /app
```

## 📋 Deployment Checklist

### Pre-deployment
- [ ] Code tested in staging environment
- [ ] Dependencies updated and verified
- [ ] Configuration files prepared
- [ ] Backup existing deployment (if applicable)
- [ ] Database migrations completed (if applicable)

### During Deployment
- [ ] Server resources available
- [ ] Environment variables set
- [ ] Application starts successfully
- [ ] Health checks passing
- [ ] API endpoints responding

### Post-deployment
- [ ] Monitor logs for errors
- [ ] Verify all endpoints working
- [ ] Performance metrics normal
- [ ] Error rates acceptable
- [ ] User acceptance testing completed

---

**Deployment Version**: 1.0.0  
**Last Updated**: August 2025  
**Support**: Create issue in repository for deployment assistance
