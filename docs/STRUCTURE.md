# 📁 Amulet-AI Project Structure

## 🏗️ Enhanced Modular Directory Structure

```
Amulet-AI/
├── 📂 api/                          # API Backend
│   ├── __init__.py
│   └── main_api.py                  # FastAPI application with enhanced features
│
├── 📂 ai_models/                    # AI Model Components
│   ├── __init__.py
│   ├── compatibility_loader.py
│   ├── enhanced_production_system.py
│   ├── labels.json
│   └── twobranch/                   # Two-branch model implementation
│       ├── config.py
│       ├── dataset.py
│       ├── integration.py           # (renamed from enhanced_integration.py)
│       ├── cnn_multilayer.py        # (renamed from enhanced_multilayer_cnn.py)
│       ├── preprocessing.py         # (unified, removed duplicate preprocess.py)
│       ├── training.py              # (renamed from enhanced_training.py)
│       ├── inference.py
│       ├── model.py
│       └── data_generator.py        # (renamed from realistic_amulet_generator.py)
│
├── 📂 core/                         # Core System Components
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   ├── error_handling.py            # Error handling & recovery
│   ├── memory_management.py         # Memory optimization
│   ├── performance.py               # Performance optimization
│   ├── security.py                  # Security features
│   └── thread_safety.py             # Thread safety components
│
├── 📂 frontend/                     # Frontend Components (Modular)
│   ├── __init__.py
│   ├── main_streamlit_app.py        # Main application entry point
│   ├── pages/                       # Modular pages for future expansion
│   │   └── (future page modules)
│   ├── components/                  # Reusable UI components
│   │   ├── __init__.py
│   │   ├── mode_selector.py         # Analysis mode selection
│   │   ├── file_uploader.py         # File upload handling
│   │   ├── image_display.py         # Image display with quality assessment
│   │   └── analysis_results.py      # Results display with enhanced visualization
│   └── utils/                       # Frontend utilities
│       ├── __init__.py
│       ├── image_processor.py       # Image processing and enhancement
│       ├── file_validator.py        # File validation and constraints
│       ├── analytics_manager.py     # User analytics and tracking
│       └── ui_helpers.py            # UI helper functions
│
├── 📂 scripts/                      # Modular Utility Scripts
│   ├── production_runner.py         # Production management
│   ├── automation/                  # Development automation
│   │   └── dev_cli.py              # Development CLI with Typer
│   ├── monitoring/                  # Performance and health monitoring
│   │   └── (monitoring scripts)
│   └── deployment/                  # Deployment automation
│       └── (deployment scripts)
│
├── 📂 deployment/                   # Multi-Environment Deployment
│   ├── docker-compose.dev.yml      # Development environment
│   ├── docker-compose.staging.yml  # Staging environment
│   ├── docker-compose.prod.yml     # Production environment
│   ├── Dockerfile.api              # API container definition
│   ├── Dockerfile.frontend         # Frontend container definition
│   ├── nginx/                      # Load balancer configurations
│   │   ├── development.conf
│   │   ├── staging.conf
│   │   └── production.conf
│   └── monitoring/                 # Monitoring stack configurations
│       ├── prometheus.yml
│       ├── grafana/
│       └── alerting/
│
├── 📂 tests/                        # Testing Framework
│   ├── test_enhanced_features.py    # Comprehensive tests
│   ├── test_components.py           # Component-specific tests
│   ├── test_api.py                  # API endpoint tests
│   └── test_integration.py          # Integration tests
│
├── 📂 trained_model/                # Model Assets
│   ├── classifier.joblib
│   ├── deployment_info.json
│   ├── label_encoder.joblib
│   ├── model_info.json
│   ├── ood_detector.joblib
│   ├── pca.joblib
│   └── scaler.joblib
│
├── 📂 docs/                         # Comprehensive Documentation
│   ├── api_spec.md                 # Complete OpenAPI/Swagger specification
│   ├── ARCHITECTURE_WORKFLOW.md     # System architecture and workflow
│   ├── QUICK_START.md               # Quick start guide
│   ├── diagrams/                    # System diagrams
│   │   ├── system_architecture.md   # Mermaid diagrams for system architecture
│   │   ├── data_flow.md            # Data flow diagrams
│   │   └── deployment.md           # Deployment architecture
│   ├── user_guides/                # User documentation
│   │   ├── installation.md
│   │   ├── usage.md
│   │   └── troubleshooting.md
│   └── developer_guides/           # Development documentation
│       ├── contributing.md
│       ├── testing.md
│       └── deployment.md
│
├── 📋 requirements.txt              # Python dependencies
├── 🔧 Makefile                     # Development task automation
├── 🔧 config_template.env          # Configuration template
├── 🔧 .env.example                 # Environment example
├── 🚫 .gitignore                   # Git ignore rules
├── 📄 README.md                    # Project overview and setup
└── 📄 STRUCTURE.md                 # This file
```

## 🎯 Key Architectural Improvements

### 🗂️ **Modular Frontend Architecture**
- **Components**: Reusable UI components with clear responsibilities
- **Utils**: Specialized utilities for image processing, validation, analytics
- **Pages**: Modular page structure for future multi-page applications
- **Separation of Concerns**: Clear separation between presentation, logic, and utilities

### 🚀 **Development Automation**
- **Makefile**: Comprehensive task automation for all environments
- **CLI Tools**: Typer-based development CLI for common tasks
- **Multi-Environment**: Separate configurations for dev/staging/production
- **Container Orchestration**: Docker Compose for each environment

### 📊 **Enhanced Documentation**
- **API Specification**: Complete OpenAPI/Swagger documentation
- **System Diagrams**: Mermaid-based architecture diagrams
- **Multi-Level Docs**: User guides, developer guides, and technical specs
- **Visual Architecture**: Clear system overview with data flow

### 🔧 **AI Model Organization**
- **Shorter Names**: More concise and readable filenames
- **Unified Processing**: Eliminated duplicate preprocessing files
- **Clear Responsibilities**: Each module has a specific, well-defined purpose

## 🛠️ **Development Commands (Makefile)**

### Quick Start
```bash
make install          # Install dependencies
make dev              # Start development server
make api              # Start API server only
make dev-full         # Start both API and Frontend
```

### Testing & Quality
```bash
make test             # Run unit tests
make test-coverage    # Run tests with coverage
make lint             # Code quality checks
make format           # Format code
```

### AI Development
```bash
make train            # Train AI models
make inference        # Test inference
make model-info       # Show model information
```

### Deployment
```bash
make docker-build     # Build Docker images
make deploy-dev       # Deploy to development
make deploy-staging   # Deploy to staging
make deploy-prod      # Deploy to production
```

### Utilities
```bash
make clean            # Clean temporary files
make docs             # Generate documentation
make status           # Check service status
```

## 🏗️ **Component Architecture**

### Frontend Components
- **ModeSelectorComponent**: Analysis mode selection (Single/Dual)
- **FileUploaderComponent**: File upload with validation
- **ImageDisplayComponent**: Image display with quality assessment
- **AnalysisResultsComponent**: Enhanced result visualization

### Frontend Utils
- **ImagePreprocessor**: Automatic image enhancement and quality assessment
- **FileValidator**: Comprehensive file validation and optimization suggestions
- **AnalyticsManager**: User behavior tracking and session management
- **UIHelpers**: Common UI utilities and formatting

## 🐳 **Multi-Environment Deployment**

### Development Environment
- Basic services (API, Frontend, Redis)
- Hot reload and debugging enabled
- Portainer for container management

### Staging Environment
- Production-like setup with PostgreSQL
- Nginx load balancer
- Prometheus monitoring
- SSL termination

### Production Environment
- High availability with multiple API instances
- PostgreSQL master-slave replication
- Redis cluster
- Full monitoring stack (Prometheus, Grafana, ELK)
- Security hardening (Fail2ban)

## 📈 **Performance & Scalability Benefits**

### Modular Loading
- Import only required components
- Lazy loading for better startup time
- Component-level caching

### Container Orchestration
- Horizontal scaling capability
- Load balancing across instances
- Health checks and auto-restart

### Monitoring & Observability
- Performance metrics collection
- Real-time health monitoring
- Log aggregation and analysis

## 🔐 **Security Enhancements**

- **Environment-based Configuration**: Separate configs per environment
- **Container Security**: Minimal attack surface
- **Network Isolation**: Proper network segmentation
- **Secret Management**: Environment variables for sensitive data

## 🚀 **Quick Development Workflow**

### 1. Setup
```bash
make install-dev      # Install with dev dependencies
make quick-setup      # Complete setup
```

### 2. Development
```bash
make dev-full         # Start full stack
# Code in modular components
make test             # Run tests
```

### 3. Deployment
```bash
make quick-test       # Quick validation
make deploy-dev       # Deploy to development
```

## 🎉 **Migration Benefits**

### Code Organization
- ✅ Modular frontend architecture
- ✅ Cleaner AI model structure  
- ✅ Comprehensive deployment strategy
- ✅ Professional development workflow

### Developer Experience
- ✅ One-command development setup
- ✅ Automated testing and deployment
- ✅ Clear documentation and diagrams
- ✅ Modern development tools

### Production Readiness
- ✅ Multi-environment support
- ✅ Container orchestration
- ✅ Monitoring and alerting
- ✅ High availability architecture

Your Amulet-AI project is now enterprise-ready with modern architecture! 🚀