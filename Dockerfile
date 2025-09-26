FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_production.txt .
RUN pip install --no-cache-dir -r requirements_production.txt

# Copy application code
COPY ai_models/ ./ai_models/
COPY backend/ ./backend/
COPY trained_model/ ./trained_model/
COPY labels.json ./
COPY labels_thai.json ./ai_models/

# Create logs directory with proper permissions
RUN mkdir -p logs && chmod 755 logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash amulet
RUN chown -R amulet:amulet /app
USER amulet

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/trained_model
ENV LOG_LEVEL=INFO
ENV PORT=8000

# Start the API server
CMD ["python", "-m", "uvicorn", "backend.api.main_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]