# 1. Use a stable Python base image
FROM python:3.11-slim

# 2. Install system dependencies required for RF math and LightGBM
# libgomp1 is required for LightGBM parallel processing
# g++ and libproj-dev are needed for pycraf and rasterio
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libproj-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the directory where your app will live inside the container
WORKDIR /app

# 4. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt google-cloud-storage

# 5. Copy your core RF design tool and web app files
COPY rf_design_tool/ ./rf_design_tool/
COPY site_deployment_demo/ ./site_deployment_demo/
COPY Model/ ./Model/

# 6. Set environment variables for Cloud Run
ENV PORT 8080
ENV PYTHONUNBUFFERED True

# 7. Launch the app with Gunicorn
# Run from inside site_deployment_demo/ so relative imports (api.site_predictor) work correctly
# --timeout 0 is critical because RF path-loss math can take several seconds
WORKDIR /app/site_deployment_demo
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
