# PaddleOCR Local Service
# Based on official PaddlePaddle GPU image with CUDA 12.6

FROM paddlepaddle/paddle:3.3.0-gpu-cuda13.0-cudnn9.13

# Build-time environment variables
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
# PORT is needed for EXPOSE and HEALTHCHECK
ENV PORT=8080

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
# Note: Base image already has paddlepaddle-gpu installed
# Install paddleocr without dependencies to avoid overwriting GPU version
# Use --ignore-installed to bypass distutils-installed system packages (e.g., blinker)
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install paddleocr --no-deps \
    && pip install --ignore-installed -r requirements.txt

# Application code
COPY app.py /app/
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["./entrypoint.sh"]
