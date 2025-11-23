# DGX Spark (ARM64 Grace Blackwell GB10) requires NGC container with CUDA 13.0 support
# 25.09-py3 is optimized for DGX Spark with PyTorch 2.9.0a and CUDA 13.0.1
FROM nvcr.io/nvidia/pytorch:25.09-py3

# Build arguments for config files
ARG NLP_CONF_FILE=presidio_analyzer/conf/spacy_zh.yaml
ARG ANALYZER_CONF_FILE=presidio_analyzer/conf/default_analyzer.yaml
ARG RECOGNIZER_REGISTRY_CONF_FILE=presidio_analyzer/conf/default_recognizers.yaml

# Environment variables (rarely change - good for caching)
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PORT=3000
ENV WORKERS=1
ENV TIMEOUT=300

WORKDIR /app

# ============================================
# Layer 1: System dependencies (rarely change)
# ============================================
RUN apt-get update \
  && apt-get install build-essential curl --no-install-recommends -y \
  && rm -rf /var/lib/apt/lists/*

# ============================================
# Layer 2: Fix numpy compatibility (rarely change)
# ============================================
# NGC image ships with numpy 2.x which breaks some packages
RUN pip install "numpy<2"

# ============================================
# Layer 3: Python dependencies (change when pyproject.toml changes)
# ============================================
COPY ./pyproject.toml /app/
RUN pip install poetry \
    && poetry install --no-root --only=main -E server \
    && rm -rf $(poetry config cache-dir)

# Install spacy-transformers for GPU-accelerated transformer models (zh_core_web_trf)
RUN pip install spacy-transformers

# ============================================
# Layer 4: NLP models (change when NLP_CONF_FILE changes)
# ============================================
# Set config file env vars (need ARG values)
ENV ANALYZER_CONF_FILE=${ANALYZER_CONF_FILE}
ENV RECOGNIZER_REGISTRY_CONF_FILE=${RECOGNIZER_REGISTRY_CONF_FILE}
ENV NLP_CONF_FILE=${NLP_CONF_FILE}

# Copy NLP config and install script first
COPY ${NLP_CONF_FILE} /app/${NLP_CONF_FILE}
COPY ./install_nlp_models.py /app/

# Install NLP models specified in NLP_CONF_FILE
RUN python install_nlp_models.py --conf_file ${NLP_CONF_FILE}

# ============================================
# Layer 5: Application code (changes frequently)
# ============================================
# Copy config files
COPY ${ANALYZER_CONF_FILE} /app/${ANALYZER_CONF_FILE}
COPY ${RECOGNIZER_REGISTRY_CONF_FILE} /app/${RECOGNIZER_REGISTRY_CONF_FILE}

# Copy application source code
COPY . /app/

EXPOSE ${PORT}
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1
CMD ["./entrypoint.sh"]
