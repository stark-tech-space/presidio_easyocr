# DGX Spark (ARM64 Grace Blackwell GB10) requires NGC container with CUDA 13.0 support
# 25.09-py3 is optimized for DGX Spark with PyTorch 2.9.0a and CUDA 13.0.1
FROM nvcr.io/nvidia/pytorch:25.09-py3

ARG NLP_CONF_FILE
ARG ANALYZER_CONF_FILE
ARG RECOGNIZER_REGISTRY_CONF_FILE
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_VIRTUALENVS_CREATE=false

ENV ANALYZER_CONF_FILE=${ANALYZER_CONF_FILE}
ENV RECOGNIZER_REGISTRY_CONF_FILE=${RECOGNIZER_REGISTRY_CONF_FILE}
ENV NLP_CONF_FILE=${NLP_CONF_FILE}

ENV PORT=3000
ENV WORKERS=1
ENV TIMEOUT=300
# EasyOCR languages: ch_tra (Traditional Chinese), en (English)
ENV OCR_LANGUAGES=ch_tra,en
ENV OCR_GPU=true

WORKDIR /app

COPY ${ANALYZER_CONF_FILE} /app/${ANALYZER_CONF_FILE}
COPY ${RECOGNIZER_REGISTRY_CONF_FILE} /app/${RECOGNIZER_REGISTRY_CONF_FILE}
COPY ${NLP_CONF_FILE} /app/${NLP_CONF_FILE}

# Install system dependencies for EasyOCR and image processing
RUN apt-get update \
  && apt-get install ffmpeg libsm6 libxext6 curl --no-install-recommends -y \
  && rm -rf /var/lib/apt/lists/*

# Downgrade numpy to <2 to fix compatibility with opencv-python, pyarrow, etc.
# NGC image ships with numpy 2.x which breaks many packages compiled against numpy 1.x
RUN pip install "numpy<2"

COPY ./pyproject.toml /app/
RUN pip install poetry \
    && poetry install --no-root --only=main -E server -E easyocr \
    && rm -rf $(poetry config cache-dir)

# Install spaCy model during build (as root) so it's available to non-root user at runtime
RUN python -m spacy download en_core_web_lg

COPY . /app/

EXPOSE ${PORT}
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1
CMD ["./entrypoint.sh"]
