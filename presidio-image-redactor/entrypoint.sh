#!/bin/sh
set -e

PORT=${PORT:-5003}
WORKERS=${WORKERS:-1}
TIMEOUT=${TIMEOUT:-300}

echo "[Image Redactor] Starting server..."
echo "[Image Redactor] Port: $PORT"
echo "[Image Redactor] Workers: $WORKERS"
echo "[Image Redactor] Timeout: $TIMEOUT"
echo "[Image Redactor] OCR_ENGINE: ${OCR_ENGINE:-easyocr}"

# Use poetry if available, otherwise use direct gunicorn
if command -v poetry >/dev/null 2>&1 && [ -f "pyproject.toml" ]; then
    exec poetry run gunicorn -w "$WORKERS" -t "$TIMEOUT" -b "0.0.0.0:$PORT" "app:create_app()"
else
    exec gunicorn -w "$WORKERS" -t "$TIMEOUT" -b "0.0.0.0:$PORT" "app:create_app()"
fi