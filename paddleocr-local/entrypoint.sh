#!/bin/bash
set -e

PORT=${PORT:-8180}
WORKERS=${WORKERS:-1}
TIMEOUT=${TIMEOUT:-300}

echo "[PaddleOCR Local] Starting server..."
echo "[PaddleOCR Local] Port: $PORT"
echo "[PaddleOCR Local] Workers: $WORKERS"
echo "[PaddleOCR Local] Timeout: $TIMEOUT"
echo "[PaddleOCR Local] GPU: ${USE_GPU:-true}"

exec gunicorn "app:create_app()" \
    --bind "0.0.0.0:$PORT" \
    --workers "$WORKERS" \
    --timeout "$TIMEOUT" \
    --worker-class gevent \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --log-level info
