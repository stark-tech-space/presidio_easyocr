#!/bin/bash
set -e

cd "$(dirname "$0")"

# 加载环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

echo "=== Presidio Image Redactor 本地启动 ==="
echo "PORT: ${PORT:-3000}"
echo "OCR_ENGINE: $OCR_ENGINE"
echo "PADDLEOCR_MODE: $PADDLEOCR_MODE"
echo "PADDLEOCR_LOCAL_URL: $PADDLEOCR_LOCAL_URL"
echo ""

# 启动服务
exec python app.py
