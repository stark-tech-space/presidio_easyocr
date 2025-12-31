#!/bin/bash
set -e

cd "$(dirname "$0")"

# 加载环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

echo "=== Presidio Analyzer 本地启动 ==="
echo "PORT: ${PORT:-3000}"
echo "NLP_CONF_FILE: $NLP_CONF_FILE"
echo "ANALYZER_CONF_FILE: $ANALYZER_CONF_FILE"
echo "RECOGNIZER_REGISTRY_CONF_FILE: $RECOGNIZER_REGISTRY_CONF_FILE"
echo ""

# 检查中文模型
python -c "import spacy; spacy.load('zh_core_web_trf')" 2>/dev/null || {
    echo "中文模型 zh_core_web_trf 未安装，正在下载..."
    python -m spacy download zh_core_web_trf
}

# 启动服务
exec python app.py
