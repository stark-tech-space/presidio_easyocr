# Presidio 本機部署指南

本文檔說明如何在本機直接啟動三個服務：
- **PaddleOCR Local** - OCR 文字辨識服務
- **Presidio Analyzer** - PII 分析服務
- **Presidio Image Redactor** - 圖像脫敏服務

## 系統需求

- Python 3.10+
- CUDA 12.x（若使用 GPU）
- 至少 8GB RAM（建議 16GB）

## 架構說明

```
┌─────────────────────┐     ┌──────────────────────┐
│  Image Redactor     │────▶│  PaddleOCR Local     │
│  (Port: 5003)       │     │  (Port: 8180)        │
└─────────────────────┘     └──────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Presidio Analyzer  │
│  (Port: 5002)       │
└──────────────────────┘
```

---

## 1. PaddleOCR Local 服務

### 1.1 安裝依賴

```bash
cd presidio_easyocr/paddleocr-local

# 建立虛擬環境（建議）
python -m venv .venv
source .venv/bin/activate

# 安裝 PaddlePaddle GPU 版本（CUDA 12.x）
python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# 或者 CPU 版本
# python -m pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 安裝其他依賴
pip install -r requirements.txt
```

### 1.2 環境變數配置

| 變數名 | 預設值 | 說明 |
|--------|--------|------|
| `PORT` | 8180 | 服務監聽端口 |
| `WORKERS` | 1 | Gunicorn worker 數量 |
| `TIMEOUT` | 300 | 請求超時時間（秒） |
| `USE_GPU` | true | 是否使用 GPU |
| `USE_DOC_ORIENTATION` | false | 啟用文檔方向偵測 |
| `USE_DOC_UNWARPING` | false | 啟用文檔矯正 |
| `USE_TEXTLINE_ORIENTATION` | false | 啟用文字行方向偵測 |

### 1.3 啟動服務

```bash
cd /home/sw/developer/codes/presidio_easyocr/paddleocr-local
source .venv/bin/activate

# 方式一：使用 entrypoint.sh
export PORT=8180
export USE_GPU=true
./entrypoint.sh

# 方式二：直接啟動
python app.py
```

### 1.4 驗證服務

```bash
# 健康檢查
curl http://127.0.0.1:8180/health

# 就緒檢查（會載入模型）
curl http://127.0.0.1:8180/ready
```

---

## 2. Presidio Analyzer 服務

### 2.1 安裝依賴

```bash
cd /home/sw/developer/codes/presidio_easyocr/presidio-analyzer

# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install -e ".[server]"
pip install python-dotenv

# 安裝中文 spaCy 模型
python -m spacy download zh_core_web_trf
```

### 2.2 環境變數配置

編輯 `.env` 文件：

```bash
# /home/sw/developer/codes/presidio_easyocr/presidio-analyzer/.env
PORT=5002
WORKERS=1
TIMEOUT=300

# Presidio 配置文件
NLP_CONF_FILE=presidio_analyzer/conf/spacy_zh.yaml
ANALYZER_CONF_FILE=presidio_analyzer/conf/default_analyzer.yaml
RECOGNIZER_REGISTRY_CONF_FILE=presidio_analyzer/conf/default_recognizers.yaml

# 若使用 LLM 服務（可選）
OPENAI_API_BASE=http://127.0.0.1:8000/v1
OPENAI_API_KEY=not-needed
```

### 2.3 啟動服務

```bash
cd /home/sw/developer/codes/presidio_easyocr/presidio-analyzer
source .venv/bin/activate

# 方式一：使用啟動腳本
./start_local.sh

# 方式二：手動啟動
export $(grep -v '^#' .env | grep -v '^$' | xargs)
python app.py
```

### 2.4 驗證服務

```bash
# 健康檢查
curl http://127.0.0.1:5002/health

# 測試分析
curl -X POST http://127.0.0.1:5002/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "王小明的電話是 0912-345-678", "language": "zh"}'
```

---

## 3. Presidio Image Redactor 服務

### 3.1 安裝依賴

```bash
cd /home/sw/developer/codes/presidio_easyocr/presidio-image-redactor

# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install -e ".[server]"
pip install python-dotenv requests

# 若要使用 EasyOCR（可選）
pip install easyocr
```

### 3.2 環境變數配置

編輯 `.env` 文件：

```bash
# /home/sw/developer/codes/presidio_easyocr/presidio-image-redactor/.env

# 服務配置
PORT=5003
WORKERS=1
TIMEOUT=300

# OCR 配置 - 使用本地 PaddleOCR
OCR_ENGINE=paddleocr_api
PADDLEOCR_MODE=local
PADDLEOCR_LOCAL_URL=http://127.0.0.1:8180/ocr
PADDLEOCR_TIMEOUT=120

# 或者使用 EasyOCR
# OCR_ENGINE=easyocr
# OCR_LANGUAGES=ch_tra,en
# OCR_GPU=false
```

### 3.3 啟動服務

```bash
cd /home/sw/developer/codes/presidio_easyocr/presidio-image-redactor
source .venv/bin/activate

# 方式一：使用啟動腳本
./start_local.sh

# 方式二：手動啟動
export $(grep -v '^#' .env | grep -v '^$' | xargs)
python app.py
```

### 3.4 驗證服務

```bash
# 健康檢查
curl http://127.0.0.1:5003/health
```

---

## 快速啟動（三個終端機）

### 終端機 1：PaddleOCR
```bash
cd /home/sw/developer/codes/presidio_easyocr/paddleocr-local
source .venv/bin/activate
PORT=8180 USE_GPU=true ./entrypoint.sh
```

### 終端機 2：Analyzer
```bash
cd /home/sw/developer/codes/presidio_easyocr/presidio-analyzer
source .venv/bin/activate
./start_local.sh
```

### 終端機 3：Image Redactor
```bash
cd /home/sw/developer/codes/presidio_easyocr/presidio-image-redactor
source .venv/bin/activate
./start_local.sh
```

---

## 背景執行（使用 nohup）

若需要在背景持續執行，可使用以下方式：

```bash
# PaddleOCR
cd /home/sw/developer/codes/presidio_easyocr/paddleocr-local
source .venv/bin/activate
nohup PORT=8180 USE_GPU=true ./entrypoint.sh > paddleocr.log 2>&1 &

# Analyzer
cd /home/sw/developer/codes/presidio_easyocr/presidio-analyzer
source .venv/bin/activate
nohup ./start_local.sh > analyzer.log 2>&1 &

# Image Redactor
cd /home/sw/developer/codes/presidio_easyocr/presidio-image-redactor
source .venv/bin/activate
nohup ./start_local.sh > redactor.log 2>&1 &
```

查看日誌：
```bash
tail -f /home/sw/developer/codes/presidio_easyocr/paddleocr-local/paddleocr.log
tail -f /home/sw/developer/codes/presidio_easyocr/presidio-analyzer/analyzer.log
tail -f /home/sw/developer/codes/presidio_easyocr/presidio-image-redactor/redactor.log
```

---

## 服務端口匯總

| 服務 | 端口 | 健康檢查 |
|------|------|----------|
| PaddleOCR Local | 8180 | `GET /health` |
| Presidio Analyzer | 5002 | `GET /health` |
| Image Redactor | 5003 | `GET /health` |

---

## 故障排除

### PaddleOCR 模型下載失敗
首次啟動時會自動下載模型，若網路不穩定可能失敗。可設置代理：
```bash
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port
```

### spaCy 中文模型問題
```bash
# 重新安裝
python -m spacy download zh_core_web_trf --force
```

### GPU 記憶體不足
調整 `WORKERS=1` 或設置 `USE_GPU=false`

### 端口衝突
修改 `.env` 中的 `PORT` 設定，確保不與其他服務衝突
