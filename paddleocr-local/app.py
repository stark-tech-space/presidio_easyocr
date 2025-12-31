"""PaddleOCR Local HTTP API Service.

Provides HTTP API compatible with remote PaddleOCR API format,
enabling seamless switching between local and remote deployments.
"""

import base64
import io
import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

# Lazy-loaded PaddleOCR instance
_ocr_instance = None

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("paddleocr-local")

# Configuration from environment
USE_GPU = os.environ.get("USE_GPU", "true").lower() == "true"
USE_DOC_ORIENTATION = os.environ.get("USE_DOC_ORIENTATION", "false").lower() == "true"
USE_DOC_UNWARPING = os.environ.get("USE_DOC_UNWARPING", "false").lower() == "true"
USE_TEXTLINE_ORIENTATION = os.environ.get("USE_TEXTLINE_ORIENTATION", "false").lower() == "true"


def get_ocr():
    """Lazy initialization of PaddleOCR instance."""
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR

        logger.info("[PaddleOCR] Initializing...")
        device = "gpu" if USE_GPU else "cpu"

        _ocr_instance = PaddleOCR(
            use_doc_orientation_classify=USE_DOC_ORIENTATION,
            use_doc_unwarping=USE_DOC_UNWARPING,
            use_textline_orientation=USE_TEXTLINE_ORIENTATION,
            device=device
        )
        logger.info(f"[PaddleOCR] Initialized with device={device}")

    return _ocr_instance


def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')

    return np.array(image)


def process_ocr_result(result) -> Dict[str, Any]:
    """Convert PaddleOCR local result to remote API compatible format.

    Remote API format:
    {
        "errorCode": 0,
        "errorMsg": "",
        "result": {
            "ocrResults": [
                {
                    "prunedResult": {
                        "dt_polys": [...],
                        "rec_polys": [...],
                        "rec_texts": [...],
                        "rec_scores": [...]
                    }
                }
            ]
        }
    }
    """
    dt_polys: List = []
    rec_polys: List = []
    rec_texts: List = []
    rec_scores: List = []

    # Process PaddleOCR 3.x result format
    # result is a list of dicts (or OCRResult objects that behave like dicts)
    for res in result:
        # Handle both dict and object with dict-like access
        if isinstance(res, dict):
            data = res
        elif hasattr(res, '__getitem__'):
            data = res
        else:
            continue

        # Get detection polygons
        if 'dt_polys' in data and data['dt_polys'] is not None:
            polys = data['dt_polys']
            if isinstance(polys, np.ndarray):
                dt_polys = polys.tolist()
            elif isinstance(polys, list):
                # Convert each polygon array to list
                dt_polys = [p.tolist() if isinstance(p, np.ndarray) else p for p in polys]
            else:
                dt_polys = list(polys) if polys else []

        # Get recognized texts
        if 'rec_texts' in data and data['rec_texts'] is not None:
            rec_texts = list(data['rec_texts'])

        # Get recognition scores
        if 'rec_scores' in data and data['rec_scores'] is not None:
            scores = data['rec_scores']
            if isinstance(scores, np.ndarray):
                rec_scores = scores.tolist()
            else:
                rec_scores = [float(s) for s in scores]

        # Get rec_polys if available, otherwise use dt_polys
        if 'rec_polys' in data and data['rec_polys'] is not None:
            polys = data['rec_polys']
            if isinstance(polys, np.ndarray):
                rec_polys = polys.tolist()
            elif isinstance(polys, list):
                rec_polys = [p.tolist() if isinstance(p, np.ndarray) else p for p in polys]
            else:
                rec_polys = list(polys) if polys else []
        else:
            rec_polys = dt_polys

    return {
        "errorCode": 0,
        "errorMsg": "",
        "result": {
            "ocrResults": [
                {
                    "prunedResult": {
                        "dt_polys": dt_polys,
                        "rec_polys": rec_polys,
                        "rec_texts": rec_texts,
                        "rec_scores": rec_scores
                    }
                }
            ]
        }
    }


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "paddleocr-local",
        "gpu_enabled": USE_GPU
    })


@app.route("/ready", methods=["GET"])
def ready():
    """Readiness check - verify OCR model is loaded."""
    try:
        get_ocr()
        return jsonify({
            "status": "ready",
            "model": "PP-OCRv5"
        })
    except Exception as e:
        return jsonify({
            "status": "not_ready",
            "error": str(e)
        }), 503


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    """OCR endpoint - compatible with remote API format.

    Request format:
    {
        "file": "<base64_image>",
        "fileType": 1,
        "useDocOrientationClassify": false,
        "useDocUnwarping": false,
        "useTextlineOrientation": false
    }

    Response format matches remote API.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "errorCode": 1,
                "errorMsg": "No JSON data provided"
            }), 400

        # Get image data
        image_b64 = data.get("file")
        if not image_b64:
            return jsonify({
                "errorCode": 2,
                "errorMsg": "Missing 'file' field (base64 image)"
            }), 400

        # Decode image
        start_time = time.time()
        image_array = decode_image(image_b64)
        decode_time = time.time() - start_time

        # Perform OCR
        ocr = get_ocr()
        start_time = time.time()
        result = ocr.predict(image_array)
        ocr_time = time.time() - start_time

        logger.info(f"[OCR] decode={decode_time:.3f}s, ocr={ocr_time:.3f}s")

        # Debug: print raw result
        print(f"[OCR DEBUG] Raw result type: {type(result)}", flush=True)
        print(f"[OCR DEBUG] Raw result: {result}", flush=True)
        if result:
            for i, res in enumerate(result):
                print(f"[OCR DEBUG] Result[{i}] type: {type(res)}", flush=True)
                if hasattr(res, 'res'):
                    print(f"[OCR DEBUG] Result[{i}].res: {res.res}", flush=True)
                if hasattr(res, '__dict__'):
                    print(f"[OCR DEBUG] Result[{i}].__dict__: {res.__dict__}", flush=True)

        # Convert result format
        response = process_ocr_result(result)
        logger.info(f"[OCR] Response: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"[OCR Error] {str(e)}", exc_info=True)
        return jsonify({
            "errorCode": 500,
            "errorMsg": str(e)
        }), 500


def create_app():
    """Factory function for gunicorn."""
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
