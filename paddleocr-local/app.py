"""PaddleOCR Local HTTP API Service.

Provides HTTP API compatible with remote PaddleOCR API format,
enabling seamless switching between local and remote deployments.

PaddleOCR 3.x compatible.
"""

import base64
import logging
import os
import tempfile
import time
from typing import Any, Dict, List

import cv2
import numpy as np
from flask import Flask, jsonify, request

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


def decode_image_to_file(image_data: str) -> str:
    """Decode base64 image and save to temp file.

    PaddleOCR 3.x works best with file paths rather than numpy arrays.
    This avoids potential format conversion issues.
    """
    image_bytes = base64.b64decode(image_data)

    # Create temp file with proper extension
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(image_bytes)
        return f.name


def decode_image_cv2(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array using cv2 (BGR format).

    PaddleOCR expects cv2.imread format (BGR, uint8, C-contiguous).
    """
    image_bytes = base64.b64decode(image_data)

    # Use cv2.imdecode to get proper BGR format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    return img


def process_ocr_result(result) -> Dict[str, Any]:
    """Convert PaddleOCR 3.x result to remote API compatible format.

    PaddleOCR 3.x returns a list of result objects.
    Each result object has a 'res' attribute containing the actual data.

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
    for res_obj in result:
        # Access the 'res' attribute which contains the actual OCR data
        if hasattr(res_obj, 'res'):
            data = res_obj.res
        elif hasattr(res_obj, '__getitem__'):
            data = res_obj.get('res', res_obj)
        elif isinstance(res_obj, dict):
            data = res_obj.get('res', res_obj)
        else:
            logger.warning(f"Unknown result type: {type(res_obj)}")
            continue

        # Get detection polygons
        if 'dt_polys' in data and data['dt_polys'] is not None:
            polys = data['dt_polys']
            if isinstance(polys, np.ndarray):
                dt_polys = polys.tolist()
            elif isinstance(polys, list):
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
    temp_file = None
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

        # Decode image - use file-based approach for reliability
        start_time = time.time()
        temp_file = decode_image_to_file(image_b64)
        decode_time = time.time() - start_time

        # Perform OCR using file path (most reliable method)
        ocr = get_ocr()
        start_time = time.time()
        result = ocr.predict(temp_file)
        ocr_time = time.time() - start_time

        logger.info(f"[OCR] decode={decode_time:.3f}s, ocr={ocr_time:.3f}s")

        # Convert result format
        response = process_ocr_result(result)
        return jsonify(response)

    except Exception as e:
        logger.error(f"[OCR Error] {str(e)}", exc_info=True)
        return jsonify({
            "errorCode": 500,
            "errorMsg": str(e)
        }), 500
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass


@app.route("/ocr/numpy", methods=["POST"])
def ocr_numpy_endpoint():
    """Alternative OCR endpoint using numpy array input.

    This endpoint uses cv2.imdecode for proper BGR format handling.
    Use this if the file-based endpoint has issues.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "errorCode": 1,
                "errorMsg": "No JSON data provided"
            }), 400

        image_b64 = data.get("file")
        if not image_b64:
            return jsonify({
                "errorCode": 2,
                "errorMsg": "Missing 'file' field (base64 image)"
            }), 400

        # Decode image using cv2 (BGR format)
        start_time = time.time()
        image_array = decode_image_cv2(image_b64)
        decode_time = time.time() - start_time

        logger.info(f"[OCR] Image shape: {image_array.shape}, dtype: {image_array.dtype}")

        # Perform OCR
        ocr = get_ocr()
        start_time = time.time()
        result = ocr.predict(image_array)
        ocr_time = time.time() - start_time

        logger.info(f"[OCR] decode={decode_time:.3f}s, ocr={ocr_time:.3f}s")

        response = process_ocr_result(result)
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
