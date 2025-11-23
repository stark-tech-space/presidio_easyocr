"""REST API server for image redactor."""

import base64
import logging
import os
from io import BytesIO
from typing import List, Optional

from flask import Flask, Response, jsonify, request
from PIL import Image, ImageDraw
from presidio_image_redactor import ImageRedactorEngine, ImageAnalyzerEngine
from presidio_image_redactor.entities import InvalidParamError
from presidio_image_redactor.entities.api_request_convertor import (
    color_fill_string_to_value,
    get_json_data,
    image_to_byte_array,
)

# Try to import EasyOCR engine
try:
    from presidio_image_redactor import EasyOCREngine
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

DEFAULT_PORT = "3000"

WELCOME_MESSAGE = r"""
 _______  _______  _______  _______ _________ ______  _________ _______
(  ____ )(  ____ )(  ____ \(  ____ \\__   __/(  __  \ \__   __/(  ___  )
| (    )|| (    )|| (    \/| (    \/   ) (   | (  \  )   ) (   | (   ) |
| (____)|| (____)|| (__    | (_____    | |   | |   ) |   | |   | |   | |
|  _____)|     __)|  __)   (_____  )   | |   | |   | |   | |   | |   | |
| (      | (\ (   | (            ) |   | |   | |   ) |   | |   | |   | |
| )      | ) \ \__| (____/\/\____) |___) (___| (__/  )___) (___| (___) |
|/       |/   \__/(_______/\_______)\_______/(______/ \_______/(_______)
"""


class Server:
    """Flask server for image redactor."""

    def __init__(self):
        self.logger = logging.getLogger("presidio-image-redactor")
        self.app = Flask(__name__)
        self.logger.info("Starting image redactor engine")

        # Initialize OCR engine
        self.ocr = None
        if EASYOCR_AVAILABLE:
            ocr_languages = os.environ.get("OCR_LANGUAGES", "ch_tra,en").split(",")
            ocr_gpu = os.environ.get("OCR_GPU", "false").lower() == "true"

            # Log GPU availability
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    gpu_name = torch.cuda.get_device_name(0)
                    self.logger.info(f"CUDA available: {cuda_available}, GPU: {gpu_name}")
                else:
                    self.logger.warning("CUDA not available, EasyOCR will use CPU")
                    if ocr_gpu:
                        self.logger.warning("OCR_GPU=true but CUDA not available, falling back to CPU")
                        ocr_gpu = False
            except Exception as e:
                self.logger.warning(f"Failed to check CUDA availability: {e}")
                ocr_gpu = False

            self.logger.info(f"Initializing EasyOCR with languages: {ocr_languages}, GPU: {ocr_gpu}")
            self.ocr = EasyOCREngine(lang_list=ocr_languages, gpu=ocr_gpu, verbose=False)

        # Initialize engines
        if self.ocr:
            self.image_analyzer = ImageAnalyzerEngine(ocr=self.ocr)
            self.engine = ImageRedactorEngine(image_analyzer_engine=self.image_analyzer)
        else:
            self.engine = ImageRedactorEngine()
            self.image_analyzer = None

        self.logger.info(WELCOME_MESSAGE)

        @self.app.route("/health")
        def health() -> str:
            """Return basic health probe result."""
            return "Presidio Image Redactor service is up"

        @self.app.route("/redact", methods=["POST"])
        def redact():
            """Return a redacted image.

            Supports two modes:
            1. texts_to_redact: Directly specify texts to redact (OCR + match)
            2. analyzer_entities: Use Presidio analyzer to detect PII
            """
            # Handle JSON request
            if request.get_json(silent=True) and "image" in request.json:
                json_data = request.json
                im = Image.open(BytesIO(base64.b64decode(json_data.get("image"))))

                # Get color fill
                color_fill_str = json_data.get("color_fill", "0,0,0")
                color_fill = parse_color_fill(color_fill_str)

                # Mode 1: Direct text matching
                texts_to_redact = json_data.get("texts_to_redact")
                if texts_to_redact:
                    if not self.ocr:
                        raise InvalidParamError(
                            "texts_to_redact requires EasyOCR. "
                            "Install with: pip install easyocr"
                        )
                    redacted_image = self._redact_by_texts(
                        im, texts_to_redact, color_fill
                    )
                else:
                    # Mode 2: Analyzer-based detection
                    analyzer_entities = json_data.get("analyzer_entities")
                    language = json_data.get("language", "en")
                    redacted_image = self.engine.redact(
                        im, color_fill,
                        entities=analyzer_entities,
                        language=language
                    )

                img_byte_arr = image_to_byte_array(redacted_image, im.format or "PNG")
                return Response(
                    base64.b64encode(img_byte_arr),
                    mimetype="application/octet-stream"
                )

            # Handle form-data request (legacy)
            elif request.files and "image" in request.files:
                params = get_json_data(request.form.get("data"))
                color_fill = color_fill_string_to_value(params)
                im = Image.open(request.files.get("image"))
                redacted_image = self.engine.redact(im, color_fill, score_threshold=0.4)
                img_byte_arr = image_to_byte_array(redacted_image, im.format)
                return Response(img_byte_arr, mimetype="application/octet-stream")
            else:
                raise InvalidParamError("Invalid parameter, please add image data")

        @self.app.route("/ocr", methods=["POST"])
        def ocr_endpoint():
            """Perform OCR on an image and return detected texts with positions.

            This can be used to preview what texts will be detected before redaction.
            """
            if not self.ocr:
                return jsonify(error="OCR not available. Install easyocr."), 400

            if not request.get_json(silent=True) or "image" not in request.json:
                return jsonify(error="No image provided"), 400

            im = Image.open(BytesIO(base64.b64decode(request.json.get("image"))))
            ocr_result = self.ocr.perform_ocr(im)

            # Format result for easier consumption
            texts = []
            for i in range(len(ocr_result.get("text", []))):
                texts.append({
                    "text": ocr_result["text"][i],
                    "left": ocr_result["left"][i],
                    "top": ocr_result["top"][i],
                    "width": ocr_result["width"][i],
                    "height": ocr_result["height"][i],
                    "confidence": ocr_result["conf"][i],
                })

            return jsonify({
                "texts": texts,
                "full_text": self.ocr.get_text_from_ocr_dict(ocr_result)
            })

        @self.app.errorhandler(InvalidParamError)
        def invalid_param(err):
            self.logger.warning(
                f"failed to redact image with validation error: {err.err_msg}"
            )
            return jsonify(error=err.err_msg), 422

        @self.app.errorhandler(Exception)
        def server_error(e):
            self.logger.error(f"A fatal error occurred during execution: {e}")
            return jsonify(error=str(e)), 500

    def _redact_by_texts(
        self,
        image: Image.Image,
        texts_to_redact: List[str],
        color_fill: tuple
    ) -> Image.Image:
        """Redact specific texts from image.

        :param image: PIL Image to redact
        :param texts_to_redact: List of texts to find and redact
        :param color_fill: Color to use for redaction
        :return: Redacted image
        """
        # Perform OCR
        ocr_result = self.ocr.perform_ocr(image)

        # Find matching bboxes
        bboxes_to_redact = []
        for i, text in enumerate(ocr_result.get("text", [])):
            # Check for exact match or substring match
            should_redact = False
            for target in texts_to_redact:
                # Exact match
                if target == text:
                    should_redact = True
                    break
                # Target is substring of OCR text
                if target in text:
                    should_redact = True
                    break
                # OCR text is substring of target (only if text is significant)
                # Require at least 2 chars to avoid single digit matches
                if len(text) >= 2 and text in target:
                    should_redact = True
                    break

            if should_redact:
                bboxes_to_redact.append({
                    "left": ocr_result["left"][i],
                    "top": ocr_result["top"][i],
                    "width": ocr_result["width"][i],
                    "height": ocr_result["height"][i],
                })

        # Draw redaction boxes
        redacted = image.copy()
        draw = ImageDraw.Draw(redacted)

        for bbox in bboxes_to_redact:
            x0 = bbox["left"]
            y0 = bbox["top"]
            x1 = x0 + bbox["width"]
            y1 = y0 + bbox["height"]
            draw.rectangle([x0, y0, x1, y1], fill=color_fill)

        return redacted


def parse_color_fill(color_str: str) -> tuple:
    """Parse color fill string to tuple.

    :param color_str: Color string like "0,0,0" or "255"
    :return: Color tuple
    """
    if "," in color_str:
        parts = color_str.split(",")
        return tuple(int(p.strip()) for p in parts)
    else:
        val = int(color_str)
        return (val, val, val)


def create_app():  # noqa
    server = Server()
    return server.app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    app.run(host="0.0.0.0", port=port)
