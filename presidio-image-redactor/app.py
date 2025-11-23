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

            # Check and log GPU availability
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_count = torch.cuda.device_count()
                    print(f"[GPU] CUDA available: True")
                    print(f"[GPU] Device count: {gpu_count}")
                    print(f"[GPU] Device 0: {gpu_name}")
                else:
                    print("[GPU] CUDA not available, EasyOCR will use CPU")
                    if ocr_gpu:
                        print("[GPU] WARNING: OCR_GPU=true but CUDA not available, falling back to CPU")
                        ocr_gpu = False
            except Exception as e:
                print(f"[GPU] Failed to check CUDA availability: {e}")
                ocr_gpu = False

            print(f"[EasyOCR] Initializing with languages: {ocr_languages}, GPU: {ocr_gpu}")
            self.ocr = EasyOCREngine(lang_list=ocr_languages, gpu=ocr_gpu, verbose=False)
            print("[EasyOCR] Initialization complete")

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

    def _is_adjacent(
        self,
        ocr_result: dict,
        idx1: int,
        idx2: int,
        h_gap_ratio: float = 1.5,
        v_overlap_ratio: float = 0.5
    ) -> bool:
        """Check if two OCR blocks are spatially adjacent.

        :param ocr_result: OCR result dict with left, top, width, height arrays
        :param idx1: Index of first block (should be to the left)
        :param idx2: Index of second block (should be to the right)
        :param h_gap_ratio: Max horizontal gap as ratio of avg char width
        :param v_overlap_ratio: Min vertical overlap ratio to consider same line
        :return: True if blocks are adjacent
        """
        left1, top1 = ocr_result["left"][idx1], ocr_result["top"][idx1]
        w1, h1 = ocr_result["width"][idx1], ocr_result["height"][idx1]
        left2, top2 = ocr_result["left"][idx2], ocr_result["top"][idx2]
        w2, h2 = ocr_result["width"][idx2], ocr_result["height"][idx2]

        # Check horizontal order: block2 should be to the right of block1
        right1 = left1 + w1
        if left2 < right1:
            return False

        # Check horizontal gap: should not be too far apart
        text1 = ocr_result["text"][idx1]
        avg_char_width = w1 / max(len(text1), 1)
        h_gap = left2 - right1
        if h_gap > avg_char_width * h_gap_ratio:
            return False

        # Check vertical alignment: blocks should overlap vertically
        bottom1, bottom2 = top1 + h1, top2 + h2
        overlap_top = max(top1, top2)
        overlap_bottom = min(bottom1, bottom2)
        overlap_height = max(0, overlap_bottom - overlap_top)

        min_height = min(h1, h2)
        if overlap_height < min_height * v_overlap_ratio:
            return False

        return True

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
        texts = ocr_result.get("text", [])

        # Bug 2 fix: Filter out targets with length < 2
        valid_targets = [t for t in texts_to_redact if len(t) >= 2]

        indices_to_redact = set()

        for target in valid_targets:
            # Single block match: target is substring of OCR text
            # Bug 3 fix: removed redundant exact match check
            for i, text in enumerate(texts):
                if target in text:
                    indices_to_redact.add(i)

            # Multi-block combination match (Bug 1 fix):
            # Try combining adjacent blocks to match target
            for start_idx in range(len(texts)):
                combined_text = ""
                combined_indices = []

                for j in range(start_idx, len(texts)):
                    # Check if adjacent (first block or adjacent to previous)
                    if j == start_idx or self._is_adjacent(
                        ocr_result, combined_indices[-1], j
                    ):
                        combined_text += texts[j]
                        combined_indices.append(j)

                        # Check if combined text matches target
                        if target in combined_text:
                            indices_to_redact.update(combined_indices)
                            break

                        # Stop if combined text exceeds target length without match
                        if len(combined_text) > len(target):
                            break
                    else:
                        break

        # Draw redaction boxes
        redacted = image.copy()
        draw = ImageDraw.Draw(redacted)

        for i in indices_to_redact:
            x0 = ocr_result["left"][i]
            y0 = ocr_result["top"][i]
            x1 = x0 + ocr_result["width"][i]
            y1 = y0 + ocr_result["height"][i]
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
