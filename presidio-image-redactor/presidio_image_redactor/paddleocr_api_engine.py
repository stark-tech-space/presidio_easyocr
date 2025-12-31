"""PaddleOCR API engine for Presidio Image Redactor.

This module provides a PaddleOCR API-based OCR engine that supports both:
- Remote mode: Uses the online PaddleOCR service (PP-OCRv5)
- Local mode: Uses locally deployed PaddleOCR Docker service
"""

import base64
import io
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from presidio_image_redactor import OCR

try:  # pragma: no cover - optional dependency
    import requests
except ImportError:  # pragma: no cover - handled at runtime
    requests = None


class PaddleOCRAPIEngine(OCR):
    """OCR engine that uses PaddleOCR API for text detection and recognition.

    Supports two modes:
    - remote: Uses online PaddleOCR service with token authentication
    - local: Uses locally deployed PaddleOCR Docker service (no auth required)

    PaddleOCR PP-OCRv5 provides high-accuracy OCR for Chinese (simplified and
    traditional), English, Japanese, and many other languages.

    :param api_url: The PaddleOCR API endpoint URL.
    :param token: The authentication token (required for remote mode, optional for local).
    :param mode: Operation mode - "remote" or "local". Default is "remote".
    :param use_doc_orientation_classify: Whether to use document orientation
                                         classification. Default is False.
    :param use_doc_unwarping: Whether to use document unwarping for curved/wrinkled
                              images. Default is False.
    :param use_textline_orientation: Whether to use textline orientation correction.
                                     Default is False.
    :param timeout: Request timeout in seconds. Default is 60.
    :param api_kwargs: Additional keyword arguments passed to the API request.

    Example usage (remote mode):
        >>> from PIL import Image
        >>> from presidio_image_redactor.paddleocr_api_engine import PaddleOCRAPIEngine
        >>>
        >>> # Create PaddleOCR API engine (remote)
        >>> ocr = PaddleOCRAPIEngine(
        ...     api_url="https://your-api-endpoint.com/ocr",
        ...     token="your-token",
        ...     mode="remote"
        ... )

    Example usage (local mode):
        >>> # Create PaddleOCR API engine (local)
        >>> ocr = PaddleOCRAPIEngine(
        ...     api_url="http://paddleocr-local:8080/ocr",
        ...     mode="local"
        ... )
        >>>
        >>> # Perform OCR
        >>> image = Image.open("document.png")
        >>> result = ocr.perform_ocr(image)
        >>> print(result['text'])  # List of detected words
    """

    def __init__(
        self,
        api_url: str,
        token: Optional[str] = None,
        mode: str = "remote",
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        timeout: int = 60,
        api_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Initialize PaddleOCR API engine with specified parameters."""
        if requests is None:
            raise ImportError(
                "requests is not installed. Install it with 'pip install requests'"
            )

        self.api_url = api_url
        self.token = token
        self.mode = mode.lower()

        # Validate mode
        if self.mode not in ("remote", "local"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'remote' or 'local'.")

        # Remote mode requires token
        if self.mode == "remote" and not token:
            raise ValueError("Token is required for remote mode")

        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.timeout = timeout
        self.api_kwargs = api_kwargs or {}

    def perform_ocr(self, image: object, **kwargs) -> dict:
        """Perform OCR on a given image using PaddleOCR API.

        :param image: PIL Image/numpy array or file path(str) to be processed
        :param kwargs: Additional OCR parameters passed to the API

        :return: Dictionary containing bboxes and text for each detected word:
                {
                    "text": ["word1", "word2", ...],
                    "left": [x1, x2, ...],
                    "top": [y1, y2, ...],
                    "width": [w1, w2, ...],
                    "height": [h1, h2, ...],
                    "conf": [confidence1, confidence2, ...]
                }
        """
        # Convert image to base64
        image_base64 = self._prepare_image(image)

        # Build headers based on mode
        headers = {"Content-Type": "application/json"}
        if self.mode == "remote" and self.token:
            headers["Authorization"] = f"token {self.token}"

        payload = {
            "file": image_base64,
            "fileType": 1,  # 1 for image, 0 for PDF
            "useDocOrientationClassify": self.use_doc_orientation_classify,
            "useDocUnwarping": self.use_doc_unwarping,
            "useTextlineOrientation": self.use_textline_orientation,
            **self.api_kwargs,
            **kwargs
        }

        # Call API
        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"PaddleOCR API request failed with status {response.status_code}: "
                f"{response.text}"
            )

        result = response.json()

        # Check for API errors
        if result.get("errorCode", 0) != 0:
            raise RuntimeError(
                f"PaddleOCR API error: {result.get('errorMsg', 'Unknown error')}"
            )

        # Parse OCR results
        ocr_results = result.get("result", {}).get("ocrResults", [])
        if not ocr_results:
            return self._empty_result()

        # Get the first page result (for single image)
        pruned_result = ocr_results[0].get("prunedResult", {})
        if not pruned_result:
            return self._empty_result()

        # Extract OCR data from prunedResult
        return self._parse_pruned_result(pruned_result)

    def _prepare_image(self, image: Union[str, np.ndarray, Image.Image]) -> str:
        """Convert various image formats to base64 string for API.

        :param image: Input image in various formats
        :return: Base64 encoded string of the image
        """
        if isinstance(image, str):
            # File path - read directly
            with open(image, "rb") as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode("ascii")

        if isinstance(image, np.ndarray):
            # Numpy array - convert to PIL first
            image = Image.fromarray(image)

        if isinstance(image, Image.Image):
            # PIL Image - convert to bytes
            buffer = io.BytesIO()
            # Convert to RGB if needed (e.g., for RGBA images)
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            image.save(buffer, format='JPEG', quality=95)
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode("ascii")

        raise ValueError(f"Unsupported image type: {type(image)}")

    def _empty_result(self) -> dict:
        """Return an empty OCR result structure."""
        return {
            "text": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "conf": []
        }

    def _parse_pruned_result(self, pruned_result: dict) -> dict:
        """Parse PaddleOCR prunedResult to Presidio format.

        PaddleOCR returns:
        - dt_polys: Detection polygons [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        - rec_texts: List of recognized texts
        - rec_scores: List of recognition confidence scores (0-1)
        - rec_polys: Recognition polygons (same format as dt_polys)

        :param pruned_result: The prunedResult from PaddleOCR API
        :return: Dictionary in Presidio format
        """
        # Use rec_polys (recognition polygons) for bounding boxes
        # Fall back to dt_polys if rec_polys not available
        polygons = pruned_result.get("rec_polys", pruned_result.get("dt_polys", []))
        rec_texts = pruned_result.get("rec_texts", [])
        rec_scores = pruned_result.get("rec_scores", [])

        # Handle case where fields might be missing
        if not rec_texts:
            return self._empty_result()

        # Ensure we have matching lengths
        num_texts = len(rec_texts)
        default_polygon = [[0, 0], [0, 0], [0, 0], [0, 0]]
        if len(polygons) < num_texts:
            polygons = polygons + [default_polygon] * (num_texts - len(polygons))
        if len(rec_scores) < num_texts:
            rec_scores = rec_scores + [0.0] * (num_texts - len(rec_scores))

        # Convert each text line to word-level entries
        word_list = []
        for i, text in enumerate(rec_texts):
            if not text or not text.strip():
                continue

            polygon = polygons[i] if i < len(polygons) else default_polygon
            confidence = rec_scores[i] if i < len(rec_scores) else 0.0

            # Split text into words and distribute bounding boxes
            word_spans = self._split_text_into_spans(text)
            if not word_spans:
                continue

            if len(word_spans) == 1:
                word_list.append({
                    "bbox": polygon,
                    "text": word_spans[0]["text"],
                    "conf": confidence,
                })
                continue

            # Split bounding box for multiple words
            word_bboxes = self._split_bbox_by_spans(polygon, word_spans, len(text))
            for span, word_bbox in zip(word_spans, word_bboxes):
                word_list.append({
                    "bbox": word_bbox,
                    "text": span["text"],
                    "conf": confidence,
                })

        return self._to_presidio_format(word_list)

    def _split_text_into_spans(self, text: str) -> List[dict]:
        """Split text into spans retaining indices to keep spatial proportions."""
        if not text:
            return []

        stripped = text.strip()
        if not stripped:
            return []

        # Languages without whitespace separators (e.g., Chinese) keep full span
        if stripped == text and " " not in text:
            return [{"text": text, "start": 0, "end": len(text)}]

        spans: List[dict] = []
        start_idx: Optional[int] = None
        for idx, char in enumerate(text):
            if char.isspace():
                if start_idx is not None:
                    span = {
                        "text": text[start_idx:idx],
                        "start": start_idx,
                        "end": idx
                    }
                    spans.append(span)
                    start_idx = None
                continue

            if start_idx is None:
                start_idx = idx

        if start_idx is not None:
            span = {
                "text": text[start_idx:],
                "start": start_idx,
                "end": len(text)
            }
            spans.append(span)

        return [span for span in spans if span["text"].strip()]

    def _split_bbox_by_spans(
        self,
        line_bbox: List[List[float]],
        spans: List[dict],
        text_length: int,
    ) -> List[List[List[float]]]:
        """Split line-level bounding boxes honoring character spans."""
        x_coords = [point[0] for point in line_bbox]
        y_coords = [point[1] for point in line_bbox]

        left = min(x_coords)
        right = max(x_coords)
        top = min(y_coords)
        bottom = max(y_coords)

        total_width = max(right - left, 1)
        safe_length = max(text_length, 1)

        bboxes: List[List[List[float]]] = []
        for span in spans:
            start_ratio = span["start"] / safe_length
            end_ratio = span["end"] / safe_length

            word_left = left + total_width * start_ratio
            word_right = left + total_width * end_ratio

            bbox = [
                [word_left, top],
                [word_right, top],
                [word_right, bottom],
                [word_left, bottom],
            ]
            bboxes.append(bbox)

        return bboxes

    def _polygon_to_bbox(self, polygon: List[List[float]]) -> Tuple[int, int, int, int]:
        """Convert polygon coordinates to (left, top, width, height) format.

        :param polygon: List of [x, y] coordinates defining the polygon
        :return: Tuple of (left, top, width, height)
        """
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]

        left = int(min(x_coords))
        top = int(min(y_coords))
        right = int(max(x_coords))
        bottom = int(max(y_coords))

        width = right - left
        height = bottom - top

        return (left, top, width, height)

    def _to_presidio_format(self, word_list: List[dict]) -> dict:
        """Convert word list to Presidio's expected dictionary format.

        :param word_list: List of dictionaries with 'bbox', 'text', and 'conf' keys
        :return: Dictionary in Presidio format with parallel arrays
        """
        result = {
            "text": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "conf": []
        }

        for word_data in word_list:
            bbox = word_data['bbox']
            left, top, width, height = self._polygon_to_bbox(bbox)

            result["text"].append(word_data['text'])
            result["left"].append(left)
            result["top"].append(top)
            result["width"].append(width)
            result["height"].append(height)
            # PaddleOCR confidence is 0-1, convert to 0-100 for Presidio
            confidence = float(word_data['conf'])
            confidence = max(0.0, min(confidence * 100.0, 100.0))
            result["conf"].append(confidence)

        return result
