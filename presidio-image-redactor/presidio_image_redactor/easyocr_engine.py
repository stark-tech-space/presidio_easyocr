"""EasyOCR engine for Presidio Image Redactor.

This module provides an EasyOCR-based OCR engine as an alternative to Tesseract.
EasyOCR offers good accuracy for Chinese (traditional and simplified), English,
and many other languages.
"""

from typing import Any, List, Tuple, Union, Optional
import numpy as np
from PIL import Image

from presidio_image_redactor import OCR

try:  # pragma: no cover - optional dependency
    import easyocr
except ImportError:  # pragma: no cover - handled at runtime
    easyocr = None


class EasyOCREngine(OCR):
    """OCR engine that uses EasyOCR for text detection and recognition.

    EasyOCR provides good accuracy for Chinese, Japanese, Korean and many other
    languages. It supports both traditional and simplified Chinese.

    :param lang_list: List of language codes for OCR (e.g., ['en'], ['ch_tra', 'en'])
                      Default is ['en'] for English.
                      Use ['ch_tra', 'en'] for Traditional Chinese + English.
                      Use ['ch_sim', 'en'] for Simplified Chinese + English.
    :param gpu: Whether to use GPU acceleration. Default is False (CPU mode).
    :param model_storage_directory: Path to store downloaded models.
    :param detect_network: Detection network to use ('craft' or 'dbnet18').
                          Default is 'craft'.
    :param verbose: Whether to show EasyOCR logs. Default is False.
    :param easyocr_kwargs: Additional keyword arguments passed to easyocr.Reader.

    Example usage:
        >>> from PIL import Image
        >>> from presidio_image_redactor.easyocr_engine import EasyOCREngine
        >>>
        >>> # Create EasyOCR engine for Traditional Chinese + English
        >>> ocr = EasyOCREngine(lang_list=['ch_tra', 'en'])
        >>>
        >>> # Perform OCR
        >>> image = Image.open("document.png")
        >>> result = ocr.perform_ocr(image)
        >>> print(result['text'])  # List of detected words

    Note:
        EasyOCR returns line-level bounding boxes by default. This engine
        splits them into word-level boxes to match Presidio's expected format.
    """

    def __init__(
        self,
        lang_list: List[str] = None,
        gpu: bool = False,
        model_storage_directory: Optional[str] = None,
        detect_network: str = 'craft',
        verbose: bool = False,
        easyocr_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Initialize EasyOCR engine with specified parameters."""
        if easyocr is None:
            raise ImportError(
                "EasyOCR is not installed. Install it with 'pip install easyocr'"
            )

        if lang_list is None:
            lang_list = ['en']

        easyocr_init_kwargs: dict[str, Any] = {
            "lang_list": lang_list,
            "gpu": gpu,
            "verbose": verbose,
            "detect_network": detect_network,
        }

        if model_storage_directory:
            easyocr_init_kwargs["model_storage_directory"] = model_storage_directory
        if easyocr_kwargs:
            easyocr_init_kwargs.update(easyocr_kwargs)

        self.reader = easyocr.Reader(**easyocr_init_kwargs)
        self.lang_list = lang_list

    def perform_ocr(self, image: object, **kwargs) -> dict:
        """Perform OCR on a given image.

        This method converts EasyOCR's line-level output to word-level format
        compatible with Presidio's expected structure.

        :param image: PIL Image/numpy array or file path(str) to be processed
        :param kwargs: Additional OCR parameters passed to readtext()
                      (e.g., detail=1, paragraph=False)

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
        # Convert image to format EasyOCR expects
        img_array = self._prepare_image(image)

        # Perform OCR (returns line-level results)
        readtext_kwargs = dict(kwargs)
        if "detail" not in readtext_kwargs:
            readtext_kwargs["detail"] = 1  # Need full info with bboxes

        ocr_result = self.reader.readtext(img_array, **readtext_kwargs)

        # Handle empty results
        if not ocr_result:
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": []
            }

        # Convert line-level results to word-level
        word_level_data = self._convert_to_word_level(ocr_result)

        # Convert to Presidio format
        return self._to_presidio_format(word_level_data)

    def _prepare_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Convert various image formats to numpy array for EasyOCR.

        :param image: Input image in various formats
        :return: numpy array representation of the image
        """
        if isinstance(image, str):
            # File path - convert to PIL then numpy for consistency
            image = Image.open(image)

        if isinstance(image, Image.Image):
            # PIL Image - convert to numpy array
            image = np.array(image)

        if not isinstance(image, np.ndarray):
            raise ValueError(f"Unsupported image type: {type(image)}")

        return image

    def _convert_to_word_level(
        self,
        easyocr_result: List[Tuple]
    ) -> List[dict]:
        """Convert EasyOCR line-level output to word-level.

        EasyOCR returns line-level results. This method keeps languages without
        whitespace intact and splits space-separated languages into spans while keeping
        their relative offsets for accurate bounding boxes.

        :param easyocr_result: List of (bbox, text, confidence) tuples from EasyOCR
        :return: List of dictionaries with word-level bounding boxes and text
        """
        word_list = []

        for line_data in easyocr_result:
            # EasyOCR format: (bbox, text, conf) - different from PaddleOCR
            bbox, text, conf = line_data
            if not text:
                continue

            word_spans = self._split_text_into_spans(text)
            if not word_spans:
                continue

            if len(word_spans) == 1:
                word_list.append(
                    {
                        "bbox": bbox,
                        "text": word_spans[0]["text"],
                        "conf": conf,
                    }
                )
                continue

            word_bboxes = self._split_bbox_by_spans(bbox, word_spans, len(text))
            for span, word_bbox in zip(word_spans, word_bboxes):
                word_list.append(
                    {
                        "bbox": word_bbox,
                        "text": span["text"],
                        "conf": conf,
                    }
                )

        return word_list

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
                    spans.append({"text": text[start_idx:idx], "start": start_idx, "end": idx})
                    start_idx = None
                continue

            if start_idx is None:
                start_idx = idx

        if start_idx is not None:
            spans.append({"text": text[start_idx:], "start": start_idx, "end": len(text)})

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
            # EasyOCR confidence is 0-1, convert to 0-100 for Presidio
            confidence = float(word_data['conf'])
            confidence = max(0.0, min(confidence * 100.0, 100.0))
            result["conf"].append(confidence)

        return result
