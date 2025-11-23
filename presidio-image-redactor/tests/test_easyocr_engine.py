"""Tests for EasyOCREngine."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image


class TestEasyOCREngine:
    """Test suite for EasyOCREngine class."""

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_initialization_with_default_params(self, mock_easyocr):
        """Test EasyOCREngine initializes with default parameters."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        mock_reader = Mock()
        mock_easyocr.Reader.return_value = mock_reader

        engine = EasyOCREngine()

        # Verify EasyOCR Reader was initialized with correct defaults
        mock_easyocr.Reader.assert_called_once_with(
            lang_list=['en'],
            gpu=False,
            verbose=False,
            detect_network='craft',
        )
        assert engine.lang_list == ['en']

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_initialization_with_custom_params(self, mock_easyocr):
        """Test EasyOCREngine initializes with custom parameters."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        mock_reader = Mock()
        mock_easyocr.Reader.return_value = mock_reader

        engine = EasyOCREngine(
            lang_list=['ch_tra', 'en'],
            gpu=True,
            verbose=True
        )

        # Verify EasyOCR Reader was initialized with custom parameters
        mock_easyocr.Reader.assert_called_once_with(
            lang_list=['ch_tra', 'en'],
            gpu=True,
            verbose=True,
            detect_network='craft',
        )
        assert engine.lang_list == ['ch_tra', 'en']

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_initialization_with_model_dir_and_extra_kwargs(self, mock_easyocr):
        """Ensure optional model directory and kwargs propagate to EasyOCR."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        mock_reader = Mock()
        mock_easyocr.Reader.return_value = mock_reader

        EasyOCREngine(
            model_storage_directory='/tmp/models',
            easyocr_kwargs={"quantize": False},
        )

        mock_easyocr.Reader.assert_called_once_with(
            lang_list=['en'],
            gpu=False,
            verbose=False,
            detect_network='craft',
            model_storage_directory='/tmp/models',
            quantize=False,
        )

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_perform_ocr_with_single_word_line(self, mock_easyocr):
        """Test OCR with a single word per line."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        # Mock EasyOCR response for single word
        # EasyOCR format: [(bbox, text, conf), ...]
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[10, 20], [100, 20], [100, 50], [10, 50]], 'Hello', 0.95),
        ]
        mock_easyocr.Reader.return_value = mock_reader

        engine = EasyOCREngine()
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        result = engine.perform_ocr(image)

        # Verify result format
        assert 'text' in result
        assert 'left' in result
        assert 'top' in result
        assert 'width' in result
        assert 'height' in result
        assert 'conf' in result

        # Verify values
        assert result['text'] == ['Hello']
        assert result['left'] == [10]
        assert result['top'] == [20]
        assert result['width'] == [90]  # 100 - 10
        assert result['height'] == [30]  # 50 - 20
        assert result['conf'] == [95.0]

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_perform_ocr_passes_extra_kwargs(self, mock_easyocr):
        """Ensure extra kwargs reach EasyOCR.readtext."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], 'A', 0.5),
        ]
        mock_easyocr.Reader.return_value = mock_reader

        engine = EasyOCREngine()
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        engine.perform_ocr(image, paragraph=True, min_size=10)

        call_kwargs = mock_reader.readtext.call_args[1]
        assert call_kwargs['paragraph'] is True
        assert call_kwargs['min_size'] == 10
        assert call_kwargs['detail'] == 1  # Default

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_perform_ocr_with_multiple_words_in_line(self, mock_easyocr):
        """Test OCR with multiple words in a single line."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        # Mock EasyOCR response for multiple words
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[10, 20], [200, 20], [200, 50], [10, 50]], 'Hello World', 0.92),
        ]
        mock_easyocr.Reader.return_value = mock_reader

        engine = EasyOCREngine()
        image = np.zeros((100, 300, 3), dtype=np.uint8)

        result = engine.perform_ocr(image)

        # Verify result has two words
        assert len(result['text']) == 2
        assert result['text'] == ['Hello', 'World']
        assert len(result['left']) == 2
        assert len(result['conf']) == 2
        assert all(conf == 92.0 for conf in result['conf'])
        # Bboxes should stay within original bounds (10..200)
        assert result['left'][0] == 10
        assert result['left'][1] >= result['left'][0]
        assert result['left'][1] + result['width'][1] <= 200

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_perform_ocr_with_multiple_lines(self, mock_easyocr):
        """Test OCR with multiple lines of text."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        # Mock EasyOCR response for multiple lines
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[10, 20], [100, 20], [100, 50], [10, 50]], 'Hello', 0.95),
            ([[10, 60], [150, 60], [150, 90], [10, 90]], 'John Doe', 0.88),
        ]
        mock_easyocr.Reader.return_value = mock_reader

        engine = EasyOCREngine()
        image = np.zeros((150, 200, 3), dtype=np.uint8)

        result = engine.perform_ocr(image)

        # Verify result has three words (Hello + John + Doe)
        assert len(result['text']) == 3
        assert result['text'] == ['Hello', 'John', 'Doe']
        assert len(result['left']) == 3
        assert len(result['conf']) == 3

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_perform_ocr_with_empty_result(self, mock_easyocr):
        """Test OCR when no text is detected."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        # Mock empty EasyOCR response
        mock_reader = Mock()
        mock_reader.readtext.return_value = []
        mock_easyocr.Reader.return_value = mock_reader

        engine = EasyOCREngine()
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        result = engine.perform_ocr(image)

        # Verify empty result structure
        assert result['text'] == []
        assert result['left'] == []
        assert result['top'] == []
        assert result['width'] == []
        assert result['height'] == []
        assert result['conf'] == []

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_perform_ocr_with_pil_image(self, mock_easyocr):
        """Test OCR with PIL Image input."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[10, 20], [100, 20], [100, 50], [10, 50]], 'Test', 0.9),
        ]
        mock_easyocr.Reader.return_value = mock_reader

        engine = EasyOCREngine()
        image = Image.new('RGB', (200, 100), color='white')

        result = engine.perform_ocr(image)

        # Verify OCR was called and result is correct
        assert mock_reader.readtext.called
        assert result['text'] == ['Test']

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_perform_ocr_with_traditional_chinese_text(self, mock_easyocr):
        """Test OCR with Traditional Chinese text (no spaces)."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        # Mock EasyOCR response for Chinese text
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[10, 20], [150, 20], [150, 50], [10, 50]], '你好世界', 0.93),
        ]
        mock_easyocr.Reader.return_value = mock_reader

        engine = EasyOCREngine(lang_list=['ch_tra', 'en'])
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        result = engine.perform_ocr(image)

        # Chinese text without spaces should be treated as one word
        assert len(result['text']) == 1
        assert result['text'] == ['你好世界']
        assert result['conf'] == [93.0]

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_get_text_from_ocr_dict_with_empty_dict(self, mock_easyocr):
        """Test get_text_from_ocr_dict with empty dictionary."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        mock_easyocr.Reader.return_value = Mock()

        engine = EasyOCREngine()
        ocr_result = {}

        text = engine.get_text_from_ocr_dict(ocr_result)

        assert text == ""

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_get_text_from_ocr_dict_with_valid_dict(self, mock_easyocr):
        """Test get_text_from_ocr_dict with valid dictionary."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        mock_easyocr.Reader.return_value = Mock()

        engine = EasyOCREngine()
        ocr_result = {
            "text": ["Hello", "World", "Test"],
            "left": [10, 100, 200],
            "top": [20, 20, 20],
            "width": [80, 90, 70],
            "height": [30, 30, 30],
            "conf": [0.9, 0.85, 0.95]
        }

        text = engine.get_text_from_ocr_dict(ocr_result)

        assert text == "Hello World Test"

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_get_text_from_ocr_dict_with_custom_separator(self, mock_easyocr):
        """Test get_text_from_ocr_dict with custom separator."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        mock_easyocr.Reader.return_value = Mock()

        engine = EasyOCREngine()
        ocr_result = {
            "text": ["Hello", "World"],
            "left": [10, 100],
            "top": [20, 20],
            "width": [80, 90],
            "height": [30, 30],
            "conf": [0.9, 0.85]
        }

        text = engine.get_text_from_ocr_dict(ocr_result, separator="+")

        assert text == "Hello+World"

    def test_import_error_when_easyocr_not_installed(self):
        """Test that ImportError is raised when EasyOCR is not installed."""
        with patch('presidio_image_redactor.easyocr_engine.easyocr', None):
            from presidio_image_redactor.easyocr_engine import EasyOCREngine

            with pytest.raises(ImportError, match="EasyOCR is not installed"):
                EasyOCREngine()

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_polygon_to_bbox_conversion(self, mock_easyocr):
        """Test conversion of polygon coordinates to bbox format."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        mock_easyocr.Reader.return_value = Mock()

        engine = EasyOCREngine()

        # Test with a simple rectangle
        polygon = [[10, 20], [100, 20], [100, 50], [10, 50]]
        left, top, width, height = engine._polygon_to_bbox(polygon)

        assert left == 10
        assert top == 20
        assert width == 90
        assert height == 30

    @patch('presidio_image_redactor.easyocr_engine.easyocr')
    def test_polygon_to_bbox_with_rotated_box(self, mock_easyocr):
        """Test polygon to bbox with rotated/skewed box."""
        from presidio_image_redactor.easyocr_engine import EasyOCREngine

        mock_easyocr.Reader.return_value = Mock()

        engine = EasyOCREngine()

        # Test with a rotated box (should still return bounding rectangle)
        polygon = [[15, 10], [105, 15], [100, 55], [10, 50]]
        left, top, width, height = engine._polygon_to_bbox(polygon)

        assert left == 10
        assert top == 10
        assert width == 95
        assert height == 45
