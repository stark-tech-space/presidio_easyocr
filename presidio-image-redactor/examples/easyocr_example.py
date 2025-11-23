#!/usr/bin/env python3
"""
Example script demonstrating how to use EasyOCR with Presidio Image Redactor.

This example shows:
1. How to use EasyOCR for English text
2. How to use EasyOCR for Traditional Chinese + English
3. How to compare results with Tesseract OCR
4. How to run EasyOCR on GPU
5. How to process multilingual documents
"""

# Import Presidio components
from presidio_image_redactor import (
    ImageRedactorEngine,
    ImageAnalyzerEngine,
    EasyOCREngine,
    TesseractOCR
)


def example_english_text():
    """Example using EasyOCR for English text."""
    print("=" * 60)
    print("Example 1: English Text with EasyOCR")
    print("=" * 60)

    # Create EasyOCR engine for English
    easyocr_engine = EasyOCREngine(lang_list=['en'])

    # Create Image Analyzer with EasyOCR
    image_analyzer = ImageAnalyzerEngine(ocr=easyocr_engine)

    # Create Redactor Engine
    engine = ImageRedactorEngine(image_analyzer_engine=image_analyzer)

    # Create or load an image
    # For this example, you would use: image = Image.open("your_image.png")
    print("Engine initialized successfully with EasyOCR (English)")
    print()


def example_traditional_chinese():
    """Example using EasyOCR for Traditional Chinese + English text."""
    print("=" * 60)
    print("Example 2: Traditional Chinese + English with EasyOCR")
    print("=" * 60)

    # Create EasyOCR engine for Traditional Chinese + English
    easyocr_engine = EasyOCREngine(lang_list=['ch_tra', 'en'])

    # Create Image Analyzer with EasyOCR
    image_analyzer = ImageAnalyzerEngine(ocr=easyocr_engine)

    # Create Redactor Engine
    engine = ImageRedactorEngine(image_analyzer_engine=image_analyzer)

    # Create or load an image
    # For this example, you would use: image = Image.open("chinese_document.png")
    print("Engine initialized successfully with EasyOCR (Traditional Chinese + English)")
    print()


def example_with_gpu():
    """Example using EasyOCR with GPU acceleration."""
    print("=" * 60)
    print("Example 3: EasyOCR with GPU Acceleration")
    print("=" * 60)

    try:
        # Create EasyOCR engine with GPU enabled
        easyocr_engine = EasyOCREngine(
            lang_list=['ch_tra', 'en'],
            gpu=True,
        )

        # Create Image Analyzer with EasyOCR
        image_analyzer = ImageAnalyzerEngine(ocr=easyocr_engine)

        # Create Redactor Engine
        engine = ImageRedactorEngine(image_analyzer_engine=image_analyzer)

        print("Engine initialized successfully with GPU acceleration")
    except Exception as e:
        print(f"GPU initialization failed (this is normal if no GPU available): {e}")
    print()


def example_comparison():
    """Example comparing EasyOCR with Tesseract."""
    print("=" * 60)
    print("Example 4: Comparing EasyOCR vs Tesseract")
    print("=" * 60)

    # Test image with both OCR engines
    # For a real comparison, you would load an actual image

    print("1. Using Tesseract OCR:")
    tesseract_ocr = TesseractOCR()
    analyzer_tesseract = ImageAnalyzerEngine(ocr=tesseract_ocr)
    engine_tesseract = ImageRedactorEngine(image_analyzer_engine=analyzer_tesseract)
    print("   - Tesseract engine initialized")

    print("\n2. Using EasyOCR:")
    easyocr_engine = EasyOCREngine(lang_list=['en'])
    analyzer_easyocr = ImageAnalyzerEngine(ocr=easyocr_engine)
    engine_easyocr = ImageRedactorEngine(image_analyzer_engine=analyzer_easyocr)
    print("   - EasyOCR engine initialized")

    print("\nBoth engines are ready for comparison!")
    print("Tip: EasyOCR typically performs better on:")
    print("  - Chinese/Japanese/Korean text")
    print("  - Mixed language documents")
    print("  - Handwritten text (in supported languages)")
    print()


def example_multilingual():
    """Example using EasyOCR for multilingual documents."""
    print("=" * 60)
    print("Example 5: Multilingual Document Processing")
    print("=" * 60)

    supported_languages = {
        'en': 'English',
        'ch_tra': 'Traditional Chinese',
        'ch_sim': 'Simplified Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'es': 'Spanish',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'th': 'Thai',
        'vi': 'Vietnamese',
    }

    print("EasyOCR supports 80+ languages. Here are some examples:")
    for lang_code, lang_name in supported_languages.items():
        print(f"  - {lang_name}: '{lang_code}'")

    print("\nExample initialization for Japanese + English:")
    easyocr_engine = EasyOCREngine(lang_list=['ja', 'en'])
    print("  EasyOCR initialized for Japanese + English text")
    print()


def example_custom_settings():
    """Example showing how to configure EasyOCR settings."""
    print("=" * 60)
    print("Example 6: Custom EasyOCR Settings")
    print("=" * 60)

    easyocr_engine = EasyOCREngine(
        lang_list=['ch_tra', 'en'],
        gpu=False,
        model_storage_directory='/tmp/easyocr_models',
        detect_network='craft',  # or 'dbnet18'
        verbose=True,
    )
    print("Custom EasyOCR settings configured successfully")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("EasyOCR Integration Examples for Presidio")
    print("=" * 60 + "\n")

    try:
        example_english_text()
        example_traditional_chinese()
        example_with_gpu()
        example_comparison()
        example_multilingual()
        example_custom_settings()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install EasyOCR: pip install presidio-image-redactor[easyocr]")
        print("2. Load your own images and test the redaction")
        print("3. Compare accuracy between different OCR engines")
        print("4. For production use, consider GPU acceleration with PyTorch CUDA")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install presidio-image-redactor[easyocr]")


if __name__ == "__main__":
    main()
