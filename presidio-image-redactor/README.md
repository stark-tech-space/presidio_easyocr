# Presidio Image Redactor

## Description

The Presidio Image Redactor is a Python based module for detecting and redacting PII text entities in images.

## Deploy Presidio image redactor to Azure

Use the following button to deploy presidio image redactor to your Azure subscription.

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmicrosoft%2Fpresidio%2Fmain%2Fpresidio-image-redactor%2Fdeploytoazure.json)

Process for standard images:

![Image Redactor Design](../docs/assets/image-redactor-design.png)

Process for DICOM files:

![DICOM image Redactor Design](../docs/assets/dicom-image-redactor-design.png)

## Installation

Pre-requisites:

- Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) by following the
  instructions on how to install it for your operating system.

  For best performance, please use the most up-to-date version of Tesseract OCR. Presidio was tested with **v5.2.0**.

### As package

To get started with Presidio-image-redactor, run the following:

```sh
pip install presidio-image-redactor
```

Once Installed, run the following command to download the default spacy model needed for
Presidio Analyzer:

```sh
python -m spacy download en_core_web_lg
```

## OCR Engines

Presidio Image Redactor supports multiple OCR engines for text detection and recognition. You can choose the engine that best fits your needs:

### Tesseract OCR (Default)

By default, Presidio uses Tesseract OCR. No additional configuration is needed if you've installed Tesseract following the installation instructions above.

```python
from presidio_image_redactor import ImageRedactorEngine

# Uses Tesseract OCR by default
engine = ImageRedactorEngine()
```

### EasyOCR (Recommended for Chinese and Asian languages)

EasyOCR offers good accuracy for Chinese (traditional and simplified), Japanese, Korean, and many other languages. It supports over 80 languages and provides a simple API.

**Installation:**

```sh
pip install presidio-image-redactor[easyocr]
```

**Usage:**

```python
from PIL import Image
from presidio_image_redactor import ImageRedactorEngine, ImageAnalyzerEngine
from presidio_image_redactor import EasyOCREngine

# Create EasyOCR engine for Traditional Chinese + English
easyocr_engine = EasyOCREngine(
    lang_list=['ch_tra', 'en'],  # Traditional Chinese + English
    gpu=False,                    # Set True if you have GPU
)

# Create Image Analyzer with EasyOCR
image_analyzer = ImageAnalyzerEngine(ocr=easyocr_engine)

# Create Redactor Engine
engine = ImageRedactorEngine(image_analyzer_engine=image_analyzer)

# Use as normal
image = Image.open("test.png")
redacted_image = engine.redact(image)
```

**Supported languages:** EasyOCR supports 80+ languages including:
- `en` - English
- `ch_tra` - Traditional Chinese
- `ch_sim` - Simplified Chinese
- `ja` - Japanese
- `ko` - Korean
- And many more (see [EasyOCR supported languages](https://www.jaided.ai/easyocr/))

**GPU Support:** EasyOCR uses PyTorch. If you have a CUDA-compatible GPU, install the GPU version of PyTorch and set `gpu=True`.

### Azure Document Intelligence OCR

For cloud-based OCR with high accuracy, you can use Azure's Document Intelligence service:

```python
from presidio_image_redactor import ImageRedactorEngine, ImageAnalyzerEngine
from presidio_image_redactor import DocumentIntelligenceOCR
import os

# Set up Azure credentials
os.environ["DOCUMENT_INTELLIGENCE_ENDPOINT"] = "your-endpoint"
os.environ["DOCUMENT_INTELLIGENCE_KEY"] = "your-key"

# Create Azure DI OCR engine
azure_ocr = DocumentIntelligenceOCR()

# Create Image Analyzer with Azure OCR
image_analyzer = ImageAnalyzerEngine(ocr=azure_ocr)

# Create Redactor Engine
engine = ImageRedactorEngine(image_analyzer_engine=image_analyzer)
```

### Choosing the Right OCR Engine

| Engine | Best For | Pros | Cons |
|--------|----------|------|------|
| **Tesseract** | General purpose, English text | Free, well-established, no cloud dependency | Lower accuracy for Asian languages |
| **EasyOCR** | Chinese, Japanese, Korean, Asian languages | Good accuracy for Asian languages, easy to use, 80+ languages | Requires PyTorch |
| **Azure DI** | Enterprise applications, high accuracy needs | Very high accuracy, cloud-backed | Requires Azure subscription, API costs |

## Getting started (standard image types)

The engine will receive 2 parameters:

1. Image to redact.
2. Color fill to redact with, by default color fill will be black. Can either be an int
   or tuple (0,0,0)

```python
from PIL import Image
from presidio_image_redactor import ImageRedactorEngine

# Get the image to redact using PIL lib (pillow)
image = Image.open("presidio-image-redactor/tests/integration/resources/ocr_test.png")

# Initialize the engine
engine = ImageRedactorEngine()

# Redact the image with pink color
redacted_image = engine.redact(image, (255, 192, 203))

# save the redacted image 
redacted_image.save("new_image.png")
# uncomment to open the image for viewing
# redacted_image.show()
```

### As docker service

In folder presidio/presidio-image-redactor run:

```
docker-compose up -d
```

### HTTP API

### redact

Receives an image and color fill (optional, default is black). Redact the image PII text
and returns a new redacted image.

```
POST /redact
```

Payload:

Sent as multipart-form. Contains image file and data of the required color fill.

```json
{
  "data": "{'color_fill':'0,0,0'}"
}
```

Result:

```
200 OK
```

curl example:

```
# use ocr_test.png as the image to redact, and 255 as the color fill. 
# out.png is the new redacted image received from the server.
curl -XPOST "http://localhost:3000/redact" -H "content-type: multipart/form-data" -F "image=@ocr_test.png" -F "data=\"{'color_fill':'255'}\"" > out.png
```

Python script example can be found under:
/presidio/e2e-tests/tests/test_image_redactor.py

## Getting started (DICOM images)

This module only redacts pixel data and does not scrub text PHI which may exist in the DICOM metadata.

We highly recommend using the DICOM image redactor engine to redact text from images **before** scrubbing metadata PHI. To redact sensitive information from metadata, consider using another package such as the [Tools for Health Data Anonymization](https://github.com/microsoft/Tools-for-Health-Data-Anonymization).

To redact burnt-in text PHI in DICOM images, see the below sample code:

```python
import pydicom
from presidio_image_redactor import DicomImageRedactorEngine

# Set input and output paths
input_path = "path/to/your/dicom/file.dcm"
output_dir = "./output"

# Initialize the engine
engine = DicomImageRedactorEngine()

# Option 1: Redact from a loaded DICOM image
dicom_image = pydicom.dcmread(input_path)
redacted_dicom_image = engine.redact(dicom_image, fill="contrast")

# Option 2: Redact from a loaded DICOM image and return redacted regions
redacted_dicom_image, bboxes = engine.redact_and_return_bbox(dicom_image, fill="contrast")

# Option 3: Redact from DICOM file and save redacted regions as json file
engine.redact_from_file(input_path, output_dir, padding_width=25, fill="contrast", save_bboxes=True)

# Option 4: Redact from directory and save redacted regions as json files
ocr_kwargs = {"ocr_threshold": 50}
engine.redact_from_directory("path/to/your/dicom", output_dir, fill="background", save_bboxes=True, ocr_kwargs=ocr_kwargs)
```

See the example notebook for more details and visual confirmation of the output: [docs/samples/python/example_dicom_image_redactor.ipynb](../docs/samples/python/example_dicom_image_redactor.ipynb).

### Side note for Windows

If you are using a Windows machine, you may run into issues if file paths are too long. Unfortunately, this is not rare when working with DICOM images that are often nested in directories with descriptive names.

To avoid errors where the code may not recognize a path as existing due to the length of the characters in the file path, please [enable long paths on your system](https://learn.microsoft.com/en-us/answers/questions/293227/longpathsenabled.html).

### DICOM Data Citation

The DICOM data used for unit and integration testing for `DicomImageRedactorEngine` are stored in this repository with permission from the original dataset owners. Please see the dataset information as follows:

> Rutherford, M., Mun, S.K., Levine, B., Bennett, W.C., Smith, K., Farmer, P., Jarosz, J., Wagner, U., Farahani, K., Prior, F. (2021). A DICOM dataset for evaluation of medical image de-identification (Pseudo-PHI-DICOM-Data) [Data set]. The Cancer Imaging Archive. DOI: <https://doi.org/10.7937/s17z-r072>
