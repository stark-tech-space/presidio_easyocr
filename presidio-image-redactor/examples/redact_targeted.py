#!/usr/bin/env python3
"""
只遮罩特定 PII 类型：人名、生日、证件号、手机号
"""

from PIL import Image
from presidio_image_redactor import EasyOCREngine, ImageAnalyzerEngine, ImageRedactorEngine
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider


def create_analyzer_with_tw_recognizers():
    """创建支持台湾 PII 格式的 Analyzer"""

    # 1. 配置中文 NLP 引擎
    configuration = {
        'nlp_engine_name': 'spacy',
        'models': [
            {'lang_code': 'zh', 'model_name': 'zh_core_web_trf'},
            {'lang_code': 'en', 'model_name': 'en_core_web_lg'},
        ],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # 2. 创建台湾身份证号识别器
    tw_id_recognizer = PatternRecognizer(
        supported_entity='TW_NATIONAL_ID',
        patterns=[
            Pattern('tw_id_full', r'[A-Z][12][0-9]{8}', 0.9),           # A123456789
            Pattern('tw_id_masked', r'[A-Z][12][0-9]{2}\*+[0-9]{2}', 0.9),  # A12****89
        ],
        supported_language='zh',
    )

    # 3. 创建台湾电话识别器
    tw_phone_recognizer = PatternRecognizer(
        supported_entity='TW_PHONE_NUMBER',
        patterns=[
            Pattern('tw_mobile', r'09[0-9]{2}[-\s]?[0-9]{3}[-\s]?[0-9]{3}', 0.85),  # 0912-345-678
            Pattern('tw_landline', r'\(?0[0-9]{1,2}\)?[-\s]?[0-9]{3,4}[-\s]?[0-9]{4}', 0.7),  # (04)24738595
        ],
        supported_language='zh',
    )

    # 4. 创建生日识别器
    birthday_recognizer = PatternRecognizer(
        supported_entity='BIRTHDAY',
        patterns=[
            Pattern('birthday_roc', r'[0-9]{2,3}/[0-9]{1,2}/[0-9]{1,2}', 0.7),  # 39/08/24 民国年
            Pattern('birthday_ad', r'[12][90][0-9]{2}/[0-9]{1,2}/[0-9]{1,2}', 0.8),  # 1990/08/24
        ],
        supported_language='zh',
    )

    # 5. 创建 Analyzer 并添加自定义识别器
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=['zh', 'en']
    )
    analyzer.registry.add_recognizer(tw_id_recognizer)
    analyzer.registry.add_recognizer(tw_phone_recognizer)
    analyzer.registry.add_recognizer(birthday_recognizer)

    return analyzer


def redact_image(image_path: str, output_path: str):
    """对图片进行 PII 遮罩"""

    print('初始化引擎...')
    analyzer = create_analyzer_with_tw_recognizers()
    ocr = EasyOCREngine(lang_list=['ch_tra', 'en'], gpu=False, verbose=False)
    image_analyzer = ImageAnalyzerEngine(analyzer_engine=analyzer, ocr=ocr)
    redactor = ImageRedactorEngine(image_analyzer_engine=image_analyzer)

    # 只检测这些实体类型
    target_entities = [
        'PERSON',           # 人名 (spaCy NER)
        'TW_NATIONAL_ID',   # 台湾身份证号
        'TW_PHONE_NUMBER',  # 台湾电话/手机
        'BIRTHDAY',         # 生日
    ]

    print(f'加载图片: {image_path}')
    image = Image.open(image_path)

    print(f'检测实体类型: {target_entities}')
    redacted, bboxes = redactor.redact_and_return_bbox(
        image,
        fill=(0, 0, 0),  # 黑色遮罩
        entities=target_entities,
        language='zh',
    )

    print(f'\n检测到 {len(bboxes)} 个 PII 区域:')
    for bbox in bboxes:
        print(f'  - {bbox}')

    redacted.save(output_path)
    print(f'\n保存到: {output_path}')
    return redacted, bboxes


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'redacted_output.png'
    else:
        input_path = '../tests/samples/處方箋1（去識別化）.png'
        output_path = '../tests/samples/redacted_targeted.png'

    redact_image(input_path, output_path)
