
from PIL import Image, ImageDraw, ImageFont
import paddleocr
import requests
import cv2
import numpy as np
import re

from paddleocr import PaddleOCR

from e1.parameter import is_black_white_background
from e1.scene import translate_scene1, translate_scene2

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.

def translate_image3(image_path, output_path, tgt_lang):
    # 读取图像（用于背景判断）
    original_cv = cv2.imread(image_path)
    height, width = original_cv.shape[:2]

    if is_black_white_background(original_cv):
        translate_scene1(image_path, output_path, tgt_lang)
    else:
        translate_scene2(original_cv, output_path, tgt_lang)

# tgt目标语言，src扫描语言
if __name__ == "__main__":
    # 扫描的语言
    src_land="german"

    # 目标语言
    tgt_lang="zh"
    ocr = PaddleOCR(use_angle_cls=True, lang=src_land)  # need to run only once to download and load model into memory

    input_image = "german.png"
    output_image = "translated_image6.jpg"
    translate_image3(input_image, output_image,tgt_lang)