from PIL import Image, ImageDraw, ImageFont
import paddleocr
import requests
import cv2
import numpy as np
import re


def clean_text(text):
    text = text.strip()
    text = re.sub(r"[·•\-–—]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
font_path = "C:\\Windows\\Fonts\\simhei.ttf"  # 这是一个例子
def translate_text_batch(texts, target_lang="ZH"):
    cleaned_texts = [clean_text(t) for t in texts]
    combined_text = "|||".join(cleaned_texts)

    url = "http://120.204.73.73:8033/api/ai-gateway/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 1743430591161rKvSr20mz35ew3p2RJMK8Xg"
    }
    payload = {
        "sessionId": "86d53fa25980411588251e70a6199663",
        "model": "Qwen-long",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"请将以下内容翻译成{target_lang}，每段用|||分隔，保持一一对应：\n{combined_text}"
                    }
                ]
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        reply = response.json()
        translated = reply["choices"][0]["message"]["content"].strip()
        lines = [line.strip() for line in translated.split("|||")]
        while len(lines) < len(texts):
            lines.append("")
        return lines[:len(texts)]
    except Exception as e:
        print(f"⚠️ 批量翻译失败: {e}")
        return texts

def get_text_width(text, font):
    bbox = font.getmask(text).getbbox()
    return bbox[2] - bbox[0] if bbox else 0



def fit_font_size(text, font_path, max_width, max_height, max_lines=3):
    for font_size in range(40, 10, -2):
        font = ImageFont.truetype(font_path, font_size)
        lines = []
        line = ""
        for ch in text:
            if get_text_width(line + ch, font) <= max_width:
                line += ch
            else:
                lines.append(line)
                line = ch
        if line:
            lines.append(line)
        if len(lines) > max_lines:
            continue
        ascent, descent = font.getmetrics()
        total_height = len(lines) * int((ascent + descent) * 1.3)
        if total_height <= max_height:
            return font, lines
    return ImageFont.truetype(font_path, 10), [text]

def fit_font_size2(text, font_path, max_width, max_height, max_font_size=30, min_font_size=10):
    font_size = max_font_size
    while font_size >= min_font_size:
        font = ImageFont.truetype(font_path, font_size)
        lines = []
        line = ""
        for ch in text:
            if get_text_width(line + ch, font) <= max_width:
                line += ch
            else:
                lines.append(line)
                line = ch
        if line:
            lines.append(line)

        ascent, descent = font.getmetrics()
        total_height = len(lines) * (ascent + descent) * 1.2
        if total_height <= max_height:
            return font, lines
        font_size -= 1

    font = ImageFont.truetype(font_path, min_font_size)
    return font, [text]
def is_white_bg(image_np, box):
    x1 = int(min([pt[0] for pt in box]))
    y1 = int(min([pt[1] for pt in box]))
    x2 = int(max([pt[0] for pt in box]))
    y2 = int(max([pt[1] for pt in box]))
    patch = image_np[y1:y2, x1:x2]
    if patch.size == 0:
        return True
    mean_color = patch.mean(axis=(0, 1))
    return np.all(mean_color > 240)

def is_chinese(text):
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)

def is_white_background(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > 225

def wrap_text(text, font, max_width):
    lines = []
    line = ''
    for char in text:
        test_line = line + char
        width = font.getbbox(test_line)[2]
        if width <= max_width:
            line = test_line
        else:
            if line:
                lines.append(line)
            line = char
    if line:
        lines.append(line)
    return lines



def is_black_white_background(cv_image, black_thresh=60, white_thresh=200, max_non_bw_ratio=0.15):
    """
    判断图像是否是黑白为主的背景图，允许少量彩色像素。

    参数:
        black_thresh: 小于该灰度值认为是黑色
        white_thresh: 大于该灰度值认为是白色
        max_non_bw_ratio: 允许非黑白像素最大比例
    """
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size

    # 黑白判定
    is_black = gray < black_thresh
    is_white = gray > white_thresh

    bw_mask = is_black | is_white
    non_bw_pixels = total_pixels - np.count_nonzero(bw_mask)
    non_bw_ratio = non_bw_pixels / total_pixels

    return non_bw_ratio <= max_non_bw_ratio
