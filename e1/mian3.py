from PIL import Image, ImageDraw, ImageFont
import paddleocr
import requests
import re
import cv2
import numpy as np
ocr = paddleocr.PaddleOCR(use_textline_orientation=True, lang='ch')

def clean_text(text):
    # 去除杂点、空格等无效字符
    text = text.strip()
    text = re.sub(r"[·•\-–—]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

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

        if len(lines) != len(texts):
            print(f"⚠️ 翻译行数不一致！原始 {len(texts)} 行，返回 {len(lines)} 行，尝试对齐")
            # 补齐或截断
            while len(lines) < len(texts):
                lines.append("")
            lines = lines[:len(texts)]

        return lines
    except Exception as e:
        print(f"⚠️ 批量翻译失败: {e}")
        return texts  # 翻译失败时返回原文

def get_text_width(text, font):
    bbox = font.getmask(text).getbbox()
    return bbox[2] - bbox[0] if bbox else 0



def translate_image(image_path, output_path):
    font_path = "C:\\Windows\\Fonts\\simhei.ttf"

    # Step 1: OCR
    result = ocr.ocr(image_path, cls=True)
    original_texts = [line[1][0] for line in result[0]]
    translations = translate_text_batch(original_texts, target_lang="ZH")

    # Step 2: 读取原图和生成修复掩码
    image_cv = cv2.imread(image_path)
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)

    for line in result[0]:
        box = line[0]
        polygon = np.array([np.array(box, dtype=np.int32)])
        cv2.fillPoly(mask, polygon, 255)

    # Step 3: 修复图像
    inpainted = cv2.inpaint(image_cv, mask, 3, cv2.INPAINT_TELEA)

    # Step 4: 将修复后图像转为 PIL 图像并绘制翻译文字
    image_pil = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(image_pil, "RGBA")

    for idx, line in enumerate(result[0]):
        try:
            box, (orig_text, _) = line
            translated = translations[idx] or orig_text

            polygon = [tuple(map(int, point)) for point in box]
            min_x = min(p[0] for p in polygon)
            max_x = max(p[0] for p in polygon)
            min_y = min(p[1] for p in polygon)
            max_y = max(p[1] for p in polygon)
            box_width = max_x - min_x
            box_height = max_y - min_y

            padding = 2
            font, lines = fit_font_size(translated, font_path, box_width - 2 * padding, box_height - 2 * padding)

            ascent, descent = font.getmetrics()
            line_spacing = int((ascent + descent) * 1.2)

            x_start = min_x + padding
            y_start = min_y + padding

            for i, l in enumerate(lines):
                y = y_start + i * line_spacing
                draw.text((x_start + 1, y + 1), l, fill="gray", font=font)
                draw.text((x_start, y), l, fill="black", font=font)

        except Exception as e:
            print(f"⚠️ 行 {idx+1} 出错: {e}")
            continue

    image_pil.convert("RGB").save(output_path)
    print(f"✅ 翻译完成，输出文件：{output_path}")



if __name__ == "__main__":
    input_image = "data3.png"
    output_image = "translated_image6.jpg"
    translate_image(input_image, output_image)
