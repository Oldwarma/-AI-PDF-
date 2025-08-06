from paddle.dataset.image import cv2
from e1.parameter import is_target_language, translate_text_batch, wrap_text, is_chinese, is_white_background
from e1.parameter import fit_font_size2
from e2.main import ocr
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


from paddleocr import PaddleOCR

def translate_scene1(image_path, output_path, tgt_lang, font_path="C:\\Windows\\Fonts\\simhei.ttf"):
    # 场景1：纯黑白背景整体翻译，重绘整张图
    original_cv = cv2.imread(image_path)
    height, width = original_cv.shape[:2]

    result = ocr.ocr(image_path, cls=True)
    original_texts = []
    for line in result[0]:
        _, (text, _) = line
        if is_target_language(text, languages=None):  # 不过滤，收集所有
            original_texts.append(text)

    if not original_texts:
        print("⚠️ 没有待翻译的文本")
        return

    translations = translate_text_batch(original_texts, tgt_lang)
    full_text = "".join(translations)

    max_width = width - 40
    x_padding = 20
    y_padding = 20
    min_font_size = 12
    max_font_size = 36
    best_font_size = min_font_size
    final_lines = []

    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = ImageFont.truetype(font_path, font_size)
        line_spacing = int(font_size * 1.5)
        wrapped_lines = wrap_text(full_text, font, max_width)
        total_height = line_spacing * len(wrapped_lines) + 2 * y_padding
        if total_height <= height:
            best_font_size = font_size
            final_lines = wrapped_lines
            break

    if not final_lines:
        font = ImageFont.truetype(font_path, min_font_size)
        line_spacing = int(min_font_size * 1.5)
        final_lines = wrap_text(full_text, font, max_width)

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, best_font_size)
    y = y_padding
    for line in final_lines:
        if y + line_spacing > height:
            break
        draw.text((x_padding, y), line, font=font, fill=(10, 10, 10))
        y += line_spacing

    image.save(output_path)
    print(f"✅ 翻译完成，已保存到: {output_path}")



def translate_scene2(image_cv, output_path, tgt_lang, font_path="C:\\Windows\\Fonts\\simhei.ttf"):
    # 场景2：复杂背景，擦除文字+翻译覆盖
    image = image_cv.copy()
    result = ocr.ocr(image, cls=True)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    original_texts = []

    for line in result[0]:
        box, (text, _) = line
        if not text.strip() or is_chinese(text):
            continue
        x_min, y_min = int(min(p[0] for p in box)), int(min(p[1] for p in box))
        x_max, y_max = int(max(p[0] for p in box)), int(max(p[1] for p in box))
        crop = image[y_min:y_max, x_min:x_max]
        if is_white_background(crop):
            continue
        cv2.fillPoly(mask, [np.array(box, dtype=np.int32)], 255)
        original_texts.append(text)

    # 膨胀掩码扩大擦除区域
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    image_pil = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    translations = translate_text_batch(original_texts, tgt_lang)

    draw_idx = 0
    for line in result[0]:
        box, (text, _) = line
        if not text.strip() or is_chinese(text):
            continue
        x_min, y_min = int(min(p[0] for p in box)), int(min(p[1] for p in box))
        x_max, y_max = int(max(p[0] for p in box)), int(max(p[1] for p in box))
        crop = image[y_min:y_max, x_min:x_max]
        if is_white_background(crop):
            continue

        translated = translations[draw_idx]
        draw_idx += 1

        box_width = x_max - x_min
        box_height = y_max - y_min

        font, lines = fit_font_size2(translated, font_path, box_width, box_height)

        ascent, descent = font.getmetrics()
        line_height = int((ascent + descent) * 1.2)

        for i, line_text in enumerate(lines):
            y = y_min + i * line_height
            # 阴影
            draw.text((x_min + 1, y + 1), line_text, font=font, fill=(0, 0, 0, 128))
            # 正文
            draw.text((x_min, y), line_text, font=font, fill=(10, 10, 10))

    image_pil.convert("RGB").save(output_path)
    print(f"✅ 翻译完成，已保存到: {output_path}")



def translate_scene3(image_cv, tgt_lang, font_path, url, auth, model, ocr):
    image = image_cv.copy()
    result = ocr.ocr(image, cls=True)

    boxes_texts = []
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    text_boxes = []
    for item in result[0]:
        if not item:  # 防御空内容
            continue
        box_info, text_info = item[0]  # 解包内部的实际内容
        box = box_info
        text, _ = text_info
        if is_target_language(text):
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            text_boxes.append((bbox, text))

    if not boxes_texts:
        print("⚠️ 没有可翻译文本")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 扩大 mask 防止残留
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    lines_group = group_text_boxes(boxes_texts)

    original_texts = [" ".join(t for _, t in line) for line in lines_group]
    all_boxes = [[box for box, _ in line] for line in lines_group]

    translations = translate_text_batch(original_texts, tgt_lang, url, auth, model)
    if translations is None:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for translated, boxes in zip(translations, all_boxes):
        x_min, y_min, x_max, y_max = get_merged_box(boxes)
        box_w, box_h = x_max - x_min, y_max - y_min

        # 初始字体大小
        font_size = min(box_h, 48)
        min_font_size = 12
        font = ImageFont.truetype(font_path, font_size)
        lines = wrap_text(translated, font, box_w)

        # 尝试缩小字体直到合适
        while font_size >= min_font_size:
            total_height = len(lines) * (font_size + 2)
            max_width = max([font.getlength(line) for line in lines]) if lines else 0

            if lines and total_height <= box_h and max_width <= box_w:
                break  # 满足要求
            font_size -= 2
            font = ImageFont.truetype(font_path, font_size)
            lines = wrap_text(translated, font, box_w)

        # 如果怎么都放不下，就强行最小字体绘制（可能溢出）
        if not lines:
            font_size = min_font_size
            font = ImageFont.truetype(font_path, font_size)
            lines = wrap_text(translated, font, box_w)

        # 最终兜底还为空，就放弃
        if not lines:
            print(f"⚠️ 无法绘制翻译：{translated}")
            continue

        # 居中排版（纵向居中）
        total_height = len(lines) * (font_size + 2)
        y_offset = y_min + max(0, (box_h - total_height) // 2)

        for line in lines:
            inpainted = draw_text_pil(inpainted, line, (x_min, y_offset), font_path, font_size)
            y_offset += font_size + 2

    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))