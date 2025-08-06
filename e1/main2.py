from flask import Flask, request, jsonify, send_file
import io
import base64
from PIL import Image
import numpy as np
import cv2

from e1.main import translate_image_from_cv2

app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        tgt_lang = request.form.get("tgt_lang", "ZH")

        # 将上传的文件读取成 OpenCV 图像格式
        image_stream = file.read()
        image_array = np.frombuffer(image_stream, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 调用翻译函数并返回结果图像（注意这里改 translate_image 返回 PIL.Image）
        translated_image = translate_image_from_cv2(image, tgt_lang)

        # 把 PIL.Image 保存成字节流（PNG）
        img_io = io.BytesIO()
        translated_image.save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
