import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from paddleocr import PaddleOCR
from PIL import Image
import pytesseract
import hanlp

# 加载可用的中文语义相似度模型
similarity_model = hanlp.load('STS_ELECTRA_BASE_ZH')

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置 Tesseract 路径（根据实际安装路径修改）
pytesseract.pytesseract.tesseract_cmd = r'D:/Program Files/Tesseract-OCR/tesseract.exe'


@app.route('/Semantic_matching', methods=['POST'])
def semantic_matching():
    """语义匹配接口"""
    data = request.json
    texta = data.get('texta')
    textb = data.get('textb')
    if not texta or not textb:
        return jsonify({"message": "请输入要比较的文本！"}), 400

    similarity = similarity_model((texta, textb))
    return jsonify({"similarity": similarity}), 200





@app.route('/ocr', methods=['POST'])
def ocr_service():
    """OCR 处理接口"""
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # 保存文件到临时路径
        temp_path = os.path.join("/tmp", file.filename)
        file.save(temp_path)

        # 根据模型和语言初始化
        model = request.form.get('model')
        language = request.form.get('language')
        input_type = request.form.get('inputType')

        if model == "PaddleOCR":
            ocr = PaddleOCR(show_log=False, use_angle_cls=True, lang='ch' if language == "中文" else 'en')
        elif model == "Tesseract":
            ocr = None
        else:
            return jsonify({"message": "无效的 OCR 模型！"}), 400

        results = []

        try:
            if input_type == "单个图片":
                if os.path.isfile(temp_path):
                    results.append(process_image(temp_path, model, ocr, language))
                else:
                    return jsonify({"message": "输入的单个图片路径无效！"}), 400
            else:
                return jsonify({"message": "无效的输入类型！"}), 400

            return jsonify({"message": "OCR 处理完成！", "results": results}), 200
        except Exception as e:
            return jsonify({"message": f"处理过程中出错: {str(e)}"}), 500
    else:
        return jsonify({"message": "Invalid file type"}), 400


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


def process_image(input_path, model, ocr, language):
    # 假设这是一个简单的OCR处理函数
    return {"content": "This is the OCR result"}



def process_image(file_path, model, ocr, language):
    """处理单张图片"""
    if model == "PaddleOCR":
        result = ocr.ocr(file_path, cls=True)
        content = '\n'.join([str(line[1][0]) for line in result[0]])
    elif model == "Tesseract":
        image = Image.open(file_path)
        content = pytesseract.image_to_string(image, lang='chi_sim' if language == "中文" else 'eng')
    else:
        content = "未识别的模型"
    return {"image_path": file_path, "content": content}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)