import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from paddleocr import PaddleOCR
from PIL import Image
import pytesseract
import csv

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置 Tesseract 路径（根据实际安装路径修改）
pytesseract.pytesseract.tesseract_cmd = r'D:/Program Files/Tesseract-OCR/tesseract.exe'

# 默认输出路径
OUTPUT_DIR = 'output'
OUTPUT_FILE = 'ocr_results.csv'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/ocr', methods=['POST'])
def ocr_service():
    data = request.json
    model = data.get('model')
    language = data.get('language')
    input_type = data.get('inputType')
    input_path = data.get('inputPath')

    # 验证路径是否存在
    if not os.path.exists(input_path):
        return jsonify({"message": "指定的路径不存在！"}), 400

    # 根据模型和语言初始化
    if model == "PaddleOCR":
        ocr = PaddleOCR(show_log=False, use_angle_cls=True, lang='ch' if language == "中文" else 'en')
    elif model == "Tesseract":
        ocr = None
    else:
        return jsonify({"message": "无效的 OCR 模型！"}), 400

    results = []

    try:
        if input_type == "文件夹":
            # 遍历文件夹中的所有图片
            for filename in os.listdir(input_path):
                file_path = os.path.join(input_path, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    results.append(process_image(file_path, model, ocr, language))
        elif input_type == "单个图片":
            if os.path.isfile(input_path):
                results.append(process_image(input_path, model, ocr, language))
            else:
                return jsonify({"message": "输入的单个图片路径无效！"}), 400
        else:
            return jsonify({"message": "无效的输入类型！"}), 400

        # 保存结果到 CSV 文件
        output_file_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        save_to_csv(output_file_path, results)

        return jsonify({"message": "OCR 处理完成！", "results": results, "outputPath": output_file_path}), 200
    except Exception as e:
        return jsonify({"message": f"处理过程中出错: {str(e)}"}), 500

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

def save_to_csv(output_file_path, results):
    """保存 OCR 结果到 CSV 文件"""
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Content"])
        for result in results:
            writer.writerow([result["image_path"], result["content"]])

@app.route('/download', methods=['GET'])
def download_file():
    """提供文件下载接口"""
    file_path = request.args.get('file')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"message": "指定的文件不存在！"}), 400

    directory, filename = os.path.split(file_path)
    try:
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        return jsonify({"message": f"下载出错: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)


