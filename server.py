import csv
import os
import logging
import sys
import subprocess
import pytesseract
from PIL import Image
from flask_cors import CORS
from paddleocr import PaddleOCR
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory, render_template
from reference import calculate_similarity
from grade import get_points  # 导入新增的deepseek评分函数

# 设置日志
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

python_path = './env/MyHanlp/Scripts/python.exe' 
pytesseract_path = r'D:/Program Files/Tesseract-OCR/tesseract.exe'
txt_compare_path = './txt_compare/text_comparer.py'

# 配置 Tesseract 路径（根据实际安装路径修改）
pytesseract.pytesseract.tesseract_cmd = pytesseract_path

# 默认输出路径
OUTPUT_DIR = 'output'
OUTPUT_FILE = 'ocr_results.csv'

# 支持的文件类型
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配置文件上传目录
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 检查文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')  # 渲染 templates 目录下的 index.html 页面

@app.route('/ocr', methods=['POST'])
def ocr_service():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"message": "没有找到文件！"}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1 and file2:
        # 保存文件到服务器
        file1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file2_path = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)

        file1.save(file1_path)
        file2.save(file2_path)

        model = request.form.get('model')
        language = request.form.get('language')
        print(file1_path, file2_path, model, language)

        ocr = None
        if model == "PaddleOCR":
            ocr = PaddleOCR(show_log=False, use_angle_cls=True, lang='ch' if language == "中文" else 'en')
        elif model == "Tesseract":
            pass
        else:
            return jsonify({"message": "无效的 OCR 模型！"}), 400

        work_content = ""
        answer_content = ""

        try:
            # PaddleOCR 识别过程
            if model == "PaddleOCR":
                result1 = ocr.ocr(file1_path, cls=True)
                work_content = '\n'.join([str(line[1][0]) for line in result1[0]])
                result2 = ocr.ocr(file2_path, cls=True)
                answer_content = '\n'.join([str(line[1][0]) for line in result2[0]])

            # Tesseract 识别过程
            elif model == "Tesseract":
                work_image = Image.open(file1_path)
                work_content = pytesseract.image_to_string(work_image, lang='chi_sim' if language == "中文" else 'eng')
                answer_image = Image.open(file2_path)
                answer_content = pytesseract.image_to_string(answer_image, lang='chi_sim' if language == "中文" else 'eng')

            return jsonify({
                "workContent": work_content,
                "answerContent": answer_content,
            }), 200

        except Exception as e:
            logging.error(f"处理过程中出错: {str(e)}")
            return jsonify({"message": f"处理过程中出错: {str(e)}"}), 500

    return jsonify({"message": "上传文件失败"}), 400

@app.route('/compare_texts', methods=['POST'])
def compare_texts():
    data = request.json
    work_content = data.get('workContent')
    answer_content = data.get('answerContent')
    use_deepseek = data.get('useDeepseek', False)  # 获取是否使用deepseek增强评分

    if not work_content or not answer_content:
        return jsonify({"message": "请输入作业内容和参考答案内容"}), 400

    # 使用原方法计算相似度
    base_similarity = calculate_similarity(work_content, answer_content)
    
    # 如果启用deepseek增强评分
    if use_deepseek:
        try:
            # 使用deepseek进行评分
            deepseek_score = get_points(answer_content, work_content)
            
            # 加权平均（这里可以调整权重）
            final_similarity = (base_similarity * 0.3) + (deepseek_score * 0.7)
            logging.info(f"基础相似度: {base_similarity}, Deepseek评分: {deepseek_score}, 最终评分: {final_similarity}")
            
            return jsonify({"similarity": final_similarity * 100}), 200
        except Exception as e:
            logging.error(f"Deepseek评分出错: {str(e)}")
            # 出错时返回基础评分
            return jsonify({"similarity": base_similarity * 100}), 200
    else:
        # 只使用原评分方法
        return jsonify({"similarity": base_similarity * 100}), 200
    
if __name__ == '__main__':
    app.run(debug=True)