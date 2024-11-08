from paddleocr import PaddleOCR
import cv2
from PIL import Image
import pytesseract
import logging

pytesseract.pytesseract.tesseract_cmd = r'D:/Program Files/Tesseract-OCR/tesseract.exe'

language_type = input('Press Enter 0/1 to choose the language type: 0 for English, 1 for Chinese: ')
# 选择 OCR 识别的语言类型
# 0: 英文 1: 中文
if language_type == '0':
    paddle_language_type = 'en'
    tesseract_language_type = 'eng'
elif language_type == '1':
    paddle_language_type = 'ch'
    tesseract_language_type = 'chi_sim'

# 初始化 OCR 模型，指定语言为中文和英文
ocr = PaddleOCR(show_log=False ,use_angle_cls=True, lang=paddle_language_type)

# 读取图像并进行 OCR 检测和识别
image_path = 'image/homework.jpeg'
# image_path = 'image/eng2.png'
result_Paddle = ocr.ocr(image_path, cls=True)

# 使用 Tesseract 进行 OCR 识别
image = Image.open(image_path)
result_Tesseract = pytesseract.image_to_string(image, lang=tesseract_language_type)

# 处理将识别结果写入文件
output_path_Paddle = 'output/paddle_test.txt'
output_path_Tesseract = 'output/tesseract_test.txt'

with open(output_path_Paddle, 'w', encoding='utf-8') as f:
    for line in result_Paddle[0]:
        f.write(str(line[1][0]) + '\n')

with open(output_path_Tesseract, 'w', encoding='utf-8') as f:
    f.write(result_Tesseract)
