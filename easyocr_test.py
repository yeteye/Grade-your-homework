import easyocr
import cv2
import matplotlib.pyplot as plt

# 创建读取器对象，指定要识别的语言
reader = easyocr.Reader(['ch_sim', 'en'])  # 支持简体中文和英文

# 读取图像
image_path = 'image/homework.jpeg'
results = reader.readtext(image_path)

# 显示结果
image = cv2.imread(image_path)

# 输出到文件
output_path = 'output/easyocr_test.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    for (bbox, text, prob) in results:
        f.write(text + '\n')
