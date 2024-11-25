from paddleocr import PaddleOCR
from PIL import Image
import pytesseract
import os
import pandas as pd

# 配置 Tesseract 的安装路径
pytesseract.pytesseract.tesseract_cmd = r'D:/Program Files/Tesseract-OCR/tesseract.exe'

# 提供选项供用户选择 OCR 模型和语言
print("请选择 OCR 模型：")
print("1: PaddleOCR")
print("2: Tesseract")
model_choice = input("请输入模型编号 (1 或 2): ")

if model_choice not in ('1', '2'):
    print("无效的选择，程序退出。")
    exit()

print("\n请选择语言类型：")
print("1: 中文")
print("2: 英文")
language_choice = input("请输入语言编号 (1 或 2): ")

if language_choice == '1':
    paddle_language_type = 'ch'
    tesseract_language_type = 'chi_sim'
elif language_choice == '2':
    paddle_language_type = 'en'
    tesseract_language_type = 'eng'
else:
    print("无效的选择，程序退出。")
    exit()

# 提示用户选择文件夹或单个图片
print("\n请选择输入类型：")
print("1: 文件夹")
print("2: 单个图片")
input_choice = input("请输入选项编号 (1 或 2): ")

if input_choice == '1':
    folder_path = input("\n请输入包含图片文件的文件夹路径: ")
    if not os.path.isdir(folder_path):
        print("指定的文件夹路径不存在，程序退出。")
        exit()
    input_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                   if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
elif input_choice == '2':
    single_file_path = input("\n请输入单个图片文件的路径: ")
    if not os.path.isfile(single_file_path):
        print("指定的图片文件路径不存在，程序退出。")
        exit()
    input_files = [single_file_path]
else:
    print("无效的选择，程序退出。")
    exit()

# 输出结果保存的 CSV 文件路径
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(output_dir, 'ocr_results.csv')

# 初始化 OCR 模型（如果选择了 PaddleOCR）
ocr = None
if model_choice == '1':
    print("\n正在初始化 PaddleOCR...")
    ocr = PaddleOCR(show_log=False, use_angle_cls=True, lang=paddle_language_type)

# 执行 OCR
results = []

print("\n正在识别图片...")
for file_path in input_files:
    try:
        if model_choice == '1':  # 使用 PaddleOCR
            result_Paddle = ocr.ocr(file_path, cls=True)
            content = '\n'.join([str(line[1][0]) for line in result_Paddle[0]])
        elif model_choice == '2':  # 使用 Tesseract
            image = Image.open(file_path)
            content = pytesseract.image_to_string(image, lang=tesseract_language_type)
        else:
            content = "未识别的模型"

        results.append({'Image Path': file_path, 'Content': content})
        print(f"已完成: {file_path}")

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

# 将结果保存到 CSV 文件
print("\n正在保存结果到 CSV 文件...")
df = pd.DataFrame(results)
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 解决中文乱码问题
print(f"识别结果已保存到: {output_csv_path}")
