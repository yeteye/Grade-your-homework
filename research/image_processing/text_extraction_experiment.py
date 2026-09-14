import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw
import os
import shutil

# 明确指定Tesseract路径
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD') or shutil.which('tesseract') or 'tesseract'

def enhance_image(image):
    """图像预处理增强函数"""
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值处理以处理不同光照条件
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 形态学操作清理噪点
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return gray, morph

def detect_text_regions(image, enhanced):
    """检测可能包含文本的区域"""
    # 使用MSER检测器寻找文本区域
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(enhanced)
    
    # 创建文本区域掩码
    text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 填充检测到的区域
    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.drawContours(text_mask, [hull], -1, (255), -1)
    
    # 扩展文本区域以覆盖完整字符
    kernel = np.ones((5, 5), np.uint8)
    text_mask = cv2.dilate(text_mask, kernel, iterations=2)
    
    return text_mask

def extract_text_pixels_enhanced(image_path, output_path=None, debug=False):
    """
    增强版文本像素提取
    
    参数:
        image_path: 输入图片路径
        output_path: 输出图片路径（可选）
        debug: 是否保存中间处理结果（调试用）
    
    返回:
        处理后的图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 图像预处理
    gray, enhanced = enhance_image(image)
    
    # 保存增强后的图像用于OCR
    
    # 使用OCR检测文本区域
    custom_config = r'--oem 3 --psm 11 -l chi_sim+eng'
    data = pytesseract.image_to_data(
        Image.fromarray(enhanced), 
        output_type=pytesseract.Output.DICT, 
        config=custom_config
    )
    
    # 创建结果图像（白色背景）
    height, width = gray.shape
    result = np.ones((height, width), dtype=np.uint8) * 255
    
    # 使用MSER检测潜在的文本区域
    text_mask = detect_text_regions(gray, enhanced)
    
    # 合并OCR检测到的文本区域
    has_valid_regions = False
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30 and data['text'][i].strip():  # 提高置信度阈值
            has_valid_regions = True
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # 确保区域在图像范围内
            if x >= 0 and y >= 0 and x+w <= width and y+h <= height:
                # 扩大检测区域以捕获整个字符
                x_expanded = max(0, x - 5)
                y_expanded = max(0, y - 5)
                w_expanded = min(width - x_expanded, w + 10)
                h_expanded = min(height - y_expanded, h + 10)
                
                # 从增强图像中提取文本区域
                region = enhanced[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded]
                
                # 复制到结果中 (文本为黑色)
                result[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded] = \
                    cv2.bitwise_not(region)
    
    # 如果OCR未检测到任何文本区域，使用MSER结果
    if not has_valid_regions:
        # 应用MSER检测的文本掩码
        text_pixels = cv2.bitwise_and(enhanced, enhanced, mask=text_mask)
        
        # 将掩码应用到结果图像
        result[text_mask > 0] = 255 - enhanced[text_mask > 0]
    
    # 最终清理和增强
    # 反转图像使文本为黑色
    # Already black text on a white background.
    
    # 减少噪点
    kernel = np.ones((2, 2), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    # 确保背景为白色
    _, result = cv2.threshold(result, 200, 255, cv2.THRESH_BINARY)
    
    # 删除临时文件
    
    # 保存调试图像
    if debug:
        cv2.imwrite("debug_gray.png", gray)
        cv2.imwrite("debug_enhanced.png", enhanced)
        cv2.imwrite("debug_text_mask.png", text_mask)
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result

def extract_text_by_connected_components(image_path, output_path=None):
    """
    使用连通区域分析提取文本像素
    
    参数:
        image_path: 输入图片路径
        output_path: 输出图片路径（可选）
    
    返回:
        处理后的图像 
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用自适应阈值将图像二值化
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 形态学操作去除噪点
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 连通区域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # 创建结果图像（白色背景）
    result = np.ones_like(gray) * 255
    
    # 筛选可能是文本的连通区域
    for i in range(1, num_labels):  # 从1开始跳过背景
        x, y, w, h, area = stats[i]
        
        # 通过几何特征筛选文本区域
        # 排除太小或太大的区域
        if area < 20 or area > gray.size / 10:
            continue
            
        # 排除宽高比异常的区域
        if w / h > 20 or h / w > 20:
            continue
            
        # 文本高度通常在一定范围内
        if h < 5 or h > gray.shape[0] / 5:
            continue
        
        # 提取该连通区域
        component_mask = np.zeros_like(gray)
        component_mask[labels == i] = 255
        
        # 将连通区域添加到结果中（黑色文本）
        result[labels == i] = 0
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result

def extract_text_hybrid(image_path, output_path=None):
    """
    混合方法提取文本像素，结合多种技术
    
    参数:
        image_path: 输入图片路径
        output_path: 输出图片路径（可选）
    
    返回:
        处理后的图像
    """
    # 使用两种方法处理图像
    result1 = extract_text_pixels_enhanced(image_path)
    result2 = extract_text_by_connected_components(image_path)
    
    # 合并结果（取两种方法的交集）
    combined = cv2.bitwise_and(result1, result2)
    
    # 最终清理和增强
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, cleaned)
    
    return cleaned

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Experimental text pixel extraction. OCR modes require Tesseract.')
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--method', choices=('connected', 'ocr', 'hybrid'), default='connected')
    args = parser.parse_args()
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        parser.error('Choose a new output path to preserve existing images.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    methods = {'connected':extract_text_by_connected_components, 'ocr':extract_text_pixels_enhanced, 'hybrid':extract_text_hybrid}
    methods[args.method](str(args.input), str(args.output))
    print(args.output)

if __name__ == "__main__":
    main()