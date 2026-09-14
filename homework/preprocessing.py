"""Reusable image cleanup, including Otsu binarization from the original experiment."""
from PIL import ImageFilter, ImageOps
from .validation import ValidationError

MODES = ('original', 'grayscale', 'binary', 'enhanced')


def otsu_threshold(histogram):
    total = sum(histogram)
    if not total:
        return 0
    full_sum = sum(i * n for i, n in enumerate(histogram))
    left_count = left_sum = 0
    best_variance, threshold = -1, 0
    for level, count in enumerate(histogram):
        left_count += count
        left_sum += level * count
        right_count = total - left_count
        if not left_count or not right_count:
            continue
        difference = left_sum / left_count - (full_sum - left_sum) / right_count
        variance = left_count * right_count * difference * difference
        if variance > best_variance:
            best_variance, threshold = variance, level
    return threshold


def prepare_image(image, mode='original'):
    if mode not in MODES:
        raise ValidationError('不支持的图像预处理方式。')
    if mode == 'original':
        return image.convert('RGB')
    gray = ImageOps.grayscale(image)
    if mode == 'grayscale':
        return gray.convert('RGB')
    if mode == 'enhanced':
        return ImageOps.autocontrast(gray).filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3)).convert('RGB')
    threshold = otsu_threshold(gray.histogram())
    return gray.point(lambda p: 255 if p > threshold else 0).convert('RGB')
