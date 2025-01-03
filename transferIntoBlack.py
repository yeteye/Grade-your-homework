import cv2

# 读取图像
image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# 使用Otsu's二值化方法自动找到最佳阈值
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 反转图像，使文字为黑色，背景为白色
binary_image = cv2.bitwise_not(binary_image)

# 保存处理后的图像
cv2.imwrite('homework_bw2.jpg', binary_image)

# 显示处理后的图像（可选）
cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()