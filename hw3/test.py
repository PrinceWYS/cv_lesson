import cv2  
import numpy as np  
  
# 读取图片 A 和 B  
image_A = cv2.imread('./pic/background.png')  
image_B = cv2.imread('./data/train/bedroom/image_0001.jpg')  
  
# 获取图片 A 和 B 的尺寸  
height_A, width_A, _ = image_A.shape  
height_B, width_B, _ = image_B.shape  
  
# 计算将图片 B 居中放置在图片 A 中的位置  
x_offset = (width_A - width_B) // 2  
y_offset = (height_A - height_B) // 2  
  
# 将图片 B 放置在图片 A 中心位置  
image_A[y_offset:y_offset+height_B, x_offset:x_offset+width_B] = image_B  
  
# 保存最终的图片  
cv2.imwrite('centered_image_A_with_B.jpg', image_A)  