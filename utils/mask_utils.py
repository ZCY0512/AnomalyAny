import numpy as np
import cv2
import random


def generate_ellipse_mask(H, W, valid_mask, min_semi_major=20, max_semi_major=100, min_semi_minor=10, max_semi_minor=80, erosion_kernel_size=10):
    """
    在有效区域(valid_mask)内随机生成一个椭圆，并返回椭圆对应的mask。
    使用腐蚀操作确保椭圆完全在有效区域内部。

    参数:
    - H: 图像的高度
    - W: 图像的宽度
    - valid_mask: 有效区域的mask，1表示有效区域，0表示无效区域
    - min_semi_major: 最小长半轴长度
    - max_semi_major: 最大长半轴长度
    - min_semi_minor: 最小短半轴长度
    - max_semi_minor: 最大短半轴长度
    - erosion_kernel_size: 腐蚀操作的核大小
    
    返回:
    - ellipse_mask: 包含椭圆的二值mask
    """
    
    # 使用腐蚀操作处理valid_mask，确保椭圆完全在有效区域内
    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    eroded_mask = cv2.erode(valid_mask, kernel, iterations=1)
    
    # 获取腐蚀后的valid区域的坐标
    valid_indices = np.argwhere(eroded_mask == 1)
    
    if len(valid_indices) == 0:
        raise ValueError("Valid mask too small after erosion, no space left for ellipse.")
    
    # 随机选择一个有效区域的中心
    center = tuple(random.choice(valid_indices))
    
    # 随机生成长半轴和短半轴的大小
    semi_major = random.randint(min_semi_major, max_semi_major)
    semi_minor = random.randint(min_semi_minor, max_semi_minor)
    
    # 检查椭圆的长半轴和短半轴尺寸是否适应腐蚀后的有效区域
    if semi_major * 2 > H or semi_minor * 2 > W:
        raise ValueError("Ellipse size is too large for the image dimensions.")
    
    # 随机选择椭圆的旋转角度
    angle = random.uniform(0, 180)
    
    # 生成一个空白的mask
    ellipse_mask = np.zeros((H, W), dtype=np.uint8)
    
    # 在椭圆mask中绘制椭圆
    cv2.ellipse(ellipse_mask, center, (semi_major, semi_minor), angle, 0, 360, 255, -1)
    
    return ellipse_mask
