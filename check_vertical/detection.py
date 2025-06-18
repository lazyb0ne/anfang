import cv2
import numpy as np
from typing import List, Tuple

import cv2
import numpy as np
from typing import Tuple, List

def detect_all_green_rebars(image_path: str) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float]]]:
    """
    尽量检测图中所有绿色柱子，不限制角度与形状，返回标注图和角度信息。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 放宽绿色范围，覆盖浅绿、深绿等
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 膨胀闭运算以填补断裂
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 80:
            continue

        rect = cv2.minAreaRect(cnt)
        angle = rect[2]
        if angle < -45:
            angle += 90

        box = cv2.boxPoints(rect)
        box = np.intp(box)
        results.append((box, angle))

        # 绘制检测框
        cv2.drawContours(image, [box], 0, (0, 255, 255), 2)
        center = np.mean(box, axis=0).astype(int)
        cv2.putText(image, f"{angle:.1f}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return image, results


def detect_rebar_angles(image_path: str, angle_tolerance: float = 5.0) -> Tuple[np.ndarray, float, List[Tuple[np.ndarray, float]]]:
    """
    检测绿色钢筋柱子的角度，排除横向或扁平的轮廓，返回标注图、主方向角、中位角。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像")

    original_image = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    boxes = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue

        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue

        # 排除扁平横条：宽高比过滤
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio < 2:  # 比如 20x10 会被排除
            continue

        angle = rect[2]
        if angle < -45:
            angle += 90

        # 排除横向角度（例如 -15° ~ +15°）
        if abs(angle) < 20:
            continue

        box = cv2.boxPoints(rect)
        box = np.intp(box)
        boxes.append((box, angle))
        angles.append(angle)

    median_angle = np.median(angles) if angles else 0

    for box, angle in boxes:
        deviation = abs(angle - median_angle)
        color = (0, 255, 0) if deviation <= angle_tolerance else (0, 0, 255)
        cv2.drawContours(original_image, [box], 0, color, 2)
        center = np.mean(box, axis=0).astype(int)
        cv2.putText(original_image, f"{angle:.1f}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return original_image, median_angle, boxes
