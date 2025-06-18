from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
STATIC_FOLDER = 'static'
STATIC_RESULT_FOLDER = os.path.join(STATIC_FOLDER, RESULT_FOLDER)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_RESULT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = datetime.now().strftime('%Y%m%d%H%M%S') + '_' + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            result_path, ratio = process_image(filepath)
            return render_template('result.html', image_url=url_for('static', filename=result_path), ratio=f"{ratio:.2%}")
    return render_template('index.html')

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 土地区域（组合：黄土 + 灰白土）
    soil_mask1 = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([35, 255, 255]))
    soil_mask2 = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 60, 230]))
    soil_mask = cv2.bitwise_or(soil_mask1, soil_mask2)

    # 排除挖掘机等设备颜色（亮黄、金属灰等）
    exclude_mask = cv2.inRange(hsv, np.array([15, 40, 120]), np.array([35, 150, 255]))
    soil_mask = cv2.bitwise_and(soil_mask, cv2.bitwise_not(exclude_mask))

    # 绿色区域（绿网）
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    land_mask = soil_mask > 0
    green_on_land_mask = (green_mask > 0) & land_mask

    # 避免除零错误
    if np.sum(land_mask) == 0:
        green_on_land_ratio = 0
    else:
        green_on_land_ratio = np.sum(green_on_land_mask) / np.sum(land_mask)

    # 标红裸露土地
    result_img = image_rgb.copy()
    result_img[land_mask & ~green_on_land_mask] = [255, 0, 0]
    result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

    result_filename = os.path.basename(image_path).replace('.', '_result.')
    relative_result_path = os.path.join(RESULT_FOLDER, result_filename)  # for URL
    absolute_result_path = os.path.join(STATIC_FOLDER, relative_result_path)  # for saving

    os.makedirs(os.path.dirname(absolute_result_path), exist_ok=True)
    cv2.imwrite(absolute_result_path, result_img_bgr)

    return relative_result_path, green_on_land_ratio

if __name__ == '__main__':
    app.run(debug=True)
