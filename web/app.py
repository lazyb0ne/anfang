import os
import time
from collections import defaultdict
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# === 统一配置区：方便统一管理和调节参数 ===

# YOLOv8 模型权重路径
MODEL_WEIGHTS_PATH = "../web/weights/best100.pt"

# 检测置信度阈值，低于该值的框将被忽略
DETECTION_CONFIDENCE = 0.3

# 摄像头索引，0通常是默认摄像头
CAMERA_INDEX = 0

# 截图保存目录（检测到未戴安全帽时截图存放目录）
FRAMES_SAVE_DIR = "frames"

# 上传文件保存目录（上传图片/视频存放目录）
UPLOADS_DIR = "uploads"

# 截图保存的最小时间间隔（秒），防止频繁保存重复图片
SAVE_INTERVAL_SECONDS = 3

# 每行检测记录页面显示的图片数量
RECORDS_IMAGES_PER_ROW = 5

# === 初始化模型和资源 ===

model = YOLO(MODEL_WEIGHTS_PATH)

# 创建必要目录
os.makedirs(FRAMES_SAVE_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

# 全局统计变量
saved_images = 0
unsafe_total = 0
class_counter = defaultdict(int)
last_save_time = time.time()


@app.route('/')
def index():
    return render_template('index.html',
                           saved_images=saved_images,
                           unsafe_total=unsafe_total,
                           class_counter=dict(class_counter),
                           yolo_params={
                               'model_path': MODEL_WEIGHTS_PATH,
                               'conf': DETECTION_CONFIDENCE,
                               'num_classes': len(model.names),
                               'camera_index': CAMERA_INDEX,
                               'save_interval': SAVE_INTERVAL_SECONDS
                           })


def gen_frames():
    global saved_images, unsafe_total, last_save_time, class_counter

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=DETECTION_CONFIDENCE, verbose=False)
        result = results[0]
        annotated_frame = result.plot()
        boxes = result.boxes

        unsafe_count = 0
        frame_class_count = defaultdict(int)

        for box in boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            conf = float(box.conf)

            if conf >= DETECTION_CONFIDENCE:
                frame_class_count[cls_name] += 1
                if cls_name == 'Hardhat':
                    class_counter[cls_name] += 1
                else:
                    unsafe_count += 1

        unsafe_total += unsafe_count

        current_time = time.time()
        if unsafe_count > 0 and current_time - last_save_time >= SAVE_INTERVAL_SECONDS:
            save_path = os.path.join(FRAMES_SAVE_DIR, f"unsafe_{int(current_time)}.jpg")
            cv2.imwrite(save_path, annotated_frame)
            saved_images += 1
            last_save_time = current_time

        ret2, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_image', methods=['GET', 'POST'])
def detect_image():
    if request.method == 'POST':
        if 'image_file' not in request.files:
            return 'No image_file part', 400
        file = request.files['image_file']
        if file.filename == '':
            return 'No selected file', 400

        # 保存上传的原始图片
        filename = f"upload_{int(time.time())}_{file.filename}"
        upload_path = os.path.join(UPLOADS_DIR, filename)
        file.save(upload_path)

        # YOLO 检测
        results = model.predict(source=upload_path, conf=DETECTION_CONFIDENCE, verbose=False)
        result = results[0]
        annotated_frame = result.plot()

        # 保存带框图片
        det_filename = f"det_{filename}"
        save_path = os.path.join(UPLOADS_DIR, det_filename)
        cv2.imwrite(save_path, annotated_frame)

        # 统计每个类别数量
        class_counts = {}
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if conf >= DETECTION_CONFIDENCE:
                cls_name = model.names[cls_id]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        return render_template(
            'detect_image.html',
            detected_image_url=url_for('uploaded_file', filename=det_filename),
            orig_image_url=url_for('uploaded_file', filename=filename),
            class_counts=class_counts,
            conf=DETECTION_CONFIDENCE,
            classes=model.names
        )
    else:
        return render_template('detect_image.html')


@app.route('/detect_video', methods=['GET', 'POST'])
def detect_video():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return 'No video_file part', 400
        file = request.files['video_file']
        if file.filename == '':
            return 'No selected file', 400

        filename = f"upload_{int(time.time())}_{file.filename}"
        upload_path = os.path.join(UPLOADS_DIR, filename)
        file.save(upload_path)

        def gen_video():
            cap_v = cv2.VideoCapture(upload_path)
            while True:
                ret, frame = cap_v.read()
                if not ret:
                    break

                results = model.predict(source=frame, conf=DETECTION_CONFIDENCE, verbose=False)
                result = results[0]
                annotated_frame = result.plot()

                ret2, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            cap_v.release()

        return Response(gen_video(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return render_template('detect_video.html')


@app.route('/records')
def records():
    image_list = []
    folder = FRAMES_SAVE_DIR
    for filename in sorted(os.listdir(folder), reverse=True):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            filepath = os.path.join(folder, filename)
            image_list.append({
                'filename': filename,
                'path': filepath,
                'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getctime(filepath)))
            })
    return render_template('records.html', images=image_list, images_per_row=RECORDS_IMAGES_PER_ROW)


@app.route('/delete_all_records', methods=['POST'])
def delete_all_records():
    folder = FRAMES_SAVE_DIR
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)
    return redirect(url_for('records'))


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOADS_DIR, filename)


@app.route('/frames/<path:filename>')
def frames_static(filename):
    return send_from_directory(FRAMES_SAVE_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True)
