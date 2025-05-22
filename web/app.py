from flask import Flask, Response, render_template, request, redirect, url_for
from ultralytics import YOLO
import cv2
import time
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from collections import defaultdict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs('frames', exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = YOLO("weights/best100.pt")
cap = cv2.VideoCapture(0)

frame_count = 0
hardhat_frame_count = 0
class_counter = defaultdict(int)
detection_log_path = "detection_log.json"

# 初始化日志文件
# 初始化日志文件：为空或格式错误也自动修复
if not os.path.exists(detection_log_path) or os.path.getsize(detection_log_path) == 0:
    with open(detection_log_path, 'w') as f:
        json.dump([], f)
else:
    # 尝试加载一次，确保文件合法
    try:
        with open(detection_log_path, 'r') as f:
            json.load(f)
    except json.JSONDecodeError:
        with open(detection_log_path, 'w') as f:
            json.dump([], f)

def save_log(data):
    with open(detection_log_path, 'r+') as f:
        logs = json.load(f)
        logs.append(data)
        f.seek(0)
        json.dump(logs, f, indent=2)

def gen_frames():
    global frame_count, hardhat_frame_count, class_counter
    last_save_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, conf=0.5, verbose=False)
        result = results[0]
        annotated_frame = result.plot()

        current_time = time.time()
        hardhat_detected = False
        log_entry = {
            "time": datetime.now().isoformat(),
            "objects": [],
        }

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            log_entry["objects"].append({"class": cls_name, "conf": round(conf, 3)})

            class_counter[cls_name] += 1
            if cls_name.lower() == "hardhat" and conf > 0.5:
                hardhat_detected = True

        if hardhat_detected and current_time - last_save_time >= 3:
            save_path = f"frames/frame_{frame_count}.jpg"
            cv2.imwrite(save_path, annotated_frame)
            print(f"Saved: {save_path}")
            frame_count += 1
            hardhat_frame_count += 1
            last_save_time = current_time
            log_entry["saved"] = save_path

        save_log(log_entry)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html',
                           saved_images=hardhat_frame_count,
                           class_counter=dict(class_counter),
                           yolo_params=model.args)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files[]')
    for file in files:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        run_detection_on_file(save_path)
    return redirect(url_for('index'))

def run_detection_on_file(path):
    img = cv2.imread(path)
    if img is None:
        return
    results = model.predict(source=img, conf=0.5, verbose=False)
    result = results[0]
    log_entry = {
        "time": datetime.now().isoformat(),
        "source": path,
        "objects": [],
    }
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        class_counter[cls_name] += 1
        log_entry["objects"].append({"class": cls_name, "conf": round(conf, 3)})
    save_log(log_entry)

@app.route('/records')
def records():
    with open(detection_log_path) as f:
        logs = json.load(f)
    return {"records": logs}

if __name__ == '__main__':
    app.run(debug=True)
