from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask import Response
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
app.config['RESULT_IMAGE'] = 'static/results/images'
app.config['RESULT_VIDEO'] = 'static/results/videos'
os.makedirs('frames', exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_IMAGE'], exist_ok=True)
os.makedirs(app.config['RESULT_VIDEO'], exist_ok=True)

model = YOLO("weights/best100.pt")
cap = cv2.VideoCapture(0)

frame_count = 0
hardhat_frame_count = 0
class_counter = defaultdict(int)
detection_log_path = "detection_log.json"

if not os.path.exists(detection_log_path) or os.path.getsize(detection_log_path) == 0:
    with open(detection_log_path, 'w') as f:
        json.dump([], f)
else:
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

@app.route('/')
def index():
    return render_template('index.html',
                           saved_images=hardhat_frame_count,
                           class_counter=dict(class_counter),
                           yolo_params=model.args)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

@app.route('/detect/image', methods=['GET', 'POST'])
def detect_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            results = model.predict(source=path, conf=0.5, verbose=False)
            result = results[0]
            annotated = result.plot()
            save_path = os.path.join(app.config['RESULT_IMAGE'], filename)
            cv2.imwrite(save_path, annotated)
            log = {
                "type": "image",
                "filename": filename,
                "saved_path": save_path,
                "time": datetime.now().isoformat(),
                "objects": []
            }
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                log["objects"].append({"class": cls_name, "conf": round(conf, 3)})
            save_log(log)
            return render_template('detect_image.html', image_url=url_for('static', filename=f'results/images/{filename}'), result=log)
    return render_template('detect_image.html')

@app.route('/detect/video', methods=['GET', 'POST'])
def detect_video():
    if request.method == 'POST':
        file = request.files['video']
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            result_path = os.path.join(app.config['RESULT_VIDEO'], filename)
            model.predict(source=path, save=True, project=app.config['RESULT_VIDEO'], name='', conf=0.5)
            log = {
                "type": "video",
                "filename": filename,
                "saved_path": result_path,
                "time": datetime.now().isoformat()
            }
            save_log(log)
            return render_template('detect_video.html', video_path=url_for('static', filename=f'results/videos/{filename}'), result=log)
    return render_template('detect_video.html')

@app.route('/records')
def records():
    with open(detection_log_path) as f:
        logs = json.load(f)
    image_logs = [log for log in logs if log.get("type") == "image" and "saved_path" in log]
    return render_template('records.html', logs=image_logs)


# ... 所有路由定义
if __name__ == "__main__":
    app.run(debug=True)
