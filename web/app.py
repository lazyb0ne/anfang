import os
import time
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, session
import cv2
from ultralytics import YOLO
from flask import send_from_directory

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = YOLO("weights/best100.pt")

os.makedirs("frames", exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

live_stats = {
    'saved_images': 0,
    'class_counter': {},
}

def generate_camera_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.5, verbose=False)
        result = results[0]
        annotated = result.plot()

        for r in result.boxes.data.tolist():
            cls_id = int(r[5])
            conf = r[4]
            cls_name = model.names[cls_id]
            if conf > 0.5:
                live_stats['class_counter'][cls_name] = live_stats['class_counter'].get(cls_name, 0) + 1
                if cls_name == 'Hardhat':
                    filename = f"frames/live_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated)
                    live_stats['saved_images'] += 1

        ret2, jpeg = cv2.imencode('.jpg', annotated)
        if not ret2:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html',
                           saved_images=live_stats['saved_images'],
                           class_counter=live_stats['class_counter'])

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_image', methods=['GET', 'POST'])
def detect_image():
    if request.method == 'POST':
        if 'image_file' not in request.files:
            return "No file part", 400

        f = request.files['image_file']
        if f.filename == '':
            return "No selected file", 400

        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(path)

        results = model.predict(source=path, conf=0.5)
        result = results[0]
        annotated = result.plot()

        output_path = os.path.join('static', 'results', f.filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)

        details = []
        for r in result.boxes.data.tolist():
            cls_id = int(r[5])
            conf = r[4]
            cls_name = model.names[cls_id]
            details.append({'cls': cls_name, 'conf': round(conf, 2)})

        return render_template('detect_image.html', image=output_path, details=details)

    return render_template('detect_image.html')

@app.route('/detect_video', methods=['GET', 'POST'])
def detect_video():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return "No video file", 400

        f = request.files['video_file']
        if f.filename == '':
            return "No selected file", 400

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(save_path)
        session['uploaded_video_path'] = save_path

        live_stats['saved_images'] = 0
        live_stats['class_counter'] = {}

        return redirect(url_for('video_detection_page'))

    return render_template('detect_video.html')

def generate_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.5, verbose=False)
        result = results[0]
        annotated = result.plot()

        for r in result.boxes.data.tolist():
            cls_id = int(r[5])
            conf = r[4]
            cls_name = model.names[cls_id]
            if conf > 0.5:
                live_stats['class_counter'][cls_name] = live_stats['class_counter'].get(cls_name, 0) + 1
                if cls_name == 'Hardhat':
                    filename = f"frames/video_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated)
                    live_stats['saved_images'] += 1

        ret2, jpeg = cv2.imencode('.jpg', annotated)
        if not ret2:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()

@app.route('/video_feed_upload')
def video_feed_upload():
    path = session.get('uploaded_video_path')
    if not path:
        return "No video uploaded", 404
    return Response(generate_video_frames(path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_detection_page')
def video_detection_page():
    return render_template('video_detection_page.html',
                           saved_images=live_stats['saved_images'],
                           class_counter=live_stats['class_counter'])

@app.route('/get_video_stats')
def get_video_stats():
    return jsonify(live_stats)

@app.route('/records')
def records():
    images = []
    for filename in os.listdir("frames"):
        if filename.endswith(".jpg"):
            path = os.path.join("frames", filename)
            images.append({
                "filename": filename,
                "time": time.ctime(os.path.getmtime(path)),
                "path": path
            })
    images.sort(key=lambda x: x["time"], reverse=True)
    return render_template("records.html", images=images)

@app.route('/frames/<path:filename>')
def frames_static(filename):
    return send_from_directory('frames', filename)

if __name__ == "__main__":
    app.run(debug=True)
