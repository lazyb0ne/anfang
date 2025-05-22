import os
import cv2
import time
import threading
import shutil
from flask import Flask, Response, render_template, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)

# 初始化YOLOv8n模型权重路径
model = YOLO('weights/best.pt')

SAVE_DIR = 'saved_frames'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

lock = threading.Lock()
latest_frame = None  # 摄像头最新原始帧
latest_detected_frame = None  # 检测后的帧（带检测框）

FRAME_RATE = 10  # 摄像头读取帧率和视频输出帧率
DETECT_INTERVAL = 1.0  # 检测间隔秒数
imgsz = 320  # 模型输入尺寸

detection_enabled = False  # 默认不检测
detection_event = threading.Event()  # 控制检测线程是否运行，默认不运行，不设置event

stats = {
    "fps": 0.0,
    "inference_time": 0.0,
    "last_detection_count": 0,
    "last_save_path": "",
}

last_save_time = 0
SAVE_INTERVAL = 3

def camera_reader():
    """摄像头线程，持续读取摄像头，保持设定帧率"""
    global latest_frame
    frame_time = 1.0 / FRAME_RATE
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(frame_time)
            continue
        with lock:
            latest_frame = frame
        elapsed = time.time() - start
        time.sleep(max(0, frame_time - elapsed))

def detector():
    """检测线程，只在检测开启时执行推理，否则阻塞等待"""
    global latest_detected_frame, last_save_time
    last_detect_time = 0

    while True:
        detection_event.wait()  # 检测关闭时阻塞等待，不占CPU

        now = time.time()
        if now - last_detect_time < DETECT_INTERVAL:
            time.sleep(0.05)
            continue

        with lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            img = latest_frame.copy()

        start_inf = time.time()
        try:
            results = model.predict(img, imgsz=imgsz, conf=0.25, verbose=False)
        except Exception as e:
            print("[ERROR] 模型推理失败:", e)
            time.sleep(0.5)
            continue
        inf_time = time.time() - start_inf

        detected_img = results[0].plot()

        with lock:
            latest_detected_frame = detected_img
            stats['inference_time'] = inf_time * 1000
            stats['last_detection_count'] = len(results[0].boxes) if results[0].boxes is not None else 0

        save_flag = False
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                if cls_name == "NO-Hardhat":
                    save_flag = True
                    break

        if save_flag and (now - last_save_time) > SAVE_INTERVAL:
            timestamp = int(now)
            save_path = os.path.join(SAVE_DIR, f"frame_{timestamp}.jpg")
            cv2.imwrite(save_path, detected_img)
            with lock:
                stats['last_save_path'] = save_path
            last_save_time = now

        last_detect_time = now

def generate():
    """视频流生成器，推送带检测结果的帧（检测开启时）或原始帧（检测关闭时）"""
    frame_time = 1.0 / FRAME_RATE
    prev_time = time.time()
    frame_count = 0

    while True:
        start = time.time()
        with lock:
            if detection_enabled and latest_detected_frame is not None:
                frame_to_send = latest_detected_frame
            elif latest_frame is not None:
                frame_to_send = latest_frame
            else:
                time.sleep(0.01)
                continue

            ret, jpeg = cv2.imencode('.jpg', frame_to_send)
            if not ret:
                time.sleep(0.01)
                continue
            frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_count += 1
        now = time.time()
        elapsed = now - prev_time
        if elapsed >= 1.0:
            with lock:
                stats['fps'] = frame_count / elapsed
            frame_count = 0
            prev_time = now

        elapsed_gen = time.time() - start
        time.sleep(max(0, frame_time - elapsed_gen))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    with lock:
        return jsonify(stats)

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_enabled
    data = request.json
    if 'enable' in data:
        detection_enabled = bool(data['enable'])
        if detection_enabled:
            detection_event.set()  # 开启检测线程运行
        else:
            detection_event.clear()  # 阻塞检测线程
            with lock:
                if latest_frame is not None:
                    global latest_detected_frame
                    latest_detected_frame = latest_frame.copy()  # 切换为原始帧，保证流畅
        return jsonify({"detection_enabled": detection_enabled})
    else:
        return jsonify({"error": "缺少参数 enable"}), 400

if __name__ == '__main__':
    threading.Thread(target=camera_reader, daemon=True).start()
    threading.Thread(target=detector, daemon=True).start()
    app.run(host='0.0.0.0', port=5001)
