from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import cv2
import time
import os

app = Flask(__name__)

# 加载模型
model = YOLO("weights/best100.pt")

# 创建保存目录
os.makedirs("frames", exist_ok=True)

# 摄像头
cap = cv2.VideoCapture(0)

# 模板页面（显示视频）
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLOv8 实时视频流</title>
</head>
<body>
    <h1>YOLOv8 视频流检测</h1>
    <img src="{{ url_for('video_feed') }}" width="800" />
</body>
</html>
"""

# 视频流生成器
def gen_frames():
    frame_count = 0
    last_save_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 推理
        results = model.predict(source=frame, conf=0.5, verbose=False)
        result = results[0]
        annotated_frame = result.plot()

        # 每3秒保存一帧
        current_time = time.time()
        if current_time - last_save_time >= 3:
            save_path = f"frames/frame_{frame_count}.jpg"
            cv2.imwrite(save_path, annotated_frame)
            print(f"Saved: {save_path}")
            last_save_time = current_time
            frame_count += 1

        # 编码成 JPEG 格式
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # 返回 multipart (MJPEG) 数据
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 关闭摄像头和窗口（用于调试关闭）
@app.route('/shutdown')
def shutdown():
    cap.release()
    cv2.destroyAllWindows()
    return "Shut down."


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
