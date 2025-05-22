import cv2
import time
import os
from ultralytics import YOLO

# 加载 YOLOv8 模型（可换 yolov8s.pt / yolov8m.pt）
# model = YOLO("yolov8n.pt")
model = YOLO("../web/weights/best100.pt")

# 打开摄像头（0 = 默认摄像头）
cap = cv2.VideoCapture(0)

# 创建保存目录
os.makedirs("frames", exist_ok=True)

# 记录上次保存时间
last_save_time = time.time()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 推理
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # 取第一张结果（因为是单帧）
    result = results[0]
    annotated_frame = result.plot()

    # 显示画好框的视频流
    cv2.imshow("YOLOv8 Live", annotated_frame)

    # 每3秒保存一帧
    current_time = time.time()
    if current_time - last_save_time >= 3:
        save_path = f"frames/frame_{frame_count}.jpg"
        cv2.imwrite(save_path, annotated_frame)
        print(f"Saved: {save_path}")
        last_save_time = current_time
        frame_count += 1

    # 按下 q 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
