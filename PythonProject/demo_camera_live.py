import json

import cv2
import time
import os
from ultralytics import YOLO

# 视频流地址（可换成你的 stream_url）
# stream_url = "https://pull-f3.douyincdn.com/media/stream-693631149851017900.m3u8?arch_hrchy=w1&auth_key=1746580391-0-0-66f7c2c096718f66e9a259ae968790ef&exp_hrchy=w1&major_anchor_level=common&t_id=037-20250430094310227F19CD76B61FAED3E1-ywzmh6"

# 加载模型
model = YOLO("yolov8n.pt")  # 或 yolov8s.pt

# 创建保存目录
save_dir = "saved_frames"
os.makedirs(save_dir, exist_ok=True)

# 读取配置
with open("config.json", "r") as f:
    config = json.load(f)
stream_url = config.get("stream_url")
print("url:"+ stream_url)
if not stream_url:
    raise ValueError("配置文件中缺少 'stream_url' 参数")

# 打开视频流
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("❌ 无法打开视频流")
    exit()

print("✅ 开始检测... 按 'q' 退出")

# 定时保存帧
last_save_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 视频流读取失败")
        break

    # 检测
    results = model.predict(source=frame, verbose=False)

    # 处理检测结果
    for r in results:
        annotated_frame = r.plot()

        # 显示画面
        cv2.imshow("YOLOv8 检测", annotated_frame)

        # 每 5 秒保存一张图片
        current_time = time.time()
        if current_time - last_save_time >= 5:
            timestamp = int(current_time)
            filename = os.path.join(save_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(filename, annotated_frame)
            print(f"🖼️ 已保存帧: {filename}")
            last_save_time = current_time

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()