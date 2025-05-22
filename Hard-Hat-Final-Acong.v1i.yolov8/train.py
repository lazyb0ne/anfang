from ultralytics import YOLO

# 加载模型（可以换 yolov8s.pt 等）
model = YOLO('yolov8n.pt')

# 训练
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=2,
    name='helmet'
)
