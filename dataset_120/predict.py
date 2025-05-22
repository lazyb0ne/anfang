from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/helmet/weights/best.pt')

# 推理数据集中的验证集图像
results = model.predict(
    source='valid/images',  # 可以是文件夹或单张图片路径
    # source='datasets/ppe/images/val',  # 可以是文件夹或单张图片路径
    save=True,                         # 保存图片结果
    save_txt=True,                     # 保存预测框为txt（YOLO格式）
    # conf=0.25,                         # 置信度阈值（可调）
    conf=0.1,                         # 置信度阈值（可调）
    imgsz=640                          # 推理图片尺寸
)
