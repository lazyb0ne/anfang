# yolov8n_project.py

from ultralytics import YOLO
import cv2

def main():
    # 加载预训练的 YOLOv8n 模型
    model = YOLO('../yolov8n.pt')  # 这会自动下载模型文件

    # 打开摄像头或读取本地视频/图片
    cap = cv2.VideoCapture(0)  # 参数 0 表示默认摄像头

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头")
            break

        # 使用模型进行推理
        results = model(frame)

        # 解析结果并在帧上绘制边界框
        annotated_frame = results[0].plot()

        # 显示结果
        cv2.imshow('YOLOv8n Detection', annotated_frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()