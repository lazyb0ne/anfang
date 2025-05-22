from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.predict(source='res', save=False)

for i, r in enumerate(results):
    r.show()
    r.save(filename=f'results/image_{i}.jpg')