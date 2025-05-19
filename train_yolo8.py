from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=r"Z:\dataset_final\data.yaml", 
    epochs=2,
    imgsz=640,
    batch=32,
    device='cpu'  
)
