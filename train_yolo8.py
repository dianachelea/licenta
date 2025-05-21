from ultralytics import YOLO

model = YOLO("runs/detect/train6/weights/best.pt")

model.train(
    data=r"dataset_final/data.yaml", 
    epochs=50,
    imgsz=640,
    batch=32,
    device='cuda'  
)
