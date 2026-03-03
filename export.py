from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="openvino", imgsz=640, dynamic=True, half=True, device="cpu")
