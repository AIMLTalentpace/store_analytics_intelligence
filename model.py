import cv2
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def process_frame(self, frame):
        results = self.model(frame, verbose=False, conf=0.25)
        annotated_frame = results[0].plot()
        return annotated_frame