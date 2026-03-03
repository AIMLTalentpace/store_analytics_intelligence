import cv2
import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path='yolov8n_openvino_model/'):
        self.model = YOLO(model_path, task='detect')

    def process_frame(self, frame):
        h, w = frame.shape[:2]

        results = self.model.track(
            frame,
            persist=True,
            classes=0,
            verbose=False,
            conf=0.25,
            iou=0.5,
            imgsz=max(h, w),  
            tracker="botsort.yaml",
            stream=False
        )

        result = results[0]
        output = frame.copy()

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box.astype(int)

                # Draw thin bounding box
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)

                label = f"{track_id}"
                font_scale = 0.4
                thickness = 1

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )

                # Label background
                cv2.rectangle(
                    output,
                    (x1, y1 - th - 4),
                    (x1 + tw + 4, y1),
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    output,
                    label,
                    (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA
                )

        return output