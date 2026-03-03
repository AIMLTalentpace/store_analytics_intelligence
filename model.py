# import cv2
# from ultralytics import YOLO

# class Detector:
#     def __init__(self, model_path='yolov8n.pt'):
#         self.model = YOLO(model_path)

#     def process_frame(self, frame):
#         results = self.model(frame, verbose=False, conf=0.25)
#         annotated_frame = results[0].plot()
#         return annotated_frame




# import cv2
# from ultralytics import YOLO

# class Detector:
#     def __init__(self, model_path='yolov8n_openvino_model/'):
#         self.model = YOLO(model_path, task='detect')

#     def process_frame(self, frame):
#         results = self.model(frame, verbose=False, conf=0.35, imgsz=640)
        
#         annotated_frame = results[0].plot()
#         return annotated_frame



# import cv2
# from ultralytics import YOLO

# class Detector:
#     def __init__(self, model_path='yolov8n_openvino_model/'):
#         self.model = YOLO(model_path, task='detect')

#     def process_frame(self, frame):
#         results = self.model.track(
#             frame,
#             persist=True,
#             classes=0,
#             verbose=False,
#             conf=0.45,
#             iou=0.5,
#             imgsz=640,
#             tracker="botsort.yaml"
#         )
        
#         if results[0].boxes.id is not None:
#             return results[0].plot()
        
#         return results[0].plot()


import cv2
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolov8n_openvino_model/'):
        self.model = YOLO(model_path, task='detect')

    def process_frame(self, frame):
        # Using ByteTrack for more stable ID association and stream=True for optimized processing
        results = self.model.track(
            frame,
            persist=True,
            classes=0,
            verbose=False,
            conf=0.45,
            iou=0.5,
            imgsz=640,
            tracker="bytetrack.yaml",
            stream=False
        )
        
        # results is a list when stream=False
        return results[0].plot()