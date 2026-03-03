import cv2
import time
from camera_manager import MultiCameraManager

cameras = [
    "rtsp://10.64.36.13:554/rtsp/streaming?channel=01&subtype=1",
    "rtsp://10.64.36.14:554/rtsp/streaming?channel=01&subtype=1",
]

manager = MultiCameraManager(
    cameras=cameras,
    buffer_size=3,
    fps=15,
    img_size=(640, 640),
    num_cores=2
)

while True:
    try:
        frames = manager.get_latest_frames()
        for cam_id, frame in frames.items():
            try:
                if frame is not None:
                    cv2.imshow(f"Camera {cam_id}", frame)
            except Exception as e:
                print(f"[Main] Display error cam {cam_id}: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            manager.stop()
            cv2.destroyAllWindows()
            break

        time.sleep(0.01)

    except Exception as e:
        print(f"[Main] Loop error: {e}")
        time.sleep(0.1)