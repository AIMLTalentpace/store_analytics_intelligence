from camera_manager import MultiCameraManager

if __name__ == "__main__":
    cameras = [
        "rtsp://10.64.36.13:554/rtsp/streaming?channel=01&subtype=1",
        "rtsp://10.64.36.14:554/rtsp/streaming?channel=01&subtype=1",
    ]

    IMG_SIZE = (800, 640)
    TARGET_FPS = 15

    manager = MultiCameraManager(
        cameras=cameras,
        buffer_size=2,
        fps=TARGET_FPS,
        img_size=IMG_SIZE
    )

    manager.display_streams()