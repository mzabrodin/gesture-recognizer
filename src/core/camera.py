from __future__ import annotations

import cv2

from src.config import VIDEO


def open_camera() -> cv2.VideoCapture:
    # як було раніше: пробуємо основну камеру, за потреби можна змінити індекс
    cap = cv2.VideoCapture(VIDEO.camera_index)
    if not cap.isOpened() and VIDEO.camera_index != 0:
        # fallback на 0, якщо кастомний індекс не спрацював
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO.height)
    return cap
