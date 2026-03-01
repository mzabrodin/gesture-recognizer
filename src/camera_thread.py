"""Camera capture thread — keeps only the latest frame."""

from __future__ import annotations

import threading

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from src.config import CAMERA_DEFAULTS


class CameraThread(QThread):
    """Reads frames from a webcam as fast as possible, discarding stale ones."""

    frame_ready = Signal()

    def __init__(self, device: int = CAMERA_DEFAULTS.device_index, parent=None) -> None:
        super().__init__(parent)
        self._device = device
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = False

    def get_frame(self) -> np.ndarray | None:
        """Return the most recent frame (BGR, already flipped) or ``None``."""
        with self._lock:
            return self._frame

    def run(self) -> None:
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_DEFAULTS.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_DEFAULTS.height)
        self._running = True

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
            self.frame_ready.emit()

        cap.release()

    def stop(self) -> None:
        """Signal the thread to terminate and wait for it to finish."""
        self._running = False
        self.wait()
