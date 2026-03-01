"""Inference thread — runs MediaPipe + TFLite at a fixed interval."""

from __future__ import annotations

from typing import Any

import mediapipe as mp
import numpy as np
from PySide6.QtCore import QThread, QTimer, Signal

from src.camera_thread import CameraThread
from src.config import INFERENCE_DEFAULTS, MODEL_PATHS
from src.gesture_handler import GestureActionHandler
from src.inference import InferenceEngine


class InferenceThread(QThread):
    """Periodically grabs the latest camera frame, runs detection + classification,
    and emits results via Qt signals."""

    result_ready = Signal(np.ndarray, str, float, list)

    def __init__(
        self,
        camera: CameraThread,
        engine: InferenceEngine,
        handler: GestureActionHandler,
        interval_ms: int = INFERENCE_DEFAULTS.inference_interval_ms,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._camera = camera
        self._engine = engine
        self._handler = handler
        self._interval_ms = interval_ms
        self._running = False
        self._timer: QTimer | None = None

    def run(self) -> None:
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATHS.hand_landmarker),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=INFERENCE_DEFAULTS.num_hands,
            min_hand_detection_confidence=INFERENCE_DEFAULTS.min_detection_confidence,
        )

        self._running = True
        landmarker = HandLandmarker.create_from_options(options)

        self._timer = QTimer()
        self._timer.setInterval(self._interval_ms)
        self._timer.timeout.connect(lambda: self._tick(landmarker))
        self._timer.start()

        self.exec()

        self._timer.stop()
        landmarker.close()

    def _tick(self, landmarker: Any) -> None:
        frame = self._camera.get_frame()
        if frame is None:
            return

        rgb = frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        landmarks_raw: list[tuple[float, float]] = []

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            gesture, confidence = self._engine.predict(lm)
            h, w = frame.shape[:2]
            landmarks_raw = [(pt.x * w, pt.y * h) for pt in lm]
        else:
            gesture, confidence = "No Hand Detected", 0.0
            self._engine.reset()

        self._handler.on_prediction(gesture, confidence)
        self.result_ready.emit(frame, gesture, confidence, landmarks_raw)

    def stop(self) -> None:
        """Terminate the event loop and wait for the thread."""
        self._running = False
        self.quit()
        self.wait()
