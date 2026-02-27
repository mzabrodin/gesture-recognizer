from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QObject, Signal

from actions.handler import GestureActionHandler
from config import THRESHOLDS

from .camera import open_camera
from .classifier import GestureClassifier, InferenceAssets, load_inference_assets
from .detector import HandDetector, HandLandmarkerResult

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
]


@dataclass
class FrameResult:
    frame: np.ndarray
    prediction: str
    confidence: float
    fps: float
    landmarks: Optional[HandLandmarkerResult]


class PipelineWorker(QObject):
    frame_ready = Signal(np.ndarray, str, float, float, object)  # frame, prediction, conf, fps, result
    action_triggered = Signal(str)  # gesture name when an action runs (e.g. for toast)
    error = Signal(str)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._stop_flag = threading.Event()
        self._state_lock = threading.Lock()
        self._current_result: Optional[HandLandmarkerResult] = None
        self._actions_enabled: bool = True

        self._assets: InferenceAssets | None = None
        self._classifier: GestureClassifier | None = None
        self._action_handler: GestureActionHandler | None = None
        self._detector: HandDetector | None = None

    def _detector_callback(
        self,
        result: Optional[HandLandmarkerResult],
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        with self._state_lock:
            self._current_result = result

    def set_actions_enabled(self, enabled: bool) -> None:
        self._actions_enabled = bool(enabled)

    def set_confidence_threshold(self, value: float) -> None:
        if self._action_handler is not None:
            self._action_handler.confidence_threshold = max(0.0, min(1.0, value))

    def start(self) -> None:
        from actions.commands import noop_action, screenshot_fullscreen

        try:
            self._assets = load_inference_assets()
            self._classifier = GestureClassifier(self._assets)
            self._action_handler = GestureActionHandler(
                confidence_threshold=THRESHOLDS.confidence,
                hold_frames=10,
                cooldown_seconds=1.5,
            )
            # Wrap actions so we can emit action_triggered for toasts
            def wrap(gesture: str, callback):
                def w():
                    callback()
                    self.action_triggered.emit(gesture)
                return w
            self._action_handler.register("peace", wrap("peace", lambda: noop_action("peace")))
            self._action_handler.register("fist", wrap("fist", lambda: noop_action("fist")))
            self._action_handler.register("middle_finger", wrap("middle_finger", screenshot_fullscreen))

            self._detector = HandDetector(self._detector_callback)

            cap = open_camera()
        except Exception as exc:  # pragma: no cover - startup failures
            self.error.emit(str(exc))
            self.finished.emit()
            return

        prev_time = time.time()
        start_time = time.time()

        with self._detector.landmarker as landmarker:  # type: ignore[union-attr]
            while not self._stop_flag.is_set() and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                timestamp_ms = int((time.time() - start_time) * 1000)
                landmarker.detect_async(mp_image, timestamp_ms)

                with self._state_lock:
                    local_result = self._current_result

                if (
                    local_result is not None
                    and local_result.hand_landmarks
                    and self._classifier is not None
                    and self._action_handler is not None
                ):
                    prediction, confidence = self._classifier.predict(local_result.hand_landmarks[0])
                else:
                    prediction = "No Hand Detected"
                    confidence = 0.0

                if self._action_handler is not None and self._actions_enabled:
                    self._action_handler.on_frame(prediction, confidence)

                curr_time = time.time()
                time_diff = curr_time - prev_time
                fps = 1.0 / time_diff if time_diff > 0 else 0.0
                prev_time = curr_time

                self.frame_ready.emit(frame, prediction, confidence, fps, local_result)

        cap.release()
        self.finished.emit()

    def stop(self) -> None:
        self._stop_flag.set()

