from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mediapipe as mp

from config import PATHS, THRESHOLDS

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


DetectionCallback = callable


@dataclass
class DetectorConfig:
    num_hands: int = 1
    min_detection_confidence: float = THRESHOLDS.min_detection_confidence


class HandDetector:
    def __init__(self, callback) -> None:
        self._callback = callback
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(PATHS.hand_landmarker)),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._on_result,
            num_hands=1,
            min_hand_detection_confidence=THRESHOLDS.min_detection_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def _on_result(
        self,
        result: Optional[HandLandmarkerResult],
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        self._callback(result, output_image, timestamp_ms)

    @property
    def landmarker(self) -> HandLandmarker:
        return self._landmarker

