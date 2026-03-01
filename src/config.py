"""Static application constants — frozen dataclasses only."""

from __future__ import annotations

import os
from dataclasses import dataclass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class ModelPaths:
    """Paths to model artefacts resolved relative to the project root."""

    root: str = os.path.join(_PROJECT_ROOT, "models")
    hand_landmarker: str = os.path.join(root, "hand_landmarker.task")
    tflite: str = os.path.join(root, "gesture_classifier.tflite")
    labels: str = os.path.join(root, "labels.txt")
    scaler: str = os.path.join(root, "scaler_params.json")


@dataclass(frozen=True)
class InferenceDefaults:
    """Default values for the inference pipeline."""

    ema_alpha: float = 0.4
    confidence_threshold: float = 0.65
    num_hands: int = 1
    min_detection_confidence: float = 0.5
    inference_interval_ms: int = 80  # ~12 FPS


@dataclass(frozen=True)
class CameraDefaults:
    """Default camera capture parameters."""

    width: int = 640
    height: int = 480
    device_index: int = 0


@dataclass(frozen=True)
class WindowDefaults:
    """Default main-window geometry."""

    title: str = "Gesture Recognizer"
    width: int = 900
    height: int = 640


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
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
)

IGNORE_GESTURES: frozenset[str] = frozenset({"no_gesture", "No Hand Detected", "Waiting"})

SCREENSHOTS_DIR: str = os.path.join(_PROJECT_ROOT, "screenshots")

MODEL_PATHS = ModelPaths()
INFERENCE_DEFAULTS = InferenceDefaults()
CAMERA_DEFAULTS = CameraDefaults()
WINDOW_DEFAULTS = WindowDefaults()
