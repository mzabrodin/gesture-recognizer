from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    base_dir: Path = Path(__file__).resolve().parent.parent
    models_dir: Path = base_dir / "models"
    hand_landmarker: Path = models_dir / "hand_landmarker.task"
    gesture_classifier: Path = models_dir / "gesture_classifier.tflite"
    labels: Path = models_dir / "labels.txt"
    scaler_params: Path = models_dir / "scaler_params.json"


@dataclass(frozen=True)
class Thresholds:
    confidence: float = 0.65
    min_detection_confidence: float = 0.5
    ema_alpha: float = 0.4


@dataclass(frozen=True)
class Video:
    width: int = 480
    height: int = 480
    # як на початку: основна камера 0, але можна переоприділити через змінну середовища
    camera_index: int = int(os.getenv("GESTURE_REC_CAMERA_INDEX", "0"))


@dataclass(frozen=True)
class Colors:
    text_ok: tuple[int, int, int] = (0, 255, 0)
    text_low_conf: tuple[int, int, int] = (255, 0, 0)
    text_error: tuple[int, int, int] = (0, 0, 255)
    skeleton: tuple[int, int, int] = (255, 255, 255)
    joint: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class Screenshots:
    # By default save to user's Pictures/GestureScreenshots
    base_dir: Path = Path(os.path.expanduser("~/Pictures")) / "GestureScreenshots"


@dataclass(frozen=True)
class DataPaths:
    base_dir: Path = Paths.base_dir / "data"
    raw_dir: Path = base_dir / "raw"
    raw_train: Path = raw_dir / "train"
    processed_dir: Path = base_dir / "processed"
    processed_landmarks: Path = processed_dir / "landmarks.csv"


@dataclass(frozen=True)
class Logs:
    base_dir: Path = Paths.base_dir / "logs"


PATHS = Paths()
THRESHOLDS = Thresholds()
VIDEO = Video()
COLORS = Colors()
SCREENSHOTS = Screenshots()
DATA = DataPaths()
LOGS = Logs()
