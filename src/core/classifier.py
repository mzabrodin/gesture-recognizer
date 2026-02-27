from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from src.config import PATHS, THRESHOLDS


@dataclass
class InferenceAssets:
    labels: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    interpreter: Interpreter
    input_details: list
    output_details: list


def load_inference_assets() -> InferenceAssets:
    """Load TFLite model, labels, and scaler parameters for inference."""
    if not PATHS.gesture_classifier.exists():
        raise FileNotFoundError(f"TFLite model not found: {PATHS.gesture_classifier}")

    with PATHS.labels.open("r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]

    with PATHS.scaler_params.open("r", encoding="utf-8") as f:
        scaler_data = json.load(f)
        scaler_mean = np.array(scaler_data["mean"], dtype=np.float32)
        scaler_scale = np.array(scaler_data["scale"], dtype=np.float32)

    interpreter = Interpreter(model_path=str(PATHS.gesture_classifier))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return InferenceAssets(labels, scaler_mean, scaler_scale, interpreter, input_details, output_details)


class GestureClassifier:
    def __init__(self, assets: InferenceAssets) -> None:
        self.assets = assets
        self._smoothed_probs: np.ndarray | None = None
        self._alpha = THRESHOLDS.ema_alpha

    def _features_from_landmarks(self, landmarks: List) -> np.ndarray:
        base_x, base_y = landmarks[0].x, landmarks[0].y
        translated = [(lm.x - base_x, lm.y - base_y) for lm in landmarks]

        flat_abs = [abs(val) for pair in translated for val in pair]
        max_val = max(flat_abs) if flat_abs else 1.0
        if max_val == 0:
            max_val = 1.0

        normalized_coords: list[float] = []
        for x, y in translated:
            normalized_coords.extend([x / max_val, y / max_val])

        features = np.array([normalized_coords], dtype=np.float32)
        scaled_features = (features - self.assets.scaler_mean) / self.assets.scaler_scale
        return scaled_features

    def predict(self, landmarks: List) -> Tuple[str, float]:
        """Return (label, confidence) using EMA-smoothed probabilities."""
        input_tensor = self._features_from_landmarks(landmarks)

        self.assets.interpreter.set_tensor(self.assets.input_details[0]["index"], input_tensor)
        self.assets.interpreter.invoke()
        predictions = self.assets.interpreter.get_tensor(self.assets.output_details[0]["index"])[0]

        if self._smoothed_probs is None:
            self._smoothed_probs = predictions
        else:
            self._smoothed_probs = (self._alpha * predictions) + ((1.0 - self._alpha) * self._smoothed_probs)

        top_index = int(np.argmax(self._smoothed_probs))
        return self.assets.labels[top_index], float(self._smoothed_probs[top_index])
