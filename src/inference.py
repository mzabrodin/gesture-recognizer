"""Inference engine — model loading and landmark classification."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from src.config import INFERENCE_DEFAULTS, MODEL_PATHS


class InferenceEngine:
    """Loads TFLite model + scaler and classifies hand landmarks with EMA smoothing."""

    def __init__(self) -> None:
        with open(MODEL_PATHS.labels, "r", encoding="utf-8") as fh:
            self.labels: list[str] = [line.strip() for line in fh if line.strip()]

        with open(MODEL_PATHS.scaler, "r", encoding="utf-8") as fh:
            scaler: dict[str, Any] = json.load(fh)
        self._mean = np.array(scaler["mean"], dtype=np.float32)
        self._scale = np.array(scaler["scale"], dtype=np.float32)

        self._interpreter = Interpreter(model_path=MODEL_PATHS.tflite)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        self._smoothed: np.ndarray | None = None
        self._alpha: float = INFERENCE_DEFAULTS.ema_alpha

    def reset(self) -> None:
        """Clear EMA state."""
        self._smoothed = None

    def predict(self, landmarks: list[Any]) -> tuple[str, float]:
        """Classify a list of 21 hand landmarks and return ``(label, confidence)``."""
        base_x, base_y = landmarks[0].x, landmarks[0].y
        translated = [(lm.x - base_x, lm.y - base_y) for lm in landmarks]

        flat_abs = [abs(v) for pair in translated for v in pair]
        max_val = max(flat_abs) or 1.0

        coords: list[float] = []
        for x, y in translated:
            coords.extend([x / max_val, y / max_val])

        features = np.array([coords], dtype=np.float32)
        scaled = (features - self._mean) / self._scale

        self._interpreter.set_tensor(self._input_details[0]["index"], scaled)
        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(self._output_details[0]["index"])[0]

        if self._smoothed is None:
            self._smoothed = raw.copy()
        else:
            self._smoothed = self._alpha * raw + (1.0 - self._alpha) * self._smoothed

        idx = int(np.argmax(self._smoothed))
        return self.labels[idx], float(self._smoothed[idx])
