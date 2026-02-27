from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config import COLORS
from core.detector import HandLandmarkerResult
from core.pipeline import HAND_CONNECTIONS


def draw_overlays(
    image: np.ndarray,
    result: Optional[HandLandmarkerResult],
    prediction: str,
    confidence: float,
    fps: float,
) -> np.ndarray:
    annotated_image = image.copy()
    height, width, _ = image.shape

    if prediction in {"No Hand Detected", "Waiting"}:
        text = prediction
        color = COLORS.text_error
    elif confidence < 0.65:
        text = f"? {prediction} ({confidence * 100:.1f}%)"
        color = COLORS.text_low_conf
    else:
        text = f"{prediction} ({confidence * 100:.1f}%)"
        color = COLORS.text_ok

    cv2.putText(
        annotated_image,
        text,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv2.LINE_AA,
    )

    fps_text = f"FPS: {int(fps)}"
    cv2.putText(
        annotated_image,
        fps_text,
        (width - 150, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    if result and result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points: list[tuple[int, int]] = []
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))

            for connection in HAND_CONNECTIONS:
                start_point = points[connection[0]]
                end_point = points[connection[1]]
                cv2.line(annotated_image, start_point, end_point, COLORS.skeleton, 2)

            for point in points:
                cv2.circle(annotated_image, point, 5, COLORS.joint, -1)

    return annotated_image

