"""
Gesture-to-action mapping and trigger logic (hold + cooldown).
Use this module to register callbacks for gestures and call on_frame() from the main loop.
"""
import time
from collections.abc import Callable


class GestureActionHandler:
    """
    Tracks gesture predictions and runs registered actions when a gesture
    is held for enough frames and cooldown has passed.
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.65,
        hold_frames: int = 10,
        cooldown_seconds: float = 1.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.hold_frames = hold_frames
        self.cooldown_seconds = cooldown_seconds
        self._registry: dict[str, Callable[[], None]] = {}
        self._last_gesture: str | None = None
        self._hold_count: int = 0
        self._last_trigger_time: float = 0.0

    def register(self, gesture: str, callback: Callable[[], None]) -> None:
        """Register an action to run when the gesture is triggered."""
        self._registry[gesture] = callback

    def on_frame(self, prediction: str, confidence: float) -> None:
        """
        Call every frame with current prediction and confidence.
        Runs the registered action when the gesture is held long enough and cooldown allows.
        """
        skip_conditions = (
            prediction in ("No Hand Detected", "Waiting", "no_gesture")
            or confidence < self.confidence_threshold
            or prediction not in self._registry
        )
        if skip_conditions:
            self._last_gesture = None
            self._hold_count = 0
            return

        if prediction != self._last_gesture:
            self._last_gesture = prediction
            self._hold_count = 0
            return

        self._hold_count += 1
        if self._hold_count < self.hold_frames:
            return

        now = time.monotonic()
        if now - self._last_trigger_time < self.cooldown_seconds:
            return

        self._last_trigger_time = now
        self._hold_count = 0
        try:
            self._registry[prediction]()
        except Exception as e:
            print(f"Gesture action error ({prediction}): {e}")
