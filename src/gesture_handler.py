"""Time-based gesture action handler with per-gesture hold and cooldown."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from src.config import IGNORE_GESTURES

log = logging.getLogger(__name__)


@dataclass
class GestureAction:
    """Descriptor for a single registered gesture action."""

    gesture: str
    hold_time: float
    cooldown: float
    callback: Callable[[], None]


class GestureActionHandler:
    """Universal gesture action registry with time-based hold verification and cooldown."""

    def __init__(self, confidence_threshold: float = 0.65) -> None:
        self.confidence_threshold = confidence_threshold
        self._actions: dict[str, GestureAction] = {}
        self._current_gesture: str | None = None
        self._hold_start: float = 0.0
        self._cooldowns: dict[str, float] = {}

    def register(self, action: GestureAction) -> None:
        """Add a gesture action to the registry."""
        self._actions[action.gesture] = action

    def unregister(self, gesture: str) -> None:
        """Remove a gesture from the registry."""
        self._actions.pop(gesture, None)
        self._cooldowns.pop(gesture, None)

    @property
    def registered_gestures(self) -> list[str]:
        """Return list of registered gesture names."""
        return list(self._actions)

    def on_prediction(self, gesture: str, confidence: float) -> None:
        """Feed a new prediction from the inference thread.

        The handler checks hold duration and cooldown before invoking any callback.
        """
        now = time.monotonic()

        if gesture in IGNORE_GESTURES or confidence < self.confidence_threshold or gesture not in self._actions:
            self._current_gesture = None
            return

        if gesture != self._current_gesture:
            self._current_gesture = gesture
            self._hold_start = now
            return

        action = self._actions[gesture]
        elapsed = now - self._hold_start

        if elapsed < action.hold_time:
            return

        last_trigger = self._cooldowns.get(gesture, 0.0)
        if now - last_trigger < action.cooldown:
            return

        self._cooldowns[gesture] = now
        self._hold_start = now
        try:
            action.callback()
        except Exception:
            log.exception("Gesture action error (%s)", gesture)
