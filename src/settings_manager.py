"""Persistent user settings backed by a JSON file in the platform config dir."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from platformdirs import user_config_dir

_APP_NAME = "gesture-recognizer"
_SETTINGS_FILE = "settings.json"


@dataclass
class UserSettings:
    """User-editable settings that persist across sessions."""

    confidence_threshold: float = 0.65
    show_skeleton: bool = True
    hold_time: float = 0.5
    cooldown: float = 0.5
    active_gestures: dict[str, bool] = field(
        default_factory=lambda: {
            "like": True,
            "dislike": True,
            "mute": True,
            "middle_finger": True,
            "peace": True,
            "fist": True,
            "rock": True,
            "stop": True,
            "call": True,
            "ok": True,
            "palm": True,
        }
    )


class SettingsManager:
    """Read / write ``UserSettings`` to a JSON file in the user config directory."""

    def __init__(self) -> None:
        self._dir = user_config_dir(_APP_NAME, ensure_exists=True)
        self._path = os.path.join(self._dir, _SETTINGS_FILE)
        self.settings = self._load()

    def _load(self) -> UserSettings:
        if not os.path.isfile(self._path):
            return UserSettings()
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data: dict[str, Any] = json.load(fh)
            defaults = UserSettings()
            return UserSettings(
                confidence_threshold=data.get("confidence_threshold", defaults.confidence_threshold),
                show_skeleton=data.get("show_skeleton", defaults.show_skeleton),
                hold_time=data.get("hold_time", defaults.hold_time),
                cooldown=data.get("cooldown", defaults.cooldown),
                active_gestures={**defaults.active_gestures, **data.get("active_gestures", {})},
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            return UserSettings()

    def save(self) -> None:
        """Persist current settings to disk."""
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self.settings), fh, indent=2)
