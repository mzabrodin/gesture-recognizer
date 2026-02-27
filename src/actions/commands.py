from __future__ import annotations

import subprocess
import sys
from datetime import datetime

import pyautogui

from src.actions.handler import GestureActionHandler
from src.config import SCREENSHOTS


def screenshot_fullscreen() -> None:
    """Take a fullscreen screenshot and save it to the configured screenshots directory."""
    SCREENSHOTS.base_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
    path = SCREENSHOTS.base_dir / name

    if sys.platform == "darwin":
        try:
            subprocess.run(
                ["screencapture", "-x", str(path)],
                check=True,
                capture_output=True,
                timeout=10,
            )
        except Exception as exc:  # pragma: no cover - platform-specific
            print(f"Screenshot failed (screencapture): {exc}")
    else:
        try:
            pyautogui.screenshot(str(path))
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"Screenshot failed (pyautogui): {exc}")


def noop_action(name: str) -> None:
    print(f"Gesture recognized: {name}")


def register_default_actions(handler: GestureActionHandler) -> None:
    """Register default gesture → command mapping on the provided handler."""
    handler.register("peace", lambda: noop_action("peace"))
    handler.register("fist", lambda: noop_action("fist"))
    handler.register("middle_finger", screenshot_fullscreen)
