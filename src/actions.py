"""Cross-platform action callbacks invoked by the gesture handler."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime

from src.config import SCREENSHOTS_DIR
from src.gesture_handler import GestureAction

log = logging.getLogger(__name__)


def _osascript(script: str) -> None:
    """Execute a short AppleScript snippet (macOS only)."""
    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True, timeout=5)
    except Exception:
        log.exception("osascript failed: %s", script)


def volume_up() -> None:
    """Increase system volume by ~10 %."""
    if sys.platform == "darwin":
        _osascript("set volume output volume ((output volume of (get volume settings)) + 10)")
    else:
        import pyautogui

        pyautogui.press("volumeup")
    log.info("Volume up")


def volume_down() -> None:
    """Decrease system volume by ~10 %."""
    if sys.platform == "darwin":
        _osascript("set volume output volume ((output volume of (get volume settings)) - 10)")
    else:
        import pyautogui

        pyautogui.press("volumedown")
    log.info("Volume down")


def toggle_mute() -> None:
    """Toggle system mute state."""
    if sys.platform == "darwin":
        _osascript("set volume output muted (not (output muted of (get volume settings)))")
    else:
        import pyautogui

        pyautogui.press("volumemute")
    log.info("Mute toggled")


def play_pause() -> None:
    """Send media play/pause key event."""
    if sys.platform == "darwin":
        _osascript('tell application "System Events" to key code 49 using {command down}')
    else:
        import pyautogui

        pyautogui.press("playpause")
    log.info("Play/Pause")


def take_screenshot() -> None:
    """Save a screenshot (cross-platform)."""
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    name = datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
    path = os.path.join(SCREENSHOTS_DIR, name)
    try:
        if sys.platform == "darwin":
            subprocess.run(["screencapture", "-x", path], check=True, capture_output=True, timeout=10)
        else:
            import pyautogui

            screenshot = pyautogui.screenshot()
            screenshot.save(path)
        log.info("Screenshot saved: %s", path)
    except Exception:
        log.exception("Screenshot failed")


def brightness_up() -> None:
    """Increase screen brightness by ~10%."""
    if sys.platform == "darwin":
        _osascript('tell application "System Events" to key code 144')
        log.info("Brightness up (macOS)")
    else:
        try:
            import screen_brightness_control as sbc

            current = sbc.get_brightness()[0]
            sbc.set_brightness(min(100, current + 10))
            log.info("Brightness up (Windows/Linux)")
        except Exception:
            log.exception("Failed to increase brightness")


def brightness_down() -> None:
    """Decrease screen brightness by ~10%."""
    if sys.platform == "darwin":
        _osascript('tell application "System Events" to key code 145')
        log.info("Brightness down (macOS)")
    else:
        try:
            import screen_brightness_control as sbc

            current = sbc.get_brightness()[0]
            sbc.set_brightness(max(0, current - 10))
            log.info("Brightness down (Windows/Linux)")
        except Exception:
            log.exception("Failed to decrease brightness")


def build_default_actions() -> list[GestureAction]:
    """Return the default set of gesture→action bindings."""
    return [
        GestureAction("like", hold_time=0.5, cooldown=0.5, callback=volume_up),
        GestureAction("dislike", hold_time=0.5, cooldown=0.5, callback=volume_down),
        GestureAction("mute", hold_time=0.8, cooldown=1.5, callback=toggle_mute),
        GestureAction("middle_finger", hold_time=1.0, cooldown=2.0, callback=take_screenshot),
        GestureAction("peace", hold_time=1.0, cooldown=2.0, callback=take_screenshot),
        GestureAction("fist", hold_time=0.8, cooldown=1.0, callback=play_pause),
        GestureAction("rock", hold_time=0.8, cooldown=1.5, callback=toggle_mute),
        GestureAction("stop", hold_time=0.6, cooldown=0.6, callback=brightness_up),
        GestureAction("palm", hold_time=0.6, cooldown=0.6, callback=brightness_down),
        GestureAction("ok", hold_time=0.8, cooldown=1.5, callback=play_pause),
        GestureAction("call", hold_time=0.8, cooldown=1.5, callback=play_pause),
    ]
