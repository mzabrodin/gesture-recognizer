"""Gesture Recognizer — application entry point."""

from __future__ import annotations

import logging
import os
import sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

import cv2
from PySide6.QtWidgets import QApplication

from src.actions import build_default_actions
from src.camera_thread import CameraThread
from src.gesture_handler import GestureActionHandler
from src.inference import InferenceEngine
from src.inference_thread import InferenceThread
from src.main_window import MainWindow
from src.settings_manager import SettingsManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def request_camera_permission_if_mac() -> None:
    """Trigger the macOS camera-permission dialog on the main thread.

    AVFoundation requires the very first ``cv2.VideoCapture`` open to happen on
    the main thread so the system permission popup can be displayed.  On
    non-macOS platforms this is a no-op.
    """
    if sys.platform != "darwin":
        return
    try:
        cap = cv2.VideoCapture(0)
        cap.read()
        cap.release()
        log.info("Camera permission granted (macOS pre-check)")
    except Exception:
        log.exception("Camera permission pre-check failed")


def main() -> None:
    """Bootstrap and run the application."""
    app = QApplication(sys.argv)

    settings_mgr = SettingsManager()
    log.info("Settings loaded from %s", settings_mgr._path)

    engine = InferenceEngine()
    log.info("Inference engine ready (%d labels)", len(engine.labels))

    handler = GestureActionHandler(confidence_threshold=settings_mgr.settings.confidence_threshold)
    for action in build_default_actions():
        if settings_mgr.settings.active_gestures.get(action.gesture, True):
            handler.register(action)

    camera = CameraThread()
    inference = InferenceThread(camera, engine, handler)

    window = MainWindow(settings_mgr, handler)
    inference.result_ready.connect(window.update_frame)
    window.show()

    camera.start()
    inference.start()

    def shutdown() -> None:
        log.info("Shutting down…")
        inference.stop()
        camera.stop()
        settings_mgr.save()

    app.aboutToQuit.connect(shutdown)

    sys.exit(app.exec())


if __name__ == "__main__":
    request_camera_permission_if_mac()
    main()
