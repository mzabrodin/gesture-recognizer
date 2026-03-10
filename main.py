"""Gesture Recognizer — application entry point."""

from __future__ import annotations

import logging
import os
import sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

import cv2
from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from src.actions import build_actions
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


def _create_native_status_item(show_callback, quit_callback) -> object | None:
    """Create a macOS native NSStatusItem using ctypes (bypasses Qt)."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        appkit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit"))  # noqa: F841

        # Helpers
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        def msg(obj, sel_name, *args, argtypes=None):
            sel = objc.sel_registerName(sel_name)
            if argtypes:
                objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + argtypes
            else:
                objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            return objc.objc_msgSend(obj, sel, *args)

        # NSStatusBar.systemStatusBar
        NSStatusBar = objc.objc_getClass(b"NSStatusBar")
        status_bar = msg(NSStatusBar, b"systemStatusBar")

        # statusBar.statusItemWithLength: -1.0 (NSVariableStatusItemLength)
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double]
        status_item = objc.objc_msgSend(
            status_bar, objc.sel_registerName(b"statusItemWithLength:"),
            ctypes.c_double(-1.0),
        )

        # Get the button and set its title
        button = msg(status_item, b"button")
        NSString = objc.objc_getClass(b"NSString")
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
        title = objc.objc_msgSend(NSString, objc.sel_registerName(b"stringWithUTF8String:"), b"\xe2\x9c\x8b")  # ✋ emoji
        msg(button, b"setTitle:", title, argtypes=[ctypes.c_void_p])

        log.info("Native macOS status item created")
        return status_item
    except Exception:
        log.exception("Failed to create native status item")
        return None


def main() -> None:
    """Bootstrap and run the application."""
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    settings_mgr = SettingsManager()
    log.info("Settings loaded from %s", settings_mgr._path)

    engine = InferenceEngine()
    log.info("Inference engine ready (%d labels)", len(engine.labels))

    handler = GestureActionHandler(confidence_threshold=settings_mgr.settings.confidence_threshold)
    for action in build_actions(settings_mgr.settings.gesture_actions):
        if settings_mgr.settings.active_gestures.get(action.gesture, True):
            handler.register(action)

    camera = CameraThread()
    inference = InferenceThread(camera, engine, handler)

    window = MainWindow(settings_mgr, handler)
    inference.result_ready.connect(window.update_frame)

    # --- System tray ---
    def show_window() -> None:
        window.show()
        window.raise_()
        window.activateWindow()

    # Try native macOS status bar first, fall back to QSystemTrayIcon
    native_item = _create_native_status_item(show_window, app.quit)

    tray = QSystemTrayIcon(app)
    tray_menu = QMenu(window)
    tray_menu.addAction("Show Window", show_window)
    tray_menu.addSeparator()
    tray_menu.addAction("Quit", app.quit)
    tray.setContextMenu(tray_menu)

    if native_item is None:
        # Non-macOS: use QSystemTrayIcon normally
        from PySide6.QtGui import QColor, QPainter, QPixmap
        size = 64
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(0, 200, 255))
        painter.setPen(QColor(0, 150, 200))
        painter.drawEllipse(4, 4, size - 8, size - 8)
        painter.end()
        tray.setIcon(QIcon(pixmap))
        tray.setToolTip("Gesture Recognizer")
        tray.activated.connect(
            lambda reason: show_window()
            if reason == QSystemTrayIcon.ActivationReason.Trigger
            else None
        )
        tray.show()

    window.set_tray(tray)
    window.show()

    camera.start()
    inference.start()

    def shutdown() -> None:
        log.info("Shutting down…")
        tray.hide()
        inference.stop()
        camera.stop()
        settings_mgr.save()

    app.aboutToQuit.connect(shutdown)

    sys.exit(app.exec())


if __name__ == "__main__":
    request_camera_permission_if_mac()
    main()
