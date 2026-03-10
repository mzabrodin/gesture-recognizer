"""PySide6 main window — video display, status labels, settings panel."""

from __future__ import annotations

import time
from functools import partial

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from src.actions import ACTION_REGISTRY, build_single_action
from src.config import HAND_CONNECTIONS, WINDOW_DEFAULTS
from src.gesture_handler import GestureActionHandler
from src.settings_manager import SettingsManager


class GestureActionsDialog(QDialog):
    """Modal dialog for customizing gesture→action mappings."""

    def __init__(
        self,
        settings_mgr: SettingsManager,
        handler: GestureActionHandler,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings_mgr
        self._handler = handler
        self._combos: dict[str, QComboBox] = {}

        self.setWindowTitle("Customize Actions")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        header = QLabel("Assign an action to each gesture:")
        header.setFont(QFont("Menlo", 13, QFont.Weight.Bold))
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setColumnStretch(1, 1)

        action_names = list(ACTION_REGISTRY.keys())

        for row_idx, (gesture, active) in enumerate(
            sorted(self._settings.settings.active_gestures.items())
        ):
            label = QLabel(gesture)
            label.setFont(QFont("Menlo", 12))
            if not active:
                label.setStyleSheet("color: #888;")
            grid.addWidget(label, row_idx, 0)

            combo = QComboBox()
            combo.addItems(action_names)
            current_action = self._settings.settings.gesture_actions.get(gesture, "")
            combo.setCurrentText(current_action)
            combo.currentTextChanged.connect(partial(self._on_action_changed, gesture))
            grid.addWidget(combo, row_idx, 1)

            self._combos[gesture] = combo

        scroll.setWidget(inner)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        layout.addWidget(buttons)

    def _on_action_changed(self, gesture: str, action_name: str) -> None:
        self._settings.settings.gesture_actions[gesture] = action_name
        if self._settings.settings.active_gestures.get(gesture, False):
            action = build_single_action(gesture, action_name)
            if action:
                self._handler.register(action)
        self._settings.save()


class MainWindow(QMainWindow):
    """Application main window with live video, overlay labels and a settings panel."""

    def __init__(
        self,
        settings_mgr: SettingsManager,
        handler: GestureActionHandler,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings_mgr
        self._handler = handler
        self._prev_time = time.monotonic()
        self._fps = 0.0

        self.setWindowTitle(WINDOW_DEFAULTS.title)
        self.resize(WINDOW_DEFAULTS.width, WINDOW_DEFAULTS.height)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        video_col = QVBoxLayout()
        root_layout.addLayout(video_col, stretch=3)

        self._video_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(480, 360)
        self._video_label.setStyleSheet("background-color: #111;")
        video_col.addWidget(self._video_label, stretch=1)

        status_row = QHBoxLayout()
        video_col.addLayout(status_row)

        self._gesture_label = QLabel("Waiting…")
        self._gesture_label.setFont(QFont("Menlo", 16, QFont.Weight.Bold))
        status_row.addWidget(self._gesture_label)

        status_row.addStretch()

        self._conf_label = QLabel("")
        self._conf_label.setFont(QFont("Menlo", 14))
        status_row.addWidget(self._conf_label)

        status_row.addStretch()

        self._fps_label = QLabel("FPS: 0")
        self._fps_label.setFont(QFont("Menlo", 14))
        status_row.addWidget(self._fps_label)

        panel = QVBoxLayout()
        root_layout.addLayout(panel, stretch=1)

        conf_group = QGroupBox("Confidence threshold")
        conf_layout = QVBoxLayout(conf_group)
        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(30, 95)
        self._conf_slider.setValue(int(self._settings.settings.confidence_threshold * 100))
        self._conf_slider_label = QLabel(f"{self._settings.settings.confidence_threshold:.0%}")
        self._conf_slider.valueChanged.connect(self._on_confidence_changed)
        conf_layout.addWidget(self._conf_slider)
        conf_layout.addWidget(self._conf_slider_label)
        panel.addWidget(conf_group)

        self._skeleton_cb = QCheckBox("Show hand skeleton")
        self._skeleton_cb.setChecked(self._settings.settings.show_skeleton)
        self._skeleton_cb.toggled.connect(self._on_skeleton_toggled)
        panel.addWidget(self._skeleton_cb)

        gestures_group = QGroupBox("Active gestures")
        gestures_scroll = QScrollArea()
        gestures_scroll.setWidgetResizable(True)
        gestures_inner = QWidget()
        gestures_layout = QVBoxLayout(gestures_inner)

        self._gesture_cbs: dict[str, QCheckBox] = {}
        for gesture, active in sorted(self._settings.settings.active_gestures.items()):
            cb = QCheckBox(gesture)
            cb.setChecked(active)
            cb.toggled.connect(partial(self._on_gesture_toggled, gesture))
            gestures_layout.addWidget(cb)
            self._gesture_cbs[gesture] = cb
        gestures_layout.addStretch()
        gestures_scroll.setWidget(gestures_inner)

        g_outer = QVBoxLayout(gestures_group)
        g_outer.addWidget(gestures_scroll)
        panel.addWidget(gestures_group)

        self._actions_btn = QPushButton("Customize actions…")
        self._actions_btn.clicked.connect(self._open_actions_dialog)
        panel.addWidget(self._actions_btn)

        panel.addStretch()


    def _open_actions_dialog(self) -> None:
        dialog = GestureActionsDialog(self._settings, self._handler, parent=self)
        dialog.exec()

    def _on_confidence_changed(self, value: int) -> None:
        pct = value / 100.0
        self._settings.settings.confidence_threshold = pct
        self._handler.confidence_threshold = pct
        self._conf_slider_label.setText(f"{pct:.0%}")
        self._settings.save()

    def _on_skeleton_toggled(self, checked: bool) -> None:
        self._settings.settings.show_skeleton = checked
        self._settings.save()

    def _on_gesture_toggled(self, gesture: str, checked: bool) -> None:
        self._settings.settings.active_gestures[gesture] = checked
        if checked:
            action_name = self._settings.settings.gesture_actions.get(gesture, "")
            action = build_single_action(gesture, action_name)
            if action:
                self._handler.register(action)
        else:
            self._handler.unregister(gesture)
        self._settings.save()


    @Slot(np.ndarray, str, float, list)
    def update_frame(
        self,
        frame: np.ndarray,
        gesture: str,
        confidence: float,
        landmarks: list[tuple[float, float]],
    ) -> None:
        """Render a new camera frame and update status labels."""
        now = time.monotonic()
        dt = now - self._prev_time
        self._fps = 1.0 / dt if dt > 0 else 0.0
        self._prev_time = now

        rgb = frame[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)

        if self._settings.settings.show_skeleton and landmarks:
            painter = QPainter(pixmap)
            pen = QPen(QColor(255, 255, 255), 2)
            painter.setPen(pen)
            for a, b in HAND_CONNECTIONS:
                if a < len(landmarks) and b < len(landmarks):
                    ax, ay = landmarks[a]
                    bx, by = landmarks[b]
                    painter.drawLine(int(ax), int(ay), int(bx), int(by))
            painter.setPen(QPen(QColor(0, 200, 255), 1))
            painter.setBrush(QColor(0, 200, 255))
            for px, py in landmarks:
                painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)
            painter.end()

        scaled = pixmap.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

        if gesture in ("No Hand Detected", "Waiting"):
            self._gesture_label.setText(gesture)
            self._gesture_label.setStyleSheet("color: #e74c3c;")
            self._conf_label.setText("")
        elif confidence < self._settings.settings.confidence_threshold:
            self._gesture_label.setText(f"? {gesture}")
            self._gesture_label.setStyleSheet("color: #3498db;")
            self._conf_label.setText(f"{confidence:.0%}")
        else:
            self._gesture_label.setText(gesture)
            self._gesture_label.setStyleSheet("color: #2ecc71;")
            self._conf_label.setText(f"{confidence:.0%}")

        self._fps_label.setText(f"FPS: {int(self._fps)}")

    def set_tray(self, tray: QSystemTrayIcon) -> None:
        """Store a reference to the tray icon so the window can hide instead of quit."""
        self._tray = tray

    def closeEvent(self, event) -> None:
        """Hide to tray instead of quitting (if tray is available)."""
        self._settings.save()
        if hasattr(self, "_tray") and self._tray.isVisible():
            event.ignore()
            self.hide()
            _macos_set_dock_icon_visible(False)
        else:
            super().closeEvent(event)

    def show(self) -> None:
        _macos_set_dock_icon_visible(True)
        super().show()


def _macos_set_dock_icon_visible(visible: bool) -> None:
    """Show or hide the Dock icon on macOS. No-op on other platforms."""
    import sys
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        NSApp = objc.objc_msgSend(
            objc.objc_getClass(b"NSApplication"),
            objc.sel_registerName(b"sharedApplication"),
        )

        # 0 = Regular (shows in Dock), 1 = Accessory (hides from Dock)
        policy = 0 if visible else 1
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        objc.objc_msgSend(NSApp, objc.sel_registerName(b"setActivationPolicy:"), policy)
    except Exception:
        pass
