from __future__ import annotations

import subprocess
import sys

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.config import SCREENSHOTS, THRESHOLDS
from src.core.pipeline import PipelineWorker
from src.gui.overlay import draw_overlays


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gesture Recognizer")
        self.resize(1100, 700)
        self._camera_visible = True
        self._quit_requested = False
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._clear_toast)

        # --- Central layout: video on the left, info panel on the right ---
        central = QWidget(self)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(16)

        # Video area
        self._video_label = QLabel(alignment=Qt.AlignCenter)
        self._video_label.setMinimumSize(480, 360)
        self._video_label.setFrameShape(QFrame.StyledPanel)
        self._video_label.setStyleSheet(
            """
            QLabel {
                background-color: #111111;
                border-radius: 12px;
                border: 1px solid #333333;
            }
            """
        )

        # Info / controls panel
        side_panel = QVBoxLayout()
        side_panel.setSpacing(12)

        title = QLabel("Gesture Recognizer")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)

        subtitle = QLabel("Real-time hand gesture control")
        subtitle.setStyleSheet("color: #aaaaaa;")

        header_box = QVBoxLayout()
        header_box.addWidget(title)
        header_box.addWidget(subtitle)

        # Status group
        status_group = QGroupBox("Current status")
        status_layout = QVBoxLayout(status_group)

        self._prediction_label = QLabel("Prediction: —")
        self._prediction_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        self._confidence_label = QLabel("Confidence: —")
        self._fps_label = QLabel("FPS: —")

        status_layout.addWidget(self._prediction_label)
        status_layout.addWidget(self._confidence_label)
        status_layout.addWidget(self._fps_label)

        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        self._camera_checkbox = QCheckBox("Show camera")
        self._camera_checkbox.setChecked(True)
        self._camera_checkbox.stateChanged.connect(self._on_camera_visible_changed)
        actions_layout.addWidget(self._camera_checkbox)

        self._actions_checkbox = QCheckBox("Enable OS actions (screenshots, etc.)")
        self._actions_checkbox.setChecked(True)
        self._actions_checkbox.stateChanged.connect(lambda state: self._worker.set_actions_enabled(bool(state)))
        actions_layout.addWidget(self._actions_checkbox)

        # Confidence threshold slider (50–95%)
        conf_label = QLabel("Confidence threshold:")
        actions_layout.addWidget(conf_label)
        self._confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self._confidence_slider.setRange(50, 95)
        self._confidence_slider.setValue(int(THRESHOLDS.confidence * 100))
        self._confidence_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._confidence_slider.setTickInterval(5)
        self._confidence_slider.valueChanged.connect(self._on_confidence_changed)
        actions_layout.addWidget(self._confidence_slider)
        self._confidence_value_label = QLabel(f"{int(THRESHOLDS.confidence * 100)}%")
        self._confidence_value_label.setStyleSheet("color: #aaaaaa;")
        actions_layout.addWidget(self._confidence_value_label)

        # Minimize to tray when closing
        self._minimize_to_tray_checkbox = QCheckBox("Minimize to tray when closing")
        self._minimize_to_tray_checkbox.setChecked(False)
        actions_layout.addWidget(self._minimize_to_tray_checkbox)
        tray_hint = QLabel("Повернути вікно: клік по іконці в треї або Show у меню")
        tray_hint.setStyleSheet("color: #888; font-size: 11px;")
        tray_hint.setWordWrap(True)
        actions_layout.addWidget(tray_hint)

        # Gesture legend — що який жест робить
        legend_group = QGroupBox("Gesture shortcuts")
        legend_layout = QVBoxLayout(legend_group)
        legend_text = QLabel("peace → (no action)\nfist → (no action)\nmiddle_finger → Screenshot")
        legend_text.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        legend_text.setWordWrap(True)
        legend_layout.addWidget(legend_text)

        # Кнопка відкрити папку зі скріншотами
        self._open_screenshots_btn = QPushButton("Open screenshots folder")
        self._open_screenshots_btn.clicked.connect(self._open_screenshots_folder)
        legend_layout.addWidget(self._open_screenshots_btn)

        # Control buttons
        buttons_layout = QHBoxLayout()
        self._quit_button = QPushButton("Quit")
        self._quit_button.clicked.connect(self.close)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(self._quit_button)

        side_panel.addLayout(header_box)
        side_panel.addWidget(status_group)
        side_panel.addWidget(actions_group)
        side_panel.addWidget(legend_group)
        side_panel.addStretch(1)
        side_panel.addLayout(buttons_layout)

        main_layout.addWidget(self._video_label, stretch=3)
        main_layout.addLayout(side_panel, stretch=2)

        self.setCentralWidget(central)

        self._status = QStatusBar()
        self.setStatusBar(self._status)

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._thread = QThread(self)
        self._worker = PipelineWorker()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.start)
        self._worker.frame_ready.connect(self.on_frame_ready)
        self._worker.action_triggered.connect(self._on_action_triggered)
        self._worker.error.connect(self.on_error)
        self._worker.finished.connect(self._thread.quit)

        self._thread.start()

    def _on_camera_visible_changed(self, state: int) -> None:
        self._set_camera_visible(state == Qt.CheckState.Checked.value)

    def _on_confidence_changed(self, value: int) -> None:
        self._confidence_value_label.setText(f"{value}%")
        self._worker.set_confidence_threshold(value / 100.0)

    def _on_action_triggered(self, gesture: str) -> None:
        if gesture == "middle_finger":
            self._status.showMessage("Screenshot saved")
            self._toast_timer.stop()
            self._toast_timer.start(3000)

    def _clear_toast(self) -> None:
        self._status.showMessage("")

    def request_quit(self) -> None:
        """Викликається з трею (Quit) — справжнє закриття застосунку."""
        self._quit_requested = True
        self.close()

    def _set_camera_visible(self, visible: bool) -> None:
        self._camera_visible = visible
        self._video_label.setVisible(visible)
        if not visible:
            self._video_label.clear()

    def _open_screenshots_folder(self) -> None:
        """Відкрити папку зі скріншотами в системному файловому менеджері."""
        path = SCREENSHOTS.base_dir
        path.mkdir(parents=True, exist_ok=True)
        abs_path = str(path.resolve())
        if sys.platform == "darwin":
            subprocess.run(["open", abs_path], check=False)
        elif sys.platform == "win32":
            subprocess.run(["explorer", abs_path], check=False)
        else:
            subprocess.run(["xdg-open", abs_path], check=False)

    @Slot(np.ndarray, str, float, float, object)
    def on_frame_ready(
        self,
        frame: np.ndarray,
        prediction: str,
        confidence: float,
        fps: float,
        result: object,
    ) -> None:
        if self._camera_visible:
            annotated = draw_overlays(frame, result, prediction, confidence, fps)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            target_size = self._video_label.size()
            if not target_size.isEmpty():
                pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._video_label.setPixmap(pixmap)

        self._prediction_label.setText(f"Prediction: {prediction}")
        self._confidence_label.setText(f"Confidence: {confidence * 100:.1f}%")
        self._fps_label.setText(f"FPS: {fps:.1f}")
        if fps >= 20:
            fps_color = "#4ade80"
        elif fps >= 10:
            fps_color = "#facc15"
        else:
            fps_color = "#f87171"
        self._fps_label.setStyleSheet(f"font-weight: 600; color: {fps_color};")
        if not self._toast_timer.isActive():
            self._status.showMessage(f"{prediction} ({confidence * 100:.1f}%) | FPS: {fps:.1f}")

    @Slot(str)
    def on_error(self, message: str) -> None:
        self._status.showMessage(f"Error: {message}")

    def closeEvent(self, event) -> None:
        if self._quit_requested or not self._minimize_to_tray_checkbox.isChecked():
            self._worker.stop()
            self._thread.quit()
            self._thread.wait(2000)
            super().closeEvent(event)
            event.accept()
        else:
            self.hide()
            event.ignore()
