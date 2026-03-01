from __future__ import annotations

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.tray_icon import TrayIcon


def main() -> None:
    app = QApplication([])
    # Простий темний стиль для всього застосунку
    app.setStyleSheet(
        """
        QMainWindow {
            background-color: #121212;
            color: #f0f0f0;
        }
        QLabel {
            color: #f0f0f0;
        }
        QGroupBox {
            border: 1px solid #333333;
            border-radius: 8px;
            margin-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #bbbbbb;
        }
        QPushButton {
            background-color: #2f6ad9;
            color: white;
            border-radius: 6px;
            padding: 6px 14px;
        }
        QPushButton:hover {
            background-color: #3b7cff;
        }
        QPushButton:pressed {
            background-color: #2757b5;
        }
        QStatusBar {
            background-color: #181818;
        }
        QToolBar {
            background-color: #181818;
            border-bottom: 1px solid #333333;
        }
        QCheckBox {
            color: #f0f0f0;
        }
        """
    )
    window = MainWindow()
    window.show()

    tray = TrayIcon(window)
    tray.show()

    app.exec()


if __name__ == "__main__":
    main()

