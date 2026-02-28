from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget


class TrayIcon(QSystemTrayIcon):
    def __init__(self, parent: QWidget) -> None:
        icon = parent.windowIcon() if parent.windowIcon() else QIcon()
        super().__init__(icon, parent)

        menu = QMenu(parent)
        show_action = QAction("Show", parent)
        quit_action = QAction("Quit", parent)

        show_action.triggered.connect(parent.showNormal)
        quit_action.triggered.connect(parent.request_quit)

        menu.addAction(show_action)
        menu.addSeparator()
        menu.addAction(quit_action)

        self.setContextMenu(menu)
        self.setToolTip("Gesture Recognizer — клік по іконці: відкрити вікно")
        self.activated.connect(self._on_activated)
        self.show()

    @Slot(QSystemTrayIcon.ActivationReason)
    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        # Клік або подвійний клік по іконці — повернути вікно
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            parent = self.parent()
            if isinstance(parent, QWidget):
                parent.showNormal()
                parent.raise_()
                parent.activateWindow()

