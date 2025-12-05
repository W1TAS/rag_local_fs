from PySide6 import QtWidgets, QtCore, QtGui
import os
from cache import get_cache_root


class LogWindow(QtWidgets.QDialog):
    def __init__(self, parent=None, log_path=None, refresh_interval_ms=1000):
        super().__init__(parent)
        self.setWindowTitle("Логи")
        self.resize(800, 600)
        self.log_path = log_path or os.path.join(get_cache_root(), "app.log")

        layout = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QTextEdit()
        self.text.setReadOnly(True)
        font = QtGui.QFont("Consolas")
        font.setPointSize(10)
        self.text.setFont(font)

        btn_layout = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Обновить")
        self.refresh_btn.clicked.connect(self.load)
        self.clear_btn = QtWidgets.QPushButton("Очистить лог")
        self.clear_btn.clicked.connect(self.clear_log)
        self.open_folder_btn = QtWidgets.QPushButton("Открыть папку")
        self.open_folder_btn.clicked.connect(self.open_folder)
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.open_folder_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)
        layout.addWidget(self.text, 1)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.load)
        self.timer.start(refresh_interval_ms)

        self.load()

    def load(self):
        try:
            if not os.path.exists(self.log_path):
                self.text.setPlainText("Лог файл не найден: %s" % self.log_path)
                return
            # read last N bytes/lines to avoid huge loads
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
            # keep last ~2000 lines
            lines = data.splitlines()
            if len(lines) > 2000:
                lines = lines[-2000:]
            self.text.setPlainText("\n".join(lines))
            self.text.moveCursor(QtGui.QTextCursor.End)
        except Exception as e:
            self.text.setPlainText(f"Ошибка чтения лога: {e}")

    def clear_log(self):
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("")
            self.load()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось очистить лог: {e}")

    def open_folder(self):
        folder = os.path.dirname(self.log_path)
        try:
            if not os.path.exists(folder):
                # Try to create the folder so explorer can open it
                os.makedirs(folder, exist_ok=True)
            # On Windows prefer os.startfile to open Explorer reliably
            if os.name == 'nt':
                try:
                    os.startfile(folder)
                    return
                except Exception:
                    pass
            # Fallback to Qt method
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))
        except Exception as e:
            QtWidgets.QMessageBox.information(self, "Не найдено", f"{folder}\nОшибка: {e}")
