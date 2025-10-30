# src/ui/main_window.py
import os
import shutil
from PyQt5 import QtWidgets, QtCore, QtGui
from coordinator import RAGCoordinator
from ui.tray import create_tray_icon


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.coordinator = RAGCoordinator.instance(folder_path)
        self.chat_html = ""
        self.typing_html = ""  # ← ФИКС: Инициализация
        self.typing_timer = None

        self.setWindowTitle("RAG Assistant")
        self.setGeometry(300, 200, 960, 720)
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setAcceptDrops(True)

        self._setup_ui()
        self._setup_tray()
        self._connect_signals()

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        # Чат — QTextBrowser
        self.chat_browser = QtWidgets.QTextBrowser()
        self.chat_browser.setOpenExternalLinks(True)
        self.chat_browser.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.chat_browser.setStyleSheet("background: #000000; border: none;")
        layout.addWidget(self.chat_browser, 1)

        # Поле ввода
        input_box = QtWidgets.QWidget()
        input_layout = QtWidgets.QHBoxLayout(input_box)
        input_layout.setContentsMargins(15, 10, 15, 10)

        self.input_field = QtWidgets.QLineEdit()
        self.input_field.setPlaceholderText("Задайте вопрос…")
        self.input_field.returnPressed.connect(self.send_question)

        self.send_btn = QtWidgets.QPushButton("Отправить")
        self.send_btn.clicked.connect(self.send_question)

        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(self.send_btn)
        layout.addWidget(input_box)

        self.statusBar().showMessage("Инициализация…")

        # Стили
        with open("src/ui/styles.css", "r", encoding="utf-8") as f:
            self.css = f.read()

    def _setup_tray(self):
        from PyQt5.QtWidgets import QApplication
        self.tray_icon = create_tray_icon(QApplication.instance())
        self.tray_icon.run_detached()

        def close_event(event):
            event.ignore()
            self.hide()
            self.tray_icon.show_message("RAG Assistant", "Работает в фоне", duration=2)
        self.closeEvent = close_event

    def _connect_signals(self):
        self.coordinator.indexing_started.connect(lambda: self.add_message("Индексация начата…"))
        self.coordinator.indexing_finished.connect(lambda: self.add_message("Готово к вопросам!"))
        self.coordinator.indexing_error.connect(lambda msg: self.add_message(f"Ошибка: {msg}"))

    def send_question(self):
        query = self.input_field.text().strip()
        if not query: return

        self.add_message(query, is_user=True)
        self.input_field.clear()
        self.show_typing()

        file_filter = None
        if " в " in query.lower():
            parts = query.lower().split(" в ")
            if len(parts) > 1:
                name = parts[1].strip().strip('"\'')
                path = os.path.join(self.folder_path, name)
                if os.path.exists(path):
                    file_filter = path

        self.coordinator.ask_async(query, file_filter, self.on_answer)

    def show_typing(self):
        self.typing_html = '<div class="message assistant-message typing visible">Генерация</div>'
        self.update_chat_html()
        self.start_typing_animation()

    def start_typing_animation(self):
        self.typing_dots = 0
        self.typing_timer = QtCore.QTimer()
        self.typing_timer.timeout.connect(self.update_typing_dots)
        self.typing_timer.start(400)

    def update_typing_dots(self):
        self.typing_dots = (self.typing_dots + 1) % 4
        self.typing_html = '<div class="message assistant-message typing visible">Генерация' + '.' * self.typing_dots + '</div>'
        self.update_chat_html()

    def stop_typing(self):
        if self.typing_timer:
            self.typing_timer.stop()
            self.typing_timer = None
        self.typing_html = ''

    def on_answer(self, response):
        self.stop_typing()

        result = response.get("result", "Нет ответа")
        sources = response.get("sources", "")
        self.add_message(result)

        if sources and sources != "Неизвестно":
            self.add_message(f"Источник: {sources}", is_source=True)

    def add_message(self, text, is_user=False, is_source=False):
        cls = "user-message" if is_user else "assistant-message"
        if is_source:
            cls = "source"
        message_html = f'<div class="message {cls} visible">{text}</div>'
        self.chat_html += message_html
        self.update_chat_html()

    def update_chat_html(self):
        full_html = f"""
        <html>
        <head><style>{self.css}</style></head>
        <body>
            {self.chat_html}
            {self.typing_html}
        </body>
        </html>
        """
        self.chat_browser.setHtml(full_html)
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        self.chat_browser.verticalScrollBar().setValue(self.chat_browser.verticalScrollBar().maximum())

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            src = url.toLocalFile()
            if os.path.isfile(src):
                dst = os.path.join(self.folder_path, os.path.basename(src))
                if os.path.exists(dst):
                    reply = QtWidgets.QMessageBox.question(
                        self, "Заменить?", f"Файл уже есть. Заменить?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                    if reply != QtWidgets.QMessageBox.Yes:
                        continue
                shutil.copy2(src, dst)
                self.add_message(f"Добавлен: <code>{os.path.basename(src)}</code>")
        self.coordinator.start_indexing()