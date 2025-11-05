# src/ui/main_window.py
import os
import shutil
from datetime import datetime
from PySide6 import QtWidgets, QtCore, QtGui
from coordinator import RAGCoordinator
from ui.tray import create_tray_icon
from ui.chat_widgets import ChatMessageWidget, add_message_item
from ui.chat_model import ChatListModel, ChatMessage
from ui.chat_delegate import ChatItemDelegate


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
        self._init_conversations()

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Sidebar (conversations/files placeholder)
        self.sidebar = QtWidgets.QListWidget()
        self.sidebar.setFixedWidth(240)
        self.sidebar.addItem("Новый чат")
        self.sidebar.addItem("Последний диалог")
        self.sidebar.setStyleSheet("border: none;")
        self.sidebar.itemClicked.connect(self.on_sidebar_click)

        # Main area (header + chat + composer)
        main_container = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        header_title = QtWidgets.QLabel("RAG Assistant • Chat")
        header_title.setObjectName("headerTitle")
        header_spacer = QtWidgets.QWidget()
        header_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.reindex_btn = QtWidgets.QPushButton("Переиндексировать")
        self.reindex_btn.clicked.connect(self.reindex)
        header_layout.addWidget(header_title)
        header_layout.addWidget(header_spacer)
        header_layout.addWidget(self.reindex_btn)

        # Chat view (QListView + model/delegate)
        self.chat_list = QtWidgets.QListView()
        self.chat_list.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.chat_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.chat_list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.chat_list.setSpacing(6)
        self.chat_list.setResizeMode(QtWidgets.QListView.Adjust)
        self.chat_list.setUniformItemSizes(False)
        self.chat_model = ChatListModel()
        self.chat_list.setModel(self.chat_model)
        self.chat_list.setItemDelegate(ChatItemDelegate(self.chat_list))

        # Composer (multiline like ChatGPT)
        composer = QtWidgets.QWidget()
        composer_layout = QtWidgets.QHBoxLayout(composer)
        composer_layout.setContentsMargins(16, 12, 16, 16)
        composer_layout.setSpacing(10)

        self.input_field = QtWidgets.QTextEdit()
        self.input_field.setPlaceholderText("Задайте вопрос…")
        self.input_field.setFixedHeight(64)
        self.input_field.installEventFilter(self)

        self.send_btn = QtWidgets.QPushButton("Отправить")
        self.send_btn.clicked.connect(self.send_question)

        composer_layout.addWidget(self.input_field, 1)
        composer_layout.addWidget(self.send_btn)

        # Assemble main
        main_layout.addWidget(header)
        main_layout.addWidget(self.chat_list, 1)
        main_layout.addWidget(composer)

        # Assemble root
        root_layout.addWidget(self.sidebar)
        root_layout.addWidget(main_container, 1)

        # Styles
        with open("src/ui/styles.css", "r", encoding="utf-8") as f:
            self.css = f.read()
        # Qt widget styles (QSS)
        qss = (
            "QPushButton {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #34b0ff, stop:1 #1a8cd8);"
            "  color: white; border: 1px solid #106ea9; border-radius: 14px;"
            "  padding: 10px 18px; font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4ec0ff, stop:1 #2298e6);"
            "}"
            "QPushButton:pressed { background: #1579c2; }"
            "QListWidget { background: #0e0e0e; color: #cfcfcf; border-right: 1px solid #1e1e1e; }"
            "QListWidget::item { padding: 10px 12px; }"
            "QListWidget::item:selected { background: #1b1b1b; }"
            "QTextEdit { background: #121212; color: #EAEAEA; border: 1px solid #1e1e1e; border-radius: 14px; padding: 10px 12px; }"
            "QTextEdit:focus { border: 1px solid #2b6cb0; }"
            "#headerTitle { font-weight: 700; letter-spacing: 0.2px; }"
            "QScrollBar:vertical { background: #0d0d0d; width: 10px; margin: 0px; }"
            "QScrollBar::handle:vertical { background: #2a2a2a; border-radius: 5px; min-height: 20px; }"
            "QScrollBar::handle:vertical:hover { background: #3a3a3a; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
            "#avatar { background: #262626; color: #9e9e9e; border-radius: 14px; }"
            "#bubbleMeta { color: #9aa; font-size: 11px; }"
            "#bubbleAssistant { background: #151515; border: 1px solid #232323; border-radius: 16px; color: #e6e6e6; }"
            "#bubbleUser { background: #1DA1F2; border: 1px solid rgba(13,110,253,0.45); border-radius: 16px; color: white; }"
        )
        self.setStyleSheet(qss)

    def _setup_tray(self):
        from PySide6.QtWidgets import QApplication
        self.tray_icon = create_tray_icon(QApplication.instance())
        self.tray_icon.run_detached()
        # Ensure tray stops when app quits
        QApplication.instance().aboutToQuit.connect(lambda: self.tray_icon.stop())

    def _connect_signals(self):
        self.coordinator.indexing_started.connect(lambda: self.add_message("Индексация начата…"))
        self.coordinator.indexing_finished.connect(lambda: self.add_message("Готово к вопросам!"))
        self.coordinator.indexing_error.connect(lambda msg: self.add_message(f"Ошибка: {msg}"))

    def send_question(self):
        query = self.input_field.toPlainText().strip()
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
        self._add_typing_item()
        self.start_typing_animation()

    def start_typing_animation(self):
        self.typing_dots = 0
        self.typing_timer = QtCore.QTimer()
        self.typing_timer.timeout.connect(self.update_typing_dots)
        self.typing_timer.start(400)

    def update_typing_dots(self):
        self.typing_dots = (self.typing_dots + 1) % 4
        self.chat_model.setTypingDots(self.typing_dots)

    def stop_typing(self):
        if self.typing_timer:
            self.typing_timer.stop()
            self.typing_timer = None
        self.typing_html = ''
        self.chat_model.removeTyping()

    def on_answer(self, response):
        self.stop_typing()

        result = response.get("result", "Нет ответа")
        sources = response.get("sources", "")
        self.add_message(result)

        if sources and sources != "Неизвестно":
            self.add_message(f"Источник: {sources}", is_source=True)

    def add_message(self, text, is_user=False, is_source=False):
        ts = datetime.now().strftime("%H:%M")
        content = text
        if is_source:
            content = f"Источник: {text}"
        self.chat_model.appendMessage(ChatMessage(text=self._html_escape(content), isUser=is_user, timestamp=ts))
        if hasattr(self, 'conversations'):
            self.conversations[self.current_chat_idx].append((content, is_user, ts))
        self._scroll_to_bottom()
        self._adjust_bubble_widths()

    def update_chat_html(self):
        # Not used with QListWidget-based chat
        pass

    def _scroll_to_bottom(self):
        self.chat_list.scrollToBottom()

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

    def closeEvent(self, event):
        try:
            if self.tray_icon:
                self.tray_icon.stop()
        except Exception:
            pass
        event.accept()

    def reindex(self):
        self.add_message("Переиндексация начата…")
        self.coordinator.start_indexing()

    def eventFilter(self, obj, event):
        if obj is self.input_field and event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and not (event.modifiers() & QtCore.Qt.ShiftModifier):
                self.send_question()
                return True
        return super().eventFilter(obj, event)

    def _html_escape(self, text):
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;")
                .replace("\n", "<br>")
        )

    def _add_typing_item(self):
        self.chat_model.appendMessage(ChatMessage(text="Генерация", isUser=False, timestamp="", kind="typing"))
        self._scroll_to_bottom()
        self._adjust_bubble_widths()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._adjust_bubble_widths()

    def _adjust_bubble_widths(self):
        # For QListView + delegate, trigger size recalculation by notifying model
        if hasattr(self, 'chat_model') and self.chat_model is not None:
            rows = self.chat_model.rowCount()
            if rows > 0:
                top_left = self.chat_model.index(0, 0)
                bottom_right = self.chat_model.index(rows - 1, 0)
                self.chat_model.dataChanged.emit(top_left, bottom_right, [])
                # Also request a layout change for safety on viewport resize
                self.chat_model.layoutChanged.emit()

    def _init_conversations(self):
        self.conversations = [[]]
        self.current_chat_idx = 0
        self.sidebar.addItem("Диалог 1")

    def on_sidebar_click(self, item):
        text = item.text()
        if text == "Новый чат":
            self.create_new_chat()
            return
        if text == "Последний диалог":
            if len(self.conversations) > 0:
                self.switch_chat(len(self.conversations) - 1)
            return
        if text.startswith("Диалог "):
            try:
                idx = int(text.split(" ")[-1]) - 1
                if 0 <= idx < len(self.conversations):
                    self.switch_chat(idx)
            except Exception:
                pass

    def create_new_chat(self):
        self.conversations.append([])
        self.current_chat_idx = len(self.conversations) - 1
        # Reset view by installing a fresh empty model
        self.chat_model = ChatListModel()
        self.chat_list.setModel(self.chat_model)
        self.typing_html = ""
        self.sidebar.addItem(f"Диалог {self.current_chat_idx + 1}")

    def switch_chat(self, idx):
        self.current_chat_idx = idx
        self.chat_model = ChatListModel()
        self.chat_list.setModel(self.chat_model)
        for content, is_user, ts in self.conversations[idx]:
            self.chat_model.appendMessage(ChatMessage(text=self._html_escape(content), isUser=is_user, timestamp=ts))
        self.typing_html = ""
        self._scroll_to_bottom()
        self._adjust_bubble_widths()