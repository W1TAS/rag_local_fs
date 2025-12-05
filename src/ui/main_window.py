# src/ui/main_window.py
import os
import shutil
from datetime import datetime
from PySide6 import QtWidgets, QtCore, QtGui
from coordinator import RAGCoordinator
from cache import get_folder_cache_dir, clear_folder_cache
from ui.tray import create_tray_icon
from ui.log_window import LogWindow
from ui.chat_widgets import ChatMessageWidget, add_message_item
from ui.chat_model import ChatListModel, ChatMessage
from ui.chat_delegate import ChatItemDelegate


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, folder_path=None):
        super().__init__()
        self.folder_path = folder_path
        self.coordinator = None
        self.chat_html = ""
        self.typing_html = ""
        self.typing_timer = None
        self.typing_states = {}
        self.pending_typing_chat_idx = None

        self.setWindowTitle("RAG Assistant")
        self.setGeometry(300, 200, 960, 720)
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setAcceptDrops(True)

        self._setup_ui()
        self._setup_tray()
        if self.folder_path:
            if RAGCoordinator._instance is not None:
                try:
                    RAGCoordinator._instance.close()
                except Exception:
                    pass
                RAGCoordinator._instance = None
            self.coordinator = RAGCoordinator.instance(self.folder_path)
            self._connect_signals()
            try:
                cache_dir = get_folder_cache_dir(self.folder_path)
                # show cache path to user in tooltip for easy discovery
                self.folder_label.setToolTip(f"Кеш: {cache_dir}")
            except Exception:
                pass
        self._init_conversations()

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Sidebar container: list + footer buttons
        self.sidebar = QtWidgets.QListWidget()
        self.sidebar.setFixedWidth(240)
        self.sidebar.addItem("Новый чат")
        self.sidebar.setStyleSheet("border: none;")
        self.sidebar.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.sidebar.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.sidebar.customContextMenuRequested.connect(self._on_sidebar_context_menu)
        self.sidebar.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sidebar.itemClicked.connect(self.on_sidebar_click)

        self.sidebar_container = QtWidgets.QWidget()
        self.sidebar_container.setFixedWidth(240)
        # ensure sidebar container background matches the sidebar
        self.sidebar_container.setStyleSheet("background: #0e0e0e;")
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(6)
        sidebar_layout.addWidget(self.sidebar)

        # Footer with utility buttons
        footer = QtWidgets.QWidget()
        footer.setStyleSheet("background: #0e0e0e;")  # Match sidebar background
        footer_layout = QtWidgets.QHBoxLayout(footer)
        # reduce top margin to avoid visible gap between list and footer
        footer_layout.setContentsMargins(6, 4, 6, 8)
        footer_layout.setSpacing(8)
        # Logs button (left) and Clear Cache (right)
        self.logs_btn = QtWidgets.QPushButton("Логи")
        self.logs_btn.setMinimumHeight(34)
        self.logs_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.logs_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #cfcfcf; border: none; padding: 6px 10px; }"
            "QPushButton:hover { color: white; }"
        )
        self.logs_btn.clicked.connect(self.open_logs)
        footer_layout.addWidget(self.logs_btn)

        footer_layout.addStretch()

        self.clear_cache_btn = QtWidgets.QPushButton("Очистить кеш")
        self.clear_cache_btn.setMinimumHeight(40)
        self.clear_cache_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # reapply gradient style explicitly so button keeps the primary look
        self.clear_cache_btn.setStyleSheet(
            "QPushButton {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #34b0ff, stop:1 #1a8cd8);"
            "  color: white; border: 1px solid #106ea9; border-radius: 14px;"
            "  padding: 10px 18px; font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4ec0ff, stop:1 #2298e6);"
            "}"
            "QPushButton:pressed { background: #1579c2; }"
        )
        self.clear_cache_btn.clicked.connect(self.on_clear_cache)
        footer_layout.addWidget(self.clear_cache_btn)

        # Disable until folder selected
        if not self.folder_path:
            self.clear_cache_btn.setEnabled(False)

        sidebar_layout.addWidget(footer)

        main_container = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        header_title = QtWidgets.QLabel("RAG Assistant • Chat")
        header_title.setObjectName("headerTitle")
        header_spacer = QtWidgets.QWidget()
        header_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.folder_label = QtWidgets.QLabel(self.folder_path or "Папка не выбрана")
        self.folder_label.setObjectName("folderLabel")
        self.choose_folder_btn = QtWidgets.QPushButton("Выбрать папку")
        self.choose_folder_btn.clicked.connect(self.select_folder_dialog)

        self.reindex_btn = QtWidgets.QPushButton("Переиндексировать")
        self.reindex_btn.clicked.connect(self.reindex)

        header_layout.addWidget(header_title)
        header_layout.addWidget(header_spacer)
        header_layout.addWidget(self.folder_label)
        header_layout.addWidget(self.choose_folder_btn)
        header_layout.addWidget(self.reindex_btn)

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

        if not self.folder_path:
            self.send_btn.setEnabled(False)
            self.input_field.setEnabled(False)
            self.reindex_btn.setEnabled(False)

        composer_layout.addWidget(self.input_field, 1)
        composer_layout.addWidget(self.send_btn)

        main_layout.addWidget(header)
        main_layout.addWidget(self.chat_list, 1)
        main_layout.addWidget(composer)

        root_layout.addWidget(self.sidebar_container)
        root_layout.addWidget(main_container, 1)

        # Load stylesheet relative to this file so app works regardless of CWD
        ui_dir = os.path.dirname(__file__)
        stylesheet_path = os.path.join(ui_dir, "styles.css")
        try:
            with open(stylesheet_path, "r", encoding="utf-8") as f:
                self.css = f.read()
        except Exception:
            # Fallback: empty stylesheet (don't crash if file not found)
            self.css = ""
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
            "QListWidget::item { padding: 10px 12px; outline: none; }"
            "QListWidget::item:focus { outline: none; }"
            "QListWidget::item:selected { background: #1b1b1b; color: #cfcfcf; }"
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
        QApplication.instance().aboutToQuit.connect(lambda: self.tray_icon.stop())

    def _connect_signals(self):
        self.coordinator.indexing_started.connect(lambda: self.add_message("Индексация начата…"))
        self.coordinator.indexing_finished.connect(lambda: self.add_message("Готово к вопросам!"))
        self.coordinator.indexing_error.connect(lambda msg: self.add_message(f"Ошибка: {msg}"))

    def select_folder_dialog(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку для индексации", os.getcwd())
        if not folder:
            return
        folder = os.path.abspath(folder)
        self.folder_path = folder
        self.folder_label.setText(self.folder_path)

        if RAGCoordinator._instance is not None:
            try:
                RAGCoordinator._instance.close()
            except Exception:
                pass
            RAGCoordinator._instance = None

        self.coordinator = RAGCoordinator.instance(self.folder_path)
        self._connect_signals()

        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.reindex_btn.setEnabled(True)
        self.clear_cache_btn.setEnabled(True)
        # Logs button always enabled (app-level)
        try:
            self.logs_btn.setEnabled(True)
        except Exception:
            pass

    def _ensure_coordinator(self):
        if self.coordinator is None:
            QtWidgets.QMessageBox.information(self, "Папка не выбрана", "Сначала выберите папку для индексации.")
            self.select_folder_dialog()
        return self.coordinator is not None


    def on_clear_cache(self):
        if not self.folder_path:
            QtWidgets.QMessageBox.information(self, "Кеш не выбран", "Сначала выберите папку.")
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Удалить кеш?",
            f"Удалить кеш для папки:\n{self.folder_path}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        success = False
        try:
            success = clear_folder_cache(self.folder_path)
        except Exception:
            success = False

        if success:
            QtWidgets.QMessageBox.information(self, "Кеш удалён", "Кеш успешно удалён.")
            # Offer reindex
            if QtWidgets.QMessageBox.question(self, "Переиндексировать?", "Переиндексировать сейчас эту папку?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes:
                if self.coordinator is None:
                    try:
                        if RAGCoordinator._instance is not None:
                            try:
                                RAGCoordinator._instance.close()
                            except Exception:
                                pass
                            RAGCoordinator._instance = None
                        self.coordinator = RAGCoordinator.instance(self.folder_path)
                        self._connect_signals()
                    except Exception:
                        pass
                if self.coordinator:
                    self.add_message("Переиндексация начата…")
                    self.coordinator.start_indexing()
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Не удалось удалить кеш. Проверьте права и повторите.")

    def open_logs(self):
        try:
            from cache import get_cache_root
            log_path = os.path.join(get_cache_root(), "app.log")
            win = LogWindow(self, log_path=log_path)
            win.exec()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось открыть логи: {e}")

    def _on_sidebar_context_menu(self, pos):
        item = self.sidebar.itemAt(pos)
        if not item:
            return
        if item.text() == "Новый чат":
            return
        menu = QtWidgets.QMenu()
        delete_act = menu.addAction("Удалить чат")
        action = menu.exec_(self.sidebar.mapToGlobal(pos))
        if action == delete_act:
            self._delete_chat(item)

    def _delete_chat(self, item: QtWidgets.QListWidgetItem):
        row = self.sidebar.row(item)
        conv_idx = row - 1
        if not (0 <= conv_idx < len(self.conversations)):
            return
        self.conversations.pop(conv_idx)
        if conv_idx in self.typing_states:
            self.typing_states.pop(conv_idx)
        self.sidebar.takeItem(row)
        for i in range(1, self.sidebar.count()):
            self.sidebar.item(i).setText(f"Диалог {i}")
        if self.current_chat_idx >= len(self.conversations):
            self.current_chat_idx = max(0, len(self.conversations) - 1)
        self.switch_chat(self.current_chat_idx)

    def send_question(self):
        query = self.input_field.toPlainText().strip()
        if not query: return

        if not self._ensure_coordinator():
            return

        self.add_message(query, is_user=True)
        self.input_field.clear()
        self.pending_typing_chat_idx = self.current_chat_idx
        self._add_typing_item_for_chat(self.pending_typing_chat_idx)
        self.start_typing_animation()

        file_filter = None
        if " в " in query.lower():
            parts = query.lower().split(" в ")
            if len(parts) > 1:
                name = parts[1].strip().strip('"\'')
                path = os.path.join(self.folder_path, name)
                if os.path.exists(path):
                    file_filter = path

        self.coordinator.ask_async(query, file_filter, lambda resp, idx=self.pending_typing_chat_idx: self.on_answer(resp, idx))

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
        if self.pending_typing_chat_idx is not None:
            self.typing_states[self.pending_typing_chat_idx] = self.typing_dots
            if self.pending_typing_chat_idx == self.current_chat_idx:
                self.chat_model.setTypingDots(self.typing_dots)

    def stop_typing(self, chat_idx=None):
        if self.typing_timer:
            self.typing_timer.stop()
            self.typing_timer = None
        if chat_idx is None:
            chat_idx = self.pending_typing_chat_idx
        if chat_idx in self.typing_states:
            self.typing_states.pop(chat_idx)
        if chat_idx == self.current_chat_idx:
            self.chat_model.removeTyping()
        if self.pending_typing_chat_idx == chat_idx:
            self.pending_typing_chat_idx = None

    def on_answer(self, response, chat_idx=None):
        if chat_idx is None:
            chat_idx = self.current_chat_idx
        self.stop_typing(chat_idx)

        result = response.get("result", "Нет ответа")
        sources = response.get("sources", "")
        self.add_message(result, chat_idx=chat_idx)

        if sources and sources != "Неизвестно":
            self.add_message(f"Источник: {sources}", is_source=True, chat_idx=chat_idx)

    def add_message(self, text, is_user=False, is_source=False, chat_idx=None):
        if chat_idx is None:
            chat_idx = self.current_chat_idx
        ts = datetime.now().strftime("%H:%M")
        content = text
        if is_source:
            content = f"{text}"

        if hasattr(self, 'conversations'):
            self.conversations[chat_idx].append((content, is_user, ts))

        if chat_idx == self.current_chat_idx:
            self.chat_model.appendMessage(ChatMessage(text=self._html_escape(content), isUser=is_user, timestamp=ts))
            self._scroll_to_bottom()
            self._adjust_bubble_widths()

    def update_chat_html(self):
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
        if self.coordinator:
            self.coordinator.start_indexing()
        else:
            self.select_folder_dialog()

    def closeEvent(self, event):
        try:
            if self.tray_icon:
                self.tray_icon.stop()
        except Exception:
            pass
        event.accept()

    def reindex(self):
        if not self._ensure_coordinator():
            return
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
        self._add_typing_item_for_chat(self.current_chat_idx)

    def _add_typing_item_for_chat(self, chat_idx: int):
        if chat_idx not in self.typing_states:
            self.typing_states[chat_idx] = 0
        if chat_idx == self.current_chat_idx:
            if not (self.chat_model._messages and self.chat_model._messages[-1].kind == "typing"):
                self.chat_model.appendMessage(ChatMessage(text="Генерация", isUser=False, timestamp="", kind="typing"))
                self._scroll_to_bottom()
                self._adjust_bubble_widths()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._adjust_bubble_widths()

    def _adjust_bubble_widths(self):
        if hasattr(self, 'chat_model') and self.chat_model is not None:
            rows = self.chat_model.rowCount()
            if rows > 0:
                top_left = self.chat_model.index(0, 0)
                bottom_right = self.chat_model.index(rows - 1, 0)
                self.chat_model.dataChanged.emit(top_left, bottom_right, [])
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
        self.chat_model = ChatListModel()
        self.chat_list.setModel(self.chat_model)
        self.typing_html = ""
        self.sidebar.addItem(f"Диалог {self.current_chat_idx + 1}")
        self.sidebar.setCurrentRow(self.current_chat_idx + 1)

    def switch_chat(self, idx):
        self.current_chat_idx = idx
        self.chat_model = ChatListModel()
        self.chat_list.setModel(self.chat_model)
        for content, is_user, ts in self.conversations[idx]:
            self.chat_model.appendMessage(ChatMessage(text=self._html_escape(content), isUser=is_user, timestamp=ts))
        if idx in self.typing_states:
            dots = self.typing_states.get(idx, 0) or 0
            text = "Генерация" + "." * dots
            self.chat_model.appendMessage(ChatMessage(text=text, isUser=False, timestamp="", kind="typing"))
        self.typing_html = ""
        try:
            self.sidebar.setCurrentRow(idx + 1)
        except Exception:
            pass
        self._scroll_to_bottom()
        self._adjust_bubble_widths()