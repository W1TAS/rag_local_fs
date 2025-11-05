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
    def __init__(self, folder_path=None):
        super().__init__()
        self.folder_path = folder_path
        # Coordinator is created only after a folder is selected; if folder_path
        # was passed on startup, we'll create it below. Otherwise keep None.
        self.coordinator = None
        self.chat_html = ""
        self.typing_html = ""  # ← ФИКС: Инициализация
        self.typing_timer = None
        # Track typing state per chat index: maps chat_idx -> dots (int)
        self.typing_states = {}
        # Which chat index currently has an active pending generation (or None)
        self.pending_typing_chat_idx = None

        self.setWindowTitle("RAG Assistant")
        self.setGeometry(300, 200, 960, 720)
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setAcceptDrops(True)

        self._setup_ui()
        self._setup_tray()
        # Connect signals only if coordinator already exists
        if self.folder_path:
            # ensure any existing coordinator is created with the passed folder
            if RAGCoordinator._instance is not None:
                try:
                    RAGCoordinator._instance.close()
                except Exception:
                    pass
                RAGCoordinator._instance = None
            self.coordinator = RAGCoordinator.instance(self.folder_path)
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
        # Sidebar items: keep 'Новый чат' and dynamic conversation items
        self.sidebar.addItem("Новый чат")
        # Removed the static "Последний диалог" item (not needed)
        self.sidebar.setStyleSheet("border: none;")
        # show selection and allow single selection
        self.sidebar.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        # context menu for deleting chats
        self.sidebar.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.sidebar.customContextMenuRequested.connect(self._on_sidebar_context_menu)
        # Avoid drawing the focus rectangle when clicking items
        self.sidebar.setFocusPolicy(QtCore.Qt.NoFocus)
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
        # Folder selector + info
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

        # If no folder configured, disable composer and reindex until folder is chosen
        if not self.folder_path:
            self.send_btn.setEnabled(False)
            self.input_field.setEnabled(False)
            self.reindex_btn.setEnabled(False)

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
        # Ensure tray stops when app quits
        QApplication.instance().aboutToQuit.connect(lambda: self.tray_icon.stop())

    def _connect_signals(self):
        self.coordinator.indexing_started.connect(lambda: self.add_message("Индексация начата…"))
        self.coordinator.indexing_finished.connect(lambda: self.add_message("Готово к вопросам!"))
        self.coordinator.indexing_error.connect(lambda msg: self.add_message(f"Ошибка: {msg}"))

    def select_folder_dialog(self):
        # Ask user to choose a folder to index
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку для индексации", os.getcwd())
        if not folder:
            return
        folder = os.path.abspath(folder)
        self.folder_path = folder
        self.folder_label.setText(self.folder_path)

        # If a coordinator already exists, close and reset it so we can start fresh
        if RAGCoordinator._instance is not None:
            try:
                RAGCoordinator._instance.close()
            except Exception:
                pass
            RAGCoordinator._instance = None

        # create coordinator for the selected folder and connect signals
        self.coordinator = RAGCoordinator.instance(self.folder_path)
        self._connect_signals()

        # enable UI controls now that folder is chosen
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.reindex_btn.setEnabled(True)

    def _ensure_coordinator(self):
        # Helper: return True if coordinator exists, otherwise prompt user
        if self.coordinator is None:
            QtWidgets.QMessageBox.information(self, "Папка не выбрана", "Сначала выберите папку для индексации.")
            self.select_folder_dialog()
        return self.coordinator is not None

    def _on_sidebar_context_menu(self, pos):
        item = self.sidebar.itemAt(pos)
        if not item:
            return
        # don't allow deleting the 'Новый чат' item
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
        # remove conversation
        self.conversations.pop(conv_idx)
        # remove typing state if present
        if conv_idx in self.typing_states:
            self.typing_states.pop(conv_idx)
        # remove item from sidebar
        self.sidebar.takeItem(row)
        # renumber remaining dialog items' labels
        for i in range(1, self.sidebar.count()):
            self.sidebar.item(i).setText(f"Диалог {i}")
        # adjust current_chat_idx
        if self.current_chat_idx >= len(self.conversations):
            self.current_chat_idx = max(0, len(self.conversations) - 1)
        # switch to the current index to refresh UI
        self.switch_chat(self.current_chat_idx)

    def send_question(self):
        query = self.input_field.toPlainText().strip()
        if not query: return

        if not self._ensure_coordinator():
            return

        # add user message to the current conversation
        self.add_message(query, is_user=True)
        self.input_field.clear()
        # mark that this chat has a pending generation
        self.pending_typing_chat_idx = self.current_chat_idx
        # add typing indicator for this chat (either visible now or persisted)
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

        # pass the chat index along so the callback can deliver to correct chat
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
        # update stored dots for the pending chat
        if self.pending_typing_chat_idx is not None:
            self.typing_states[self.pending_typing_chat_idx] = self.typing_dots
            # if the pending chat is currently visible, update its model
            if self.pending_typing_chat_idx == self.current_chat_idx:
                self.chat_model.setTypingDots(self.typing_dots)

    def stop_typing(self, chat_idx=None):
        # Stop the typing animation for a specific chat (default: pending)
        if self.typing_timer:
            self.typing_timer.stop()
            self.typing_timer = None
        if chat_idx is None:
            chat_idx = self.pending_typing_chat_idx
        # remove stored typing state
        if chat_idx in self.typing_states:
            self.typing_states.pop(chat_idx)
        # if the chat is currently visible, remove typing message from view
        if chat_idx == self.current_chat_idx:
            self.chat_model.removeTyping()
        # clear pending pointer if it matches
        if self.pending_typing_chat_idx == chat_idx:
            self.pending_typing_chat_idx = None

    def on_answer(self, response, chat_idx=None):
        # chat_idx is the conversation index the answer belongs to
        if chat_idx is None:
            chat_idx = self.current_chat_idx
        # stop typing for that chat
        self.stop_typing(chat_idx)

        result = response.get("result", "Нет ответа")
        sources = response.get("sources", "")
        # deliver message to the target conversation
        self.add_message(result, chat_idx=chat_idx)

        if sources and sources != "Неизвестно":
            self.add_message(f"Источник: {sources}", is_source=True, chat_idx=chat_idx)

    def add_message(self, text, is_user=False, is_source=False, chat_idx=None):
        # If chat_idx omitted, append to current chat
        if chat_idx is None:
            chat_idx = self.current_chat_idx
        ts = datetime.now().strftime("%H:%M")
        content = text
        if is_source:
            content = f"Источник: {text}"

        # Append to conversation storage
        if hasattr(self, 'conversations'):
            # store raw content (not escaped) for persistence
            self.conversations[chat_idx].append((content, is_user, ts))

        # If the message belongs to the currently visible chat, update the model
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
        # start indexing if possible, otherwise prompt to select folder
        if self.coordinator:
            self.coordinator.start_indexing()
        else:
            # no coordinator yet: ask user to choose folder to index
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
        # delegate to per-chat helper
        self._add_typing_item_for_chat(self.current_chat_idx)

    def _add_typing_item_for_chat(self, chat_idx: int):
        # ensure typing state exists
        if chat_idx not in self.typing_states:
            self.typing_states[chat_idx] = 0
        # if the chat is currently visible, add a typing message to the model (if not already present)
        if chat_idx == self.current_chat_idx:
            if not (self.chat_model._messages and self.chat_model._messages[-1].kind == "typing"):
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
        # select the newly created chat
        self.sidebar.setCurrentRow(self.current_chat_idx + 1)

    def switch_chat(self, idx):
        self.current_chat_idx = idx
        self.chat_model = ChatListModel()
        self.chat_list.setModel(self.chat_model)
        for content, is_user, ts in self.conversations[idx]:
            self.chat_model.appendMessage(ChatMessage(text=self._html_escape(content), isUser=is_user, timestamp=ts))
        # if this conversation currently has a typing state, show it
        if idx in self.typing_states:
            dots = self.typing_states.get(idx, 0) or 0
            text = "Генерация" + "." * dots
            self.chat_model.appendMessage(ChatMessage(text=text, isUser=False, timestamp="", kind="typing"))
        self.typing_html = ""
        # visually select the sidebar item for this conversation
        try:
            self.sidebar.setCurrentRow(idx + 1)
        except Exception:
            pass
        self._scroll_to_bottom()
        self._adjust_bubble_widths()