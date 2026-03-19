# src/ui/main_window.py
import os
from datetime import datetime
from typing import Optional

from PyQt6 import QtWidgets, QtCore, QtGui

from cache import clear_folder_cache, get_folder_cache_dir
from config import MODEL_NAME, SUPPORTED_FORMATS
from coordinator import RAGCoordinator
from ui.chat_delegate import ChatItemDelegate
from ui.chat_model import ChatListModel, ChatMessage
from ui.log_window import LogWindow
from ui.tray import create_tray_icon


class MainWindow(QtWidgets.QMainWindow):
    _DARK_QSS = """
    * { font-family: "Segoe UI", Arial, sans-serif; font-size: 13px; }

    QMainWindow, QWidget { background: #1e1e1e; color: #d4d4d4; }
    QLabel { color: #d4d4d4; border: none; margin: 0px; padding: 0px; }

    QTreeView {
        background: #252526;
        color: #d4d4d4;
        border: 1px solid #404040;
        border-radius: 6px;
        outline: none;
        gridline-color: transparent;
    }
    QTreeView::item { padding: 4px 8px; }
    QTreeView::item:hover { background: #2a2d2e; }
    QTreeView::item:selected { background: #4a4a4a; color: #ffffff; }
    QTreeView::branch:has-siblings { border-image: none; }
    QTreeView::branch:open:has-children { border-image: none; }
    QTreeView::branch:closed:has-children { border-image: none; }

    QSplitter::handle { background: #404040; width: 6px; }
    QSplitter::handle:horizontal { height: 6px; }
    QSplitter::handle:hover { background: #666666; }

    QTextBrowser, QTextEdit {
        background: #1e1e1e;
        color: #d4d4d4;
        border: 1px solid #404040;
        border-radius: 6px;
        padding: 10px 12px;
    }

    QLineEdit {
        background: #121212;
        color: #d4d4d4;
        border: 1px solid #404040;
        border-radius: 6px;
        padding: 8px 10px;
    }
    QLineEdit:focus { border: 1px solid #666666; }

    QPushButton {
        background: transparent;
        color: #d4d4d4;
        border: 1px solid #404040;
        border-radius: 8px;
        padding: 8px 12px;
        font-weight: 600;
    }
    QPushButton:hover { background: #2a2d2e; border-color: #666666; color: #ffffff; }
    QPushButton:pressed { background: #3a3d3e; border-color: #666666; }
    QPushButton:disabled {
        color: #6b6b6b;
        background: #252526;
        border-color: #404040;
    }
    QPushButton#accentButton { background: #4a4a4a; border-color: #666666; }
    QPushButton#accentButton:hover { background: #5a5a5a; border-color: #888888; }

    QProgressBar {
        border: 1px solid #404040;
        background: #252526;
        border-radius: 6px;
        height: 10px;
    }
    QProgressBar::chunk { background-color: #666666; border-radius: 6px; }

    QStatusBar { background: #252526; border-top: 1px solid #404040; padding: 4px 0px; }
    QStatusBar::item { border: none !important; padding: 0px 0px; margin: 0px; }

    QMenuBar { background: #252526; border-bottom: 1px solid #404040; }
    QMenuBar::item { background: transparent; padding: 6px 10px; color: #d4d4d4; }
    QMenuBar::item:selected { background: #2a2d2e; }

    QMenu { background: #252526; color: #d4d4d4; border: 1px solid #404040; }
    QMenu::item { padding: 6px 16px; }
    QMenu::item:selected { background: #4a4a4a; color: #ffffff; }

    QListWidget {
        background: #252526;
        color: #d4d4d4;
        border: 1px solid #404040;
        border-radius: 6px;
    }
    QListWidget::item { padding: 8px 10px; }
    QListWidget::item:hover { background: #2a2d2e; }
    QListWidget::item:selected { background: #4a4a4a; color: #ffffff; }

    QScrollBar:vertical { background: #0d0d0d; width: 8px; }
    QScrollBar::handle:vertical { background: #2a2a2a; border-radius: 4px; min-height: 20px; }
    QScrollBar::handle:vertical:hover { background: #3a3a3a; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }

    QScrollBar:horizontal { background: #0d0d0d; height: 8px; }
    QScrollBar::handle:horizontal { background: #2a2a2a; border-radius: 4px; min-width: 20px; }
    QScrollBar::handle:horizontal:hover { background: #3a3a3a; }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
    """

    def __init__(self, folder_path: Optional[str] = None):
        super().__init__()

        self.folder_path: Optional[str] = folder_path
        self.coordinator: Optional[RAGCoordinator] = None

        self.conversations = []
        self.current_chat_idx = 0

        self.typing_timer: Optional[QtCore.QTimer] = None
        self.typing_states = {}  # chat_idx -> dots
        self.pending_typing_chat_idx: Optional[int] = None
        self.pending_typing_dots = 0

        self.setWindowTitle("RAG Assistant")
        self.setGeometry(300, 200, 1120, 760)
        self._set_window_icon()

        self.setup_ui()
        self._setup_tray()

        if self.folder_path:
            self._start_for_folder(os.path.abspath(self.folder_path), connect_signals=True)

        self._init_conversations()

    def _set_window_icon(self):
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            icons_dir_candidates = [
                os.path.join(repo_root, "assets", "icons"),
                os.path.join(repo_root, "..", "assets", "icons"),
            ]
            icons_dir = next((p for p in icons_dir_candidates if os.path.isdir(p)), icons_dir_candidates[0])
            for name in ("app_icon.ico", "app_icon.png", "icon.png"):
                p = os.path.join(icons_dir, name)
                if os.path.exists(p):
                    self.setWindowIcon(QtGui.QIcon(p))
                    break
        except Exception:
            pass

    @classmethod
    def apply_dark_theme(cls, app: QtWidgets.QApplication) -> None:
        app.setStyleSheet(cls._DARK_QSS)

    def setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Left panel (~280px): chats + explorer tree
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(280)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        left_layout.addWidget(QtWidgets.QLabel("Chats"))
        self.chat_threads_list = QtWidgets.QListWidget()
        self.chat_threads_list.setFixedHeight(140)
        self.chat_threads_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_threads_list.customContextMenuRequested.connect(self._on_chat_threads_context_menu)
        self.chat_threads_list.itemClicked.connect(self.on_chat_thread_clicked)
        left_layout.addWidget(self.chat_threads_list, 0)

        left_layout.addWidget(QtWidgets.QLabel("Files"))
        self.file_tree = QtWidgets.QTreeView()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.file_tree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.file_tree.clicked.connect(self.on_file_clicked)
        self.fs_model: Optional[QtWidgets.QFileSystemModel] = None
        left_layout.addWidget(self.file_tree, 1)

        # Right: header + vertical splitter
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        header_layout.setSpacing(12)

        header_title = QtWidgets.QLabel("RAG Assistant")
        header_title.setStyleSheet("font-weight: 700; letter-spacing: 0.2px;")
        header_spacer = QtWidgets.QWidget()
        header_spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)

        self.folder_label = QtWidgets.QLabel(self.folder_path or "No folder selected")
        self.choose_folder_btn = QtWidgets.QPushButton("Open Folder")
        self.choose_folder_btn.clicked.connect(self.select_folder_dialog)

        self.reindex_btn = QtWidgets.QPushButton("Re-index")
        self.reindex_btn.clicked.connect(lambda: self.start_indexing(reason="Re-indexing"))

        header_layout.addWidget(header_title)
        header_layout.addWidget(header_spacer)
        header_layout.addWidget(self.folder_label)
        header_layout.addWidget(self.choose_folder_btn)
        header_layout.addWidget(self.reindex_btn)

        self.center_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Module 1: Document preview
        preview_panel = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        preview_layout.setSpacing(8)

        preview_toolbar = QtWidgets.QWidget()
        preview_toolbar_layout = QtWidgets.QHBoxLayout(preview_toolbar)
        preview_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        preview_toolbar_layout.setSpacing(8)

        self.preview_title = QtWidgets.QLabel("Document Preview")
        self.preview_title.setStyleSheet("color:#9aa; font-weight:600;")
        preview_toolbar_layout.addWidget(self.preview_title)
        preview_toolbar_layout.addStretch()

        self.preview_browser = QtWidgets.QTextBrowser()
        self.preview_browser.setReadOnly(True)
        self.preview_browser.setOpenExternalLinks(False)
        self.preview_browser.setStyleSheet("font-family: Consolas, 'JetBrains Mono', monospace; font-size: 12.5px;")
        self.preview_browser.setPlainText("Select a file in Files to view its contents.")

        preview_layout.addWidget(preview_toolbar)
        preview_layout.addWidget(self.preview_browser, 1)

        # Module 2: Chat
        chat_panel = QtWidgets.QWidget()
        chat_layout = QtWidgets.QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(10, 10, 10, 10)
        chat_layout.setSpacing(10)

        chat_top = QtWidgets.QWidget()
        chat_top_layout = QtWidgets.QHBoxLayout(chat_top)
        chat_top_layout.setContentsMargins(0, 0, 0, 0)
        chat_top_layout.setSpacing(10)

        chat_top_label = QtWidgets.QLabel("RAG Chat")
        chat_top_label.setStyleSheet("font-weight:700; color:#9aa;")
        chat_top_layout.addWidget(chat_top_label)
        chat_top_layout.addStretch()

        self.clear_chat_btn = QtWidgets.QPushButton("Clear Chat")
        self.clear_chat_btn.setObjectName("accentButton")
        self.clear_chat_btn.clicked.connect(self.clear_chat_history)
        chat_top_layout.addWidget(self.clear_chat_btn)

        self.chat_list = QtWidgets.QListView()
        self.chat_list.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.chat_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.chat_list.setSpacing(6)
        self.chat_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.chat_list.customContextMenuRequested.connect(self._on_chat_context_menu)

        self.chat_model = ChatListModel()
        self.chat_list.setModel(self.chat_model)
        self.chat_list.setItemDelegate(ChatItemDelegate(self.chat_list))

        composer = QtWidgets.QWidget()
        composer_layout = QtWidgets.QHBoxLayout(composer)
        composer_layout.setContentsMargins(0, 0, 0, 0)
        composer_layout.setSpacing(10)

        self.input_field = QtWidgets.QLineEdit()
        self.input_field.setPlaceholderText("Ask about your documents... (Enter to send)")
        self.input_field.returnPressed.connect(self.send_question)

        self.send_btn = QtWidgets.QPushButton("Send")
        self.send_btn.setObjectName("accentButton")
        self.send_btn.clicked.connect(self.send_question)

        composer_layout.addWidget(self.input_field, 1)
        composer_layout.addWidget(self.send_btn, 0)

        chat_layout.addWidget(chat_top)
        chat_layout.addWidget(self.chat_list, 1)
        chat_layout.addWidget(composer, 0)

        self.center_splitter.addWidget(preview_panel)
        self.center_splitter.addWidget(chat_panel)
        self.center_splitter.setStretchFactor(0, 1)
        self.center_splitter.setStretchFactor(1, 2)

        right_layout.addWidget(header, 0)
        right_layout.addWidget(self.center_splitter, 1)

        root_layout.addWidget(left_panel, 0)
        root_layout.addWidget(right_container, 1)

        self._setup_menubar()
        self._setup_statusbar()
        self.apply_ui_enabled_state()
        self.load_folder_tree(self.folder_path)

    def _setup_menubar(self):
        menubar = self.menuBar()
        menubar.clear()

        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open Folder", self.select_folder_dialog)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("Settings", self.select_folder_dialog)
        tools_menu.addSeparator()
        tools_menu.addAction("Clear Cache", self.on_clear_cache)

        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about_dialog)
        help_menu.addAction("Logs", self.open_logs)

    def _setup_statusbar(self):
        sb = QtWidgets.QStatusBar()
        self.setStatusBar(sb)

        self.status_progress_bar = QtWidgets.QProgressBar()
        self.status_progress_bar.setMaximum(100)
        self.status_progress_bar.setMinimum(0)
        self.status_progress_bar.setValue(0)
        self.status_progress_bar.setVisible(False)
        self.status_progress_bar.setTextVisible(False)

        self.status_progress_label = QtWidgets.QLabel("Ready")
        self.status_progress_label.setStyleSheet("padding-left: 8px;")
        self.status_model_label = QtWidgets.QLabel(f"Model: {MODEL_NAME}")
        self.status_model_label.setStyleSheet("padding: 0px 8px; margin: 0px; border: none;")

        sb.addPermanentWidget(self.status_progress_bar, 0)
        sb.addPermanentWidget(self.status_progress_label, 1)
        sb.addPermanentWidget(self.status_model_label, 0)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sb.addPermanentWidget(spacer)

        self.status_settings_btn = QtWidgets.QPushButton("Settings")
        self.status_settings_btn.clicked.connect(self.select_folder_dialog)
        self.status_settings_btn.setMaximumWidth(110)
        self.status_settings_btn.setStyleSheet("margin: 2px 4px;")
        sb.addPermanentWidget(self.status_settings_btn)

        self.status_logs_btn = QtWidgets.QPushButton("Show Logs")
        self.status_logs_btn.clicked.connect(self.open_logs)
        self.status_logs_btn.setMaximumWidth(110)
        self.status_logs_btn.setStyleSheet("margin: 2px 4px;")
        sb.addPermanentWidget(self.status_logs_btn)

        self.status_clear_cache_btn = QtWidgets.QPushButton("Clear Cache")
        self.status_clear_cache_btn.clicked.connect(self.on_clear_cache)
        self.status_clear_cache_btn.setMaximumWidth(120)
        self.status_clear_cache_btn.setEnabled(bool(self.folder_path))
        self.status_clear_cache_btn.setStyleSheet("margin: 2px 4px 2px 4px;")
        sb.addPermanentWidget(self.status_clear_cache_btn)

    def _setup_tray(self):
        try:
            app = QtWidgets.QApplication.instance()
            if not app:
                return
            self.tray_icon = create_tray_icon(app)
            self.tray_icon.run_detached()
            app.aboutToQuit.connect(lambda: self.tray_icon.stop())
        except Exception:
            self.tray_icon = None

    def apply_ui_enabled_state(self):
        has_folder = bool(self.folder_path)
        self.send_btn.setEnabled(has_folder)
        self.input_field.setEnabled(has_folder)
        self.reindex_btn.setEnabled(has_folder)
        self.status_clear_cache_btn.setEnabled(has_folder)

    def _connect_signals(self):
        if not self.coordinator:
            return
        self.coordinator.indexing_started.connect(self._on_indexing_started)
        self.coordinator.indexing_finished.connect(self._on_indexing_finished)
        self.coordinator.indexing_error.connect(self._on_indexing_error)
        if hasattr(self.coordinator, "indexing_progress"):
            self.coordinator.indexing_progress.connect(self._on_indexing_progress)

    def _start_for_folder(self, folder_path: str, connect_signals: bool):
        self.folder_path = os.path.abspath(folder_path)
        self.folder_label.setText(self.folder_path)
        self.load_folder_tree(self.folder_path)

        if RAGCoordinator._instance is not None:
            try:
                RAGCoordinator._instance.close()
            except Exception:
                pass
            RAGCoordinator._instance = None

        self.coordinator = RAGCoordinator.instance(self.folder_path)
        if connect_signals:
            self._connect_signals()

        try:
            cache_dir = get_folder_cache_dir(self.folder_path)
            self.folder_label.setToolTip(f"Cache: {cache_dir}")
        except Exception:
            pass

        self.apply_ui_enabled_state()
        if getattr(self.coordinator, "is_indexing", False):
            self._on_indexing_started()

    def select_folder_dialog(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select a folder for indexing", os.getcwd()
        )
        if not folder:
            return
        self._start_for_folder(os.path.abspath(folder), connect_signals=True)

    def load_folder_tree(self, folder_path: Optional[str]):
        self.file_tree.setModel(None)
        self.fs_model = None

        if not folder_path:
            return

        self.fs_model = QtGui.QFileSystemModel(self)
        self.fs_model.setRootPath(folder_path)
        root_index = self.fs_model.index(folder_path)

        self.file_tree.setModel(self.fs_model)
        self.file_tree.setRootIndex(root_index)
        
        # Hide all columns except the first one (filename)
        for col in range(1, self.fs_model.columnCount()):
            self.file_tree.setColumnHidden(col, True)
        
        try:
            self.file_tree.expand(root_index)
        except Exception:
            pass

    def on_file_clicked(self, index: QtCore.QModelIndex):
        if not self.fs_model or not index.isValid():
            return
        try:
            if self.fs_model.isDir(index):
                return
        except Exception:
            return

        file_path = self.fs_model.filePath(index)
        self._load_file_preview(file_path)

    def _load_file_preview(self, file_path: str):
        if not file_path or not os.path.exists(file_path):
            self.preview_browser.setPlainText("Файл не найден.")
            return

        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        try:
            if ext in ("txt", "md"):
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    self.preview_browser.setPlainText(f.read())
            elif ext == "html":
                import bs4

                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    soup = bs4.BeautifulSoup(f.read(), "html.parser")
                self.preview_browser.setPlainText(soup.get_text())
            elif ext == "pdf":
                import PyPDF2

                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join((page.extract_text() or "") for page in reader.pages)
                self.preview_browser.setPlainText(text)
            elif ext == "docx":
                from docx import Document as DocxDocument

                doc = DocxDocument(file_path)
                self.preview_browser.setPlainText("\n".join(p.text for p in doc.paragraphs))
            elif ext in ("png", "jpg", "jpeg"):
                self.preview_browser.setPlainText(
                    "Изображение: OCR/превью может занять много времени.\n"
                    "TODO: добавить фоновое извлечение текста для изображений."
                )
            else:
                self.preview_browser.setPlainText(
                    f"Preview for extension '.{ext}' is not implemented.\n"
                    f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
                )
        except Exception as e:
            self.preview_browser.setPlainText(f"Preview error: {e}")

    def _ensure_coordinator(self) -> bool:
        if self.coordinator is None:
            QtWidgets.QMessageBox.information(
                self, "No folder selected", "First select a folder for indexing."
            )
            self.select_folder_dialog()
        return self.coordinator is not None

    def _scroll_to_bottom(self):
        try:
            self.chat_list.scrollToBottom()
        except Exception:
            pass

    def start_typing_animation(self):
        self.pending_typing_dots = 0
        self.typing_timer = QtCore.QTimer(self)
        self.typing_timer.timeout.connect(self.update_typing_dots)
        self.typing_timer.start(400)

    def update_typing_dots(self):
        self.pending_typing_dots = (self.pending_typing_dots + 1) % 4
        if self.pending_typing_chat_idx is not None:
            self.typing_states[self.pending_typing_chat_idx] = self.pending_typing_dots
            if self.pending_typing_chat_idx == self.current_chat_idx:
                self.chat_model.setTypingDots(self.pending_typing_dots)

    def stop_typing(self, chat_idx: Optional[int] = None):
        if self.typing_timer:
            self.typing_timer.stop()
            self.typing_timer = None

        if chat_idx is None:
            chat_idx = self.pending_typing_chat_idx

        if chat_idx is not None:
            self.typing_states.pop(chat_idx, None)

        if chat_idx == self.current_chat_idx:
            self.chat_model.removeTyping()

        if self.pending_typing_chat_idx == chat_idx:
            self.pending_typing_chat_idx = None

    def _add_typing_item_for_chat(self, chat_idx: int):
        if chat_idx not in self.typing_states:
            self.typing_states[chat_idx] = 0
        if chat_idx == self.current_chat_idx:
            if not (self.chat_model._messages and self.chat_model._messages[-1].kind == "typing"):
                self.chat_model.appendMessage(
                    ChatMessage(text="Генерация", isUser=False, timestamp="", kind="typing")
                )
                self._scroll_to_bottom()
                self._adjust_bubble_widths()

    def _adjust_bubble_widths(self):
        if not self.chat_model:
            return
        rows = self.chat_model.rowCount()
        if rows <= 0:
            return
        try:
            top_left = self.chat_model.index(0, 0)
            bottom_right = self.chat_model.index(rows - 1, 0)
            self.chat_model.dataChanged.emit(top_left, bottom_right, [])
            self.chat_model.layoutChanged.emit()
        except Exception:
            pass

    def send_question(self):
        query = self.input_field.text().strip()
        if not query:
            return
        if not self._ensure_coordinator():
            return

        file_filter = None
        if self.folder_path and " в " in query.lower():
            parts = query.lower().split(" в ")
            if len(parts) > 1:
                name = parts[1].strip().strip('"\'')
                path = os.path.join(self.folder_path, name)
                if os.path.exists(path):
                    file_filter = path

        self.add_message(query, is_user=True)
        self.input_field.clear()

        self.pending_typing_chat_idx = self.current_chat_idx
        self._add_typing_item_for_chat(self.pending_typing_chat_idx)
        self.start_typing_animation()

        self.coordinator.ask_async(
            query,
            file_filter,
            lambda resp, idx=self.pending_typing_chat_idx: self.on_answer(resp, idx),
        )

    def on_answer(self, response: dict, chat_idx: Optional[int] = None):
        if chat_idx is None:
            chat_idx = self.current_chat_idx
        self.stop_typing(chat_idx)

        result = response.get("result", "Нет ответа")
        sources = response.get("sources", "")

        self.add_message(result, chat_idx=chat_idx)
        if sources and sources != "Неизвестно":
            self.add_message(f"Источник: {sources}", is_source=True, chat_idx=chat_idx)

    def append_chat_message(self, user_text: str, ai_text: str):
        self.add_message(user_text, is_user=True)
        self.add_message(ai_text, is_user=False)

    def add_message(
        self,
        text: str,
        is_user: bool = False,
        is_source: bool = False,
        chat_idx: Optional[int] = None,
    ):
        if chat_idx is None:
            chat_idx = self.current_chat_idx

        ts = datetime.now().strftime("%H:%M")
        content = text if not is_source else f"{text}"

        if 0 <= chat_idx < len(self.conversations):
            self.conversations[chat_idx].append((content, is_user, ts))

        if chat_idx == self.current_chat_idx:
            self.chat_model.appendMessage(
                ChatMessage(text=self._html_escape(content), isUser=is_user, timestamp=ts)
            )
            self._scroll_to_bottom()
            self._adjust_bubble_widths()

    def _html_escape(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
            .replace("\n", "<br>")
        )

    def _on_indexing_started(self):
        self.status_progress_bar.setVisible(True)
        self.status_progress_bar.setValue(0)
        self.status_progress_label.setText("Indexing started...")
        self.status_clear_cache_btn.setEnabled(True)

    def _on_indexing_progress(self, current_files: int, total_files: int, percent: int):
        if not self.status_progress_bar.isVisible():
            self.status_progress_bar.setVisible(True)
        self.status_progress_bar.setValue(max(0, min(100, percent)))
        if total_files > 0:
            self.status_progress_label.setText(
                f"Processing {current_files}/{total_files} files • {percent}%"
            )
        else:
            self.status_progress_label.setText(f"Indexing... • {percent}%")

    def _on_indexing_finished(self):
        self.status_progress_bar.setVisible(False)
        self.status_progress_label.setText("Ready for questions!")
        self.apply_ui_enabled_state()
        self.add_message("Ready for questions!", chat_idx=self.current_chat_idx)

    def _on_indexing_error(self, msg: str):
        self.status_progress_bar.setVisible(False)
        self.status_progress_label.setText("Indexing error")
        self.add_message(f"Error: {msg}", chat_idx=self.current_chat_idx)

    def start_indexing(self, reason: str = "Indexing"):
        if not self._ensure_coordinator():
            return
        self.status_progress_label.setText(f"{reason}...")
        self.status_progress_bar.setVisible(True)
        self.status_progress_bar.setValue(0)
        self.add_message("Indexing started...", chat_idx=self.current_chat_idx)
        self.coordinator.start_indexing()

    def reindex(self):
        self.start_indexing(reason="Re-indexing")

    def _init_conversations(self):
        self.conversations = [[]]
        self.current_chat_idx = 0
        self.chat_threads_list.clear()
        self.chat_threads_list.addItem("New Chat")

    def on_chat_thread_clicked(self, item: QtWidgets.QListWidgetItem):
        text = item.text()
        if text == "New Chat":
            self.create_new_chat()
            return
        if text.startswith("Chat "):
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
        self.chat_list.setItemDelegate(ChatItemDelegate(self.chat_list))

        conv_number = self.current_chat_idx + 1
        existing = [self.chat_threads_list.item(i).text() for i in range(self.chat_threads_list.count())]
        if f"Chat {conv_number}" not in existing:
            self.chat_threads_list.addItem(f"Chat {conv_number}")

        for i in range(self.chat_threads_list.count()):
            if self.chat_threads_list.item(i).text() == f"Chat {conv_number}":
                self.chat_threads_list.setCurrentRow(i)
                break

        self.typing_states = {}
        self.pending_typing_chat_idx = None
        self._scroll_to_bottom()

    def switch_chat(self, idx: int):
        self.current_chat_idx = idx
        self.chat_model = ChatListModel()
        self.chat_list.setModel(self.chat_model)
        self.chat_list.setItemDelegate(ChatItemDelegate(self.chat_list))

        if 0 <= idx < len(self.conversations):
            for content, is_user, ts in self.conversations[idx]:
                self.chat_model.appendMessage(
                    ChatMessage(text=self._html_escape(content), isUser=is_user, timestamp=ts)
                )

        if idx in self.typing_states:
            dots = self.typing_states.get(idx, 0) or 0
            text = "Генерация" + "." * dots
            self.chat_model.appendMessage(
                ChatMessage(text=text, isUser=False, timestamp="", kind="typing")
            )

        self._scroll_to_bottom()
        self._adjust_bubble_widths()

    def _on_chat_threads_context_menu(self, pos: QtCore.QPoint):
        item = self.chat_threads_list.itemAt(pos)
        if not item:
            return
        if item.text() == "New Chat":
            return

        menu = QtWidgets.QMenu(self)
        delete_act = menu.addAction("Delete Chat")
        action = menu.exec(self.chat_threads_list.mapToGlobal(pos))
        if action == delete_act:
            self._delete_chat(item)

    def _on_chat_context_menu(self, pos: QtCore.QPoint):
        """Handle right-click on chat messages for copying"""
        index = self.chat_list.indexAt(pos)
        if not index.isValid():
            return
        
        text = self.chat_model.data(index, ChatListModel.TextRole) or ""
        if not text:
            return
        
        # Remove HTML tags for display
        import re
        clean_text = re.sub('<[^<]+?>', '', text)
        
        menu = QtWidgets.QMenu(self)
        copy_act = menu.addAction("Copy")
        action = menu.exec(self.chat_list.mapToGlobal(pos))
        if action == copy_act:
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setText(clean_text)

    def _delete_chat(self, item: QtWidgets.QListWidgetItem):
        row = self.chat_threads_list.row(item)
        conv_idx = row - 1  # row=0 is "New Chat"
        if not (0 <= conv_idx < len(self.conversations)):
            return

        self.conversations.pop(conv_idx)
        self.typing_states.pop(conv_idx, None)
        self.chat_threads_list.takeItem(row)

        for i in range(1, self.chat_threads_list.count()):
            self.chat_threads_list.item(i).setText(f"Chat {i}")

        if self.current_chat_idx >= len(self.conversations):
            self.current_chat_idx = max(0, len(self.conversations) - 1)
        self.switch_chat(self.current_chat_idx)

    def clear_chat_history(self):
        self.conversations = [[]]
        self.current_chat_idx = 0
        self.chat_threads_list.clear()
        self.chat_threads_list.addItem("New Chat")
        self.chat_model = ChatListModel()
        self.chat_list.setModel(self.chat_model)
        self.chat_list.setItemDelegate(ChatItemDelegate(self.chat_list))
        self.chat_threads_list.setCurrentRow(0)
        self.stop_typing(self.current_chat_idx)

    def on_clear_cache(self):
        if not self.folder_path:
            QtWidgets.QMessageBox.information(self, "No cache selected", "First select a folder.")
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete cache?",
            f"Delete cache for:\n{self.folder_path}?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            success = clear_folder_cache(self.folder_path)
        except Exception:
            success = False

        if success:
            QtWidgets.QMessageBox.information(self, "Cache cleared", "Cache cleared successfully.")
            if (
                QtWidgets.QMessageBox.question(
                    self,
                    "Reindex?",
                    "Reindex this folder now?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                )
                == QtWidgets.QMessageBox.StandardButton.Yes
            ):
                self.reindex()
        else:
            QtWidgets.QMessageBox.warning(
                self, "Error", "Failed to clear cache. Check permissions and try again."
            )

    def open_logs(self):
        try:
            from cache import get_cache_root

            log_path = os.path.join(get_cache_root(), "app.log")
            win = LogWindow(self, log_path=log_path)
            win.exec()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to open logs: {e}")

    def show_about_dialog(self):
        QtWidgets.QMessageBox.information(
            self,
            "About",
            "RAG Assistant (PyQt6)\nLocal indexing (FAISS + embeddings) + Ollama RAG.",
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._adjust_bubble_widths()

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            if getattr(self, "tray_icon", None):
                self.tray_icon.stop()
        except Exception:
            pass
        super().closeEvent(event)