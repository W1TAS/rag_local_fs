import sys
import os
import logging
from logging.handlers import RotatingFileHandler
from cache import get_cache_root, prepare_virtual_folder_for_file
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    """
    Usage:
      python app.py <path> [--tell|--ask]

    - <path> can be a file or a directory. If it's a file, the parent directory
      will be used for indexing and the file path will be passed as a file filter
      to the assistant. NEW: if launched with --tell or --ask, single file will be
      indexed only (via virtual folder in cache) instead of scanning parent folder.
    - --tell : index and immediately ask the canned question "О чем файлы"
    - --ask  : index and open UI so user can type their question
    """
    folder_path = None
    initial_mode = None
    initial_file_filter = None

    if len(sys.argv) >= 2:
        p = os.path.abspath(sys.argv[1])
        if os.path.exists(p):
            if os.path.isdir(p):
                folder_path = p
            else:
                folder_path = os.path.abspath(os.path.dirname(p))
                initial_file_filter = p
        else:
            print("Путь не найден:", p)
            sys.exit(1)

    if len(sys.argv) >= 3:
        arg = sys.argv[2].lower()
        if arg in ("--tell", "tell"):
            initial_mode = "tell"
        elif arg in ("--ask", "ask"):
            initial_mode = "ask"

    if initial_file_filter and initial_mode in ("tell", "ask"):
        try:
            logging.info(f"Single-file mode: preparing virtual folder for {initial_file_filter}")
            folder_path = prepare_virtual_folder_for_file(initial_file_filter)
            # Clear the file filter: virtual folder contains only this file,
            # so we don't need to filter. This avoids metadata lookup issues
            # where the file path may change after being moved.
            initial_file_filter = None
        except Exception as e:
            logging.error(f"Failed to prepare virtual folder: {e}")
            print(f"Ошибка при подготовке виртуальной папки: {e}")
            sys.exit(1)

    try:
        log_dir = get_cache_root()
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "app.log")
        handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(fmt)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr

        class StreamToLogger:
            def __init__(self, logger, level=logging.INFO):
                self.logger = logger
                self.level = level
            def write(self, buf):
                try:
                    for line in buf.rstrip().splitlines():
                        self.logger.log(self.level, line)
                except Exception:
                    try:
                        orig_stdout.write(buf)
                    except Exception:
                        pass
            def flush(self):
                try:
                    orig_stdout.flush()
                except Exception:
                    pass

        sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
        sys.stderr = orig_stderr
        logging.info("RAG Assistant starting")
    except Exception:
        try:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        except Exception:
            pass

    app = QApplication(sys.argv)
    # Set application icon (taskbar) from assets/icons if available
    try:
        from PySide6.QtGui import QIcon
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        icons_dir_candidates = [
            os.path.join(repo_root, 'assets', 'icons'),
            os.path.join(repo_root, '..', 'assets', 'icons'),
        ]
        icons_dir = next((p for p in icons_dir_candidates if os.path.isdir(p)), icons_dir_candidates[0])
        app_icon_candidates = [
            os.path.join(icons_dir, 'app_icon.ico'),
            os.path.join(icons_dir, 'app_icon.png'),
        ]
        app_icon_path = next((p for p in app_icon_candidates if os.path.exists(p)), None)
        if app_icon_path:
            try:
                app.setWindowIcon(QIcon(app_icon_path))
            except Exception:
                pass
    except Exception:
        pass
    app.main_window = MainWindow(folder_path)

    app.aboutToQuit.connect(lambda: app.main_window.coordinator.close() if app.main_window.coordinator else None)

    if folder_path and initial_mode:
        def _on_index_ready():
            try:
                coord = app.main_window.coordinator
                if not coord:
                    return
                if initial_mode == "tell":
                    coord.ask_async("О чем файлы", initial_file_filter, lambda resp: app.main_window.on_answer(resp))
                elif initial_mode == "ask":
                    app.main_window.input_field.setFocus()
                    if initial_file_filter:
                        app.main_window.input_field.setPlainText(f"Вопрос про файл {os.path.basename(initial_file_filter)}:\n")
                try:
                    coord.indexing_finished.disconnect(_on_index_ready)
                except Exception:
                    pass
            except Exception:
                pass

        coord = app.main_window.coordinator
        if coord:
            try:
                coord.indexing_finished.connect(_on_index_ready)
            except Exception:
                _on_index_ready()

    app.main_window.show()
    code = app.exec()
    sys.exit(code)


if __name__ == "__main__":
    main()