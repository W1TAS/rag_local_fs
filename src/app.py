# src/app.py
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
    initial_mode = None  # 'tell' or 'ask' or None
    initial_file_filter = None

    if len(sys.argv) >= 2:
        p = os.path.abspath(sys.argv[1])
        if os.path.exists(p):
            if os.path.isdir(p):
                folder_path = p
            else:
                # file: depending on mode, either use virtual folder or parent folder
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

    # If launching with a single file in --tell or --ask mode, use virtual folder
    # so we only index that file, not the entire parent folder
    if initial_file_filter and initial_mode in ("tell", "ask"):
        try:
            logging.info(f"Single-file mode: preparing virtual folder for {initial_file_filter}")
            folder_path = prepare_virtual_folder_for_file(initial_file_filter)
            initial_file_filter = None  # clear filter since we're now indexing just the file
        except Exception as e:
            logging.error(f"Failed to prepare virtual folder: {e}")
            print(f"Ошибка при подготовке виртуальной папки: {e}")
            sys.exit(1)

    # Configure logging early so modules can log to file
    try:
        log_dir = get_cache_root()
        # ensure log directory exists before creating handlers
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "app.log")
        handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(fmt)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

        # Redirect stdout to logger so prints are captured.
        # Do NOT redirect `sys.stderr` to the logger: when the logging
        # subsystem itself encounters an error it writes to stderr, and
        # redirecting stderr into the logging system can produce infinite
        # recursion and confusing "NoneType has no attribute 'write'" errors.
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
                    # If logging fails, fall back to original stdout to avoid recursion
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
        # keep stderr as the original stream so logging internals can write errors
        sys.stderr = orig_stderr
        logging.info("RAG Assistant starting")
    except Exception:
        # If logging setup fails, restore original streams and continue
        try:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.main_window = MainWindow(folder_path)

    # Ensure proper cleanup on quit
    app.aboutToQuit.connect(lambda: app.main_window.coordinator.close() if app.main_window.coordinator else None)

    # If an initial action was requested from context menu, schedule it
    if folder_path and initial_mode:
        def _on_index_ready():
            try:
                coord = app.main_window.coordinator
                if not coord:
                    return
                if initial_mode == "tell":
                    # Ask the canned question about files/folder
                    coord.ask_async("О чем файлы", initial_file_filter, lambda resp: app.main_window.on_answer(resp))
                elif initial_mode == "ask":
                    # Focus input field so user can type their question; pre-select text
                    app.main_window.input_field.setFocus()
                    if initial_file_filter:
                        # Optionally pre-fill clarification to user indicating file is targeted
                        app.main_window.input_field.setPlainText(f"Вопрос про файл {os.path.basename(initial_file_filter)}:\n")
                # disconnect after first call
                try:
                    coord.indexing_finished.disconnect(_on_index_ready)
                except Exception:
                    pass
            except Exception:
                pass

        # If coordinator already exists, attach to its signal
        coord = app.main_window.coordinator
        if coord:
            try:
                coord.indexing_finished.connect(_on_index_ready)
            except Exception:
                # if can't connect, try invoking directly
                _on_index_ready()

    app.main_window.show()
    code = app.exec()
    sys.exit(code)


if __name__ == "__main__":
    main()