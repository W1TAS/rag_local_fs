# src/app.py
import sys
import os
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    """
    Usage:
      python app.py <path> [--tell|--ask]

    - <path> can be a file or a directory. If it's a file, the parent directory
      will be used for indexing and the file path will be passed as a file filter
      to the assistant.
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
                # file: index its parent directory but set file filter
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
                        app.main_window.input_field.setPlainText(f"Вопрос про файл: {os.path.basename(initial_file_filter)}\n")
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