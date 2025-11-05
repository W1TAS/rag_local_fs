# src/app.py
import sys
import os
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    if len(sys.argv) < 2:
        print("Использование: python app.py <путь_к_папке_с_документами>")
        sys.exit(1)

    folder_path = os.path.abspath(sys.argv[1])
    if not os.path.isdir(folder_path):
        print("Папка не найдена.")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.main_window = MainWindow(folder_path)
    # Ensure proper cleanup on quit
    app.aboutToQuit.connect(lambda: app.main_window.coordinator.close())
    app.main_window.show()
    code = app.exec()
    sys.exit(code)
    
if __name__ == "__main__":
    main()