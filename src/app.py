# src/app.py
import sys
import os
from PyQt5.QtWidgets import QApplication
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
    app.main_window.show()
    code = app.exec_()
    app.main_window.coordinator.close()  # <--- Добавь!
    sys.exit(code)
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()