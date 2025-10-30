# src/ui/tray.py
from pystray import Icon, Menu, MenuItem
from PIL import Image
import os

def create_tray_icon(app):
    # Иконка (можно заменить на свою)
    icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
    if not os.path.exists(icon_path):
        # Создаём простую иконку
        image = Image.new("RGB", (64, 64), "#4CAF50")
    else:
        image = Image.open(icon_path)

    def show_window():
        app.main_window.show()
        app.main_window.raise_()
        app.main_window.activateWindow()

    def quit_app():
        app.quit()

    menu = Menu(
        MenuItem("Показать", show_window),
        MenuItem("Переиндексировать", lambda: app.main_window.reindex()),
        MenuItem("Выход", quit_app),
    )

    icon = Icon("RAG Assistant", image, menu=menu)
    return icon