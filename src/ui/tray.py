from pystray import Icon, Menu, MenuItem
from PIL import Image
import os


def create_tray_icon(app):
    # assets may live in rag_local_fs/assets/icons or at the parent project root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    icons_dir_candidates = [
        os.path.join(repo_root, 'assets', 'icons'),
        os.path.join(repo_root, '..', 'assets', 'icons'),
    ]
    icons_dir = next((p for p in icons_dir_candidates if os.path.isdir(p)), icons_dir_candidates[0])
    tray_icon_candidates = [
        os.path.join(icons_dir, 'tray_icon.ico'),
        os.path.join(icons_dir, 'tray_icon.png'),
        os.path.join(icons_dir, 'tray.png'),
    ]
    image = None
    for p in tray_icon_candidates:
        if os.path.exists(p):
            try:
                image = Image.open(p)
                break
            except Exception:
                image = None
    if image is None:
        image = Image.new("RGB", (64, 64), "#FFFFFF")

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