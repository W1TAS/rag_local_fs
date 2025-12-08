"""
Install/uninstall per-user Explorer context menu entries for this app.

Creates two menu items for both files and folders:
 - "RAG: Рассказать об этом"  -> launches app with --tell
 - "RAG: Спросить у ассистента" -> launches app with --ask

This script writes to HKCU so admin privileges are not required.

Usage:
  python install_context_menu.py install
  python install_context_menu.py uninstall

After install, right-click any file or folder in Explorer and choose one of the actions.
"""
import os
import sys
import winreg

APP_LABEL_TELL = "RAG: Рассказать об этом"
APP_LABEL_ASK = "RAG: Спросить у ассистента"


def _create_cmd(python_exe, app_py, mode):
    # mode is '--tell' or '--ask'
    # Use "%1" to receive the clicked path from Explorer
    return f'"{python_exe}" "{app_py}" "%1" {mode}'


def _create_key(root, path):
    try:
        key = winreg.CreateKey(root, path)
        return key
    except Exception as e:
        print("Ошибка при создании ключа:", path, e)
        return None


def install():
    python_exe = sys.executable
    # Prefer pythonw on Windows to avoid opening a console window
    run_exe = python_exe
    if os.name == 'nt':
        pythonw_candidate = os.path.join(os.path.dirname(python_exe), 'pythonw.exe')
        if os.path.exists(pythonw_candidate):
            run_exe = pythonw_candidate
        else:
            # fallback to pythonw from PATH if any
            try:
                import shutil
                pw = shutil.which('pythonw')
                if pw:
                    run_exe = pw
            except Exception:
                pass
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    app_py = os.path.join(repo_root, 'src', 'app.py')
    # Try both rag_local_fs/assets/icons and parent repo assets/icons
    icons_dir_candidates = [
        os.path.join(repo_root, 'assets', 'icons'),
        os.path.join(repo_root, '..', 'assets', 'icons'),
    ]
    icons_dir = next((p for p in icons_dir_candidates if os.path.isdir(p)), icons_dir_candidates[0])
    context_icon = os.path.join(icons_dir, 'context_icon.ico')

    if not os.path.exists(app_py):
        print("Не найден файл приложения:", app_py)
        return

    # Use HKCU\Software\Classes to avoid requiring admin rights
    hkcu = winreg.HKEY_CURRENT_USER
    bases = [r"Software\Classes\*\shell", r"Software\Classes\Directory\shell"]

    for base in bases:
        # Tell action
        tell_key_path = base + "\\RAGAssistantTell"
        cmd_tell = _create_cmd(run_exe, app_py, '--tell')
        k = _create_key(hkcu, tell_key_path)
        if k:
            winreg.SetValueEx(k, None, 0, winreg.REG_SZ, APP_LABEL_TELL)
            # set icon (optional) - prefer custom icon in assets/icons
            try:
                if os.path.exists(context_icon):
                    winreg.SetValueEx(k, 'Icon', 0, winreg.REG_SZ, context_icon)
                else:
                    winreg.SetValueEx(k, 'Icon', 0, winreg.REG_SZ, python_exe)
            except Exception:
                pass
            winreg.CloseKey(k)
            cmd_k = _create_key(hkcu, tell_key_path + '\\command')
            if cmd_k:
                winreg.SetValueEx(cmd_k, None, 0, winreg.REG_SZ, cmd_tell)
                winreg.CloseKey(cmd_k)

        # Ask action
        ask_key_path = base + "\\RAGAssistantAsk"
        cmd_ask = _create_cmd(run_exe, app_py, '--ask')
        k = _create_key(hkcu, ask_key_path)
        if k:
            winreg.SetValueEx(k, None, 0, winreg.REG_SZ, APP_LABEL_ASK)
            try:
                if os.path.exists(context_icon):
                    winreg.SetValueEx(k, 'Icon', 0, winreg.REG_SZ, context_icon)
                else:
                    winreg.SetValueEx(k, 'Icon', 0, winreg.REG_SZ, python_exe)
            except Exception:
                pass
            winreg.CloseKey(k)
        cmd_k = _create_key(hkcu, ask_key_path + '\\command')
        if cmd_k:
            winreg.SetValueEx(cmd_k, None, 0, winreg.REG_SZ, cmd_ask)
            winreg.CloseKey(cmd_k)

    print("Контекстные пункты установлены (в HKCU). Перезапустите Проводник при необходимости.")


def uninstall():
    hkcu = winreg.HKEY_CURRENT_USER
    bases = [r"Software\Classes\*\shell\RAGAssistantTell",
             r"Software\Classes\Directory\shell\RAGAssistantTell",
             r"Software\Classes\*\shell\RAGAssistantAsk",
             r"Software\Classes\Directory\shell\RAGAssistantAsk"]

    for p in bases:
        try:
            # Need to delete command subkey first
            try:
                winreg.DeleteKey(hkcu, p + "\\command")
            except FileNotFoundError:
                pass
            try:
                winreg.DeleteKey(hkcu, p)
            except FileNotFoundError:
                pass
        except Exception as e:
            print("Ошибка при удалении", p, e)

    print("Контекстные пункты удалены (если они существовали).")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python install_context_menu.py install|uninstall")
        sys.exit(1)
    cmd = sys.argv[1].lower()
    if cmd == 'install':
        install()
    elif cmd == 'uninstall':
        uninstall()
    else:
        print("Unknown command:", cmd)