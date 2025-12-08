import os
import hashlib


def get_cache_root():
    localappdata = os.getenv('LOCALAPPDATA')
    if localappdata and 'Packages' not in localappdata:
        base = os.path.join(localappdata, 'RAGAssistant')
    else:
        userprofile = os.getenv('USERPROFILE') or os.path.expanduser('~')
        base = os.path.join(userprofile, 'AppData', 'Local', 'RAGAssistant')
    os.makedirs(base, exist_ok=True)
    return base


def get_folder_cache_dir(folder_path: str) -> str:
    folder_path = os.path.abspath(folder_path)
    name = os.path.basename(folder_path.rstrip(os.sep)) or 'root'
    h = hashlib.sha256(folder_path.encode('utf-8')).hexdigest()[:16]
    cache_root = get_cache_root()
    folder_cache = os.path.join(cache_root, f"{name}_{h}")
    os.makedirs(folder_cache, exist_ok=True)
    return folder_cache


def clear_folder_cache(folder_path: str) -> bool:
    import shutil

    try:
        cache_root = get_cache_root()
        folder_cache = get_folder_cache_dir(folder_path)
        folder_cache = os.path.abspath(folder_cache)
        cache_root = os.path.abspath(cache_root)
        if not folder_cache.startswith(cache_root):
            return False
        if os.path.exists(folder_cache):
            shutil.rmtree(folder_cache)
        return True
    except Exception:
        return False


def prepare_virtual_folder_for_file(file_path: str) -> str:
    import shutil

    if not os.path.isfile(file_path):
        raise ValueError(f"Not a file: {file_path}")

    file_path = os.path.abspath(file_path)
    cache_root = get_cache_root()
    file_hash = hashlib.sha256(file_path.encode('utf-8')).hexdigest()[:12]
    file_name = os.path.basename(file_path)
    virtual_folder = os.path.join(cache_root, f"_singlefile_{file_hash}")
    os.makedirs(virtual_folder, exist_ok=True)

    dest = os.path.join(virtual_folder, file_name)
    try:
        os.link(file_path, dest) if not os.path.exists(dest) else None
    except (OSError, FileExistsError):
        if not os.path.exists(dest):
            shutil.copy2(file_path, dest)

    return virtual_folder
