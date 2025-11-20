import os
import hashlib


def get_cache_root():
    # Prefer Windows LOCALAPPDATA, fallback to user's home .cache
    localappdata = os.getenv('LOCALAPPDATA')
    if localappdata:
        base = os.path.join(localappdata, 'RAGAssistant')
    else:
        base = os.path.join(os.path.expanduser('~'), '.cache', 'rag_assistant')
    os.makedirs(base, exist_ok=True)
    return base


def get_folder_cache_dir(folder_path: str) -> str:
    """Return a per-folder cache directory inside the global cache root.

    Uses a short SHA256 of the absolute folder path to avoid collisions and keep
    directory names manageable.
    """
    folder_path = os.path.abspath(folder_path)
    name = os.path.basename(folder_path.rstrip(os.sep)) or 'root'
    h = hashlib.sha256(folder_path.encode('utf-8')).hexdigest()[:16]
    cache_root = get_cache_root()
    folder_cache = os.path.join(cache_root, f"{name}_{h}")
    os.makedirs(folder_cache, exist_ok=True)
    return folder_cache
