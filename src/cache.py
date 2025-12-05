import os
import hashlib


def get_cache_root():
    # Prefer the standard Windows Local AppData folder (C:\Users\<user>\AppData\Local).
    # Avoid using the MS Store package-local cache path (which contains 'Packages').
    localappdata = os.getenv('LOCALAPPDATA')
    if localappdata and 'Packages' not in localappdata:
        base = os.path.join(localappdata, 'RAGAssistant')
    else:
        # Fallback: construct path from USERPROFILE (standard non-packaged location)
        userprofile = os.getenv('USERPROFILE') or os.path.expanduser('~')
        base = os.path.join(userprofile, 'AppData', 'Local', 'RAGAssistant')
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


def clear_folder_cache(folder_path: str) -> bool:
    """Remove the cache directory for the given folder_path.

    Returns True if removed or did not exist, False on error.
    Safety: ensures that deletion target is inside the cache root.
    """
    import shutil

    try:
        cache_root = get_cache_root()
        folder_cache = get_folder_cache_dir(folder_path)
        # Safety check: ensure folder_cache is inside cache_root
        folder_cache = os.path.abspath(folder_cache)
        cache_root = os.path.abspath(cache_root)
        if not folder_cache.startswith(cache_root):
            return False
        if os.path.exists(folder_cache):
            shutil.rmtree(folder_cache)
        return True
    except Exception:
        return False
