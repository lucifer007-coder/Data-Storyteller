import os
from typing import Tuple
from config.settings import Config

def allowed_file(filename: str) -> bool:
    if not filename:
        return False
    _, ext = os.path.splitext(filename)
    return ext.lower() in Config.SUPPORTED_FORMATS

def check_file_size_bytes(file_size_bytes: int) -> Tuple[bool, str]:
    """Return (is_ok, message) comparing uploaded file size (bytes) to MAX_FILE_SIZE (MB)."""
    max_mb = Config.MAX_FILE_SIZE
    size_mb = file_size_bytes / (1024**2)
    if size_mb > max_mb:
        return False, f"File size {size_mb:.1f} MB exceeds the maximum allowed {max_mb} MB."
    return True, f"File size {size_mb:.1f} MB within allowed limit."