import os
import shutil
from pathlib import Path


def force_delete_directory(dir_path):
    path = Path(dir_path)
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)