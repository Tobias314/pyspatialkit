from typing import Union
from pathlib import Path
from sys import platform
import os
import shutil

def close_all_files_delete_dir(dir_path):
    if platform == "linux" or platform == "linux2": #very hacky: force close file handles
        for n in range(4,1000):
            try:
                os.close(n)
            except:
                pass
    path = Path(dir_path)
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

def get_testdata_path():
    path = Path(os.path.realpath(__file__))
    return path.parent.parent / "testdata/"

def get_tmp_path():
    path = Path(os.path.realpath(__file__))
    return path.parent / "tmp/"