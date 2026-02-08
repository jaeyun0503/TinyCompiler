"""
Run this script to empty o and dot folders.
"""

import os
import shutil
from pathlib import Path

def empty_folder(path):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)

        if os.path.isfile(full_path) or os.path.islink(full_path):
            os.remove(full_path)
        else:
            shutil.rmtree(full_path)

if __name__ == "__main__":
    empty_folder(Path(__file__).resolve().parent / "o")
    empty_folder(Path(__file__).resolve().parent / "dot")