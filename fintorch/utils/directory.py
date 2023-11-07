import os
import pathlib


def create_directory(path: str) -> bool:
    if not os.path.exists(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
