import importlib
import inspect
import os
import pathlib
from typing import Dict, Callable


def create_directory(path: str) -> bool:
    if not os.path.exists(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def import_class(module: str):
    for name, cls in inspect.getmembers(importlib.import_module(module), inspect.isclass):
        if module == cls.__module__:
            return cls


def call_with_dict(function: Callable, params: Dict):
    parameters_name = [p.name for p in inspect.signature(function).parameters.values()]
    params = {k.replace('-', '_'): v for k, v in params.items()}
    params = {k: v for k, v in params.items() if k in parameters_name}
    return function(**params)
