import importlib
import inspect
from typing import Dict, Callable


def import_class(module: str, contains: str = None):
    for name, cls in inspect.getmembers(importlib.import_module(module), inspect.isclass):
        if contains is None or contains.lower() in name.lower():
            return cls


def call_with_dict(function: Callable, params: Dict):
    parameters_name = [p.name for p in inspect.signature(function).parameters.values()]
    params = {k.replace('-', '_'): v for k, v in params.items()}
    params = {k: v for k, v in params.items() if k in parameters_name}
    return function(**params)
