from typing import List, Callable, Tuple


class Event:
    def __init__(self):
        self.callbacks: List[Callable] = []

    def add(self, callback: Callable) -> None:
        self.callbacks.append(callback)

    def trigger(self, args: Tuple):
        for callback in self.callbacks:
            callback(*args)
