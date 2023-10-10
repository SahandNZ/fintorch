from typing import Dict, List, Set, Tuple

import pandas as pd


class Data:
    def __init__(self):
        self.__symbols: Set[str] = set()
        self.__time_frames: Set[int] = set()
        self.__dict: Dict[Tuple[str, int], pd.DataFrame] = {}

    @property
    def symbols(self) -> List[str]:
        return list(self.__symbols)

    @property
    def time_frames(self) -> List[str]:
        return list(self.__time_frames)

    def __setitem__(self, key: Tuple[str, int], value):
        self.__symbols.add(key[0])
        self.__time_frames.add(key[1])
        self.__dict[key] = value

    def __getitem__(self, key):
        return self.__dict[key]
