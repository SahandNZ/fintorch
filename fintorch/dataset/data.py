from typing import Dict, List, Set, Tuple

import pandas as pd


class Data:
    def __init__(self, df_dict: Dict[Tuple[str, int], pd.DataFrame] = None):
        self.__symbols: Set[str] = set([item[0] for item in df_dict.keys()]) if df_dict is not None else set()
        self.__time_frames: Set[int] = set([item[1] for item in df_dict.keys()]) if df_dict is not None else set()
        self.__dict: Dict[Tuple[str, int], pd.DataFrame] = df_dict if df_dict is not None else {}

    @property
    def symbols(self) -> List[str]:
        return list(self.__symbols)

    @property
    def time_frames(self) -> List[str]:
        return list(self.__time_frames)

    def crop(self, start_timestamp: int = None, stop_timestamp: int = None):
        if start_timestamp is None and stop_timestamp is None:
            raise ValueError("Either start_timestamp or stop_timestamp must be passed.")

        data = Data()
        for key, df in self.__dict.items():
            if start_timestamp is not None:
                df = df[start_timestamp <= df.index]
            if stop_timestamp is not None:
                df = df[df.index < stop_timestamp]
            data[key] = df

        return data

    def __setitem__(self, key: Tuple[str, int], value):
        self.__symbols.add(key[0])
        self.__time_frames.add(key[1])
        self.__dict[key] = value

    def __getitem__(self, key):
        return self.__dict[key]
