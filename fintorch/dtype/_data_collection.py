from typing import Dict, List, Tuple

import pandas as pd

from ._symbol_info import SymbolInfo
from ..enum import TimeFrame


class DataCollection:
    def __init__(
            self,
            symbols_info_dict: Dict[str, SymbolInfo] = None,
            funding_rates_df_dict: Dict[str, pd.DataFrame] = None,
            candles_df_dict: Dict[Tuple[str, TimeFrame], pd.DataFrame] = None,
    ):
        self.__symbol_to_symbols_info_dict: Dict[str, SymbolInfo] = symbols_info_dict or {}
        self.__symbol_to_funding_rates_df_dict: Dict[str, pd.DataFrame] = funding_rates_df_dict or {}
        self.__stf_to_candles_df_dict: Dict[Tuple[str, TimeFrame], pd.DataFrame] = candles_df_dict or {}

    @property
    def symbols(self) -> List[str]:
        return list(set(self.__symbol_to_symbols_info_dict.keys()))

    @property
    def time_frames(self) -> List[TimeFrame]:
        return list(set([item[1] for item in self.__stf_to_candles_df_dict.keys()]))

    def has_symbol_info(self, symbol: str) -> bool:
        return symbol in self.__symbol_to_symbols_info_dict

    def set_symbol_info(self, symbol: str, symbol_info: SymbolInfo) -> None:
        self.__symbol_to_symbols_info_dict[symbol] = symbol_info

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return self.__symbol_to_symbols_info_dict[symbol]

    def has_candles_df(self, symbol: str, time_frame: TimeFrame) -> bool:
        return (symbol, time_frame) in self.__stf_to_candles_df_dict

    def set_candles_df(self, symbol: str, time_frame: TimeFrame, df: pd.DataFrame) -> None:
        self.__stf_to_candles_df_dict[(symbol, time_frame)] = df

    def get_candles_df(self, symbol: str, time_frame: TimeFrame):
        return self.__stf_to_candles_df_dict[(symbol, time_frame)]

    def set_funding_rates_df(self, symbol: str, df: pd.DataFrame) -> None:
        self.__symbol_to_funding_rates_df_dict[symbol] = df

    def get_funding_rates_df(self, symbol: str) -> pd.DataFrame:
        return self.__symbol_to_funding_rates_df_dict[symbol]
