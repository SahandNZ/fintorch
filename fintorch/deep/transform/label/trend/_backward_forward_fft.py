import pandas as pd
from matplotlib import pyplot as plt

from .._backward_forward import BackwardForwardLabelTransform
from .....enum import TimeFrame


class BackwardForwardFftLabelTransform(BackwardForwardLabelTransform):
    def __init__(
            self,
            symbol: str,
            time_frame: TimeFrame,
            backward: int = 10,
            forward: int = 10,
            muting_percentage: int = 95
    ):
        super().__init__(
            name="Backward Forward FFT",
            short_name="B.F-FFT",
            description="",
            symbol=symbol,
            time_frame=time_frame,
            backward=backward,
            forward=forward,
            classes=["UP", "DOWN"]
        )

        self.__muting_percentage: int = muting_percentage

    @property
    def muting_percentage(self) -> int:
        return self.__muting_percentage

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        pass
