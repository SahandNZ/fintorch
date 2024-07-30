from ..enum import TimeFrame
from ..strategy import *

INITIAL_CAPITAL = 1000
INTERVAL = TimeFrame.MIN15

STRATEGY_TIME_FRAMES = [
    TimeFrame.HOUR4,
    TimeFrame.HOUR1,
]

STRATEGY_TYPES = [
    ActiveMarketToggleStrategy,
    ActiveMarketTrailingStopLossStrategy,
    
    ## these strategy will not include in final product
    # ActiveLimitTrailingStopLossStrategy,
    # ActiveLimitToggleStrategy,
    # PassiveLimitToggleStrategy,
    # PassiveMarketToggleStrategy,
]

__all__ = ["INITIAL_CAPITAL", "INTERVAL", "STRATEGY_SYMBOLS", "STRATEGY_TIME_FRAMES", "STRATEGY_TYPES"]