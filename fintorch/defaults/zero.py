from ..enum import TimeFrame
from rich.progress import SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, \
    MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
    
    
# Date
START_DATE = "2019-01-01"
STOP_DATE = "2025-01-01"

# config files
API_CONFIG = "finland-proxy"
LOGGER_CONFIG = "debug"
SYMBOLS_CONFIG = 10

# Time frames
TIME_FRAMES = [
    TimeFrame.DAY1,
    TimeFrame.HOUR4,
    TimeFrame.HOUR1,
    TimeFrame.MIN15
]

# Rich progress columns
RICH_PROGRESS_COLUMNS = [
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(show_speed=True),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
]

__all__ = [
    "START_DATE", "STOP_DATE", "API_CONFIG", "LOGGER_CONFIG",
    "SYMBOLS_CONFIG", "TIME_FRAMES", "RICH_PROGRESS_COLUMNS"
]