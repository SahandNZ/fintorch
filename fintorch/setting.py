import os

import numpy as np
from rich.progress import *

from fintorch.enum import TimeFrame

# Directories
HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.environ.get("FINTORCH_DATA_DIR", os.path.join(HOME_DIR, "Data"))
CANDLE_DIR = os.path.join(DATA_DIR, "candle")
TRANSFORM_DIR = os.path.join(DATA_DIR, "transform")
MODULE_DIR = os.path.join(DATA_DIR, "module")

# Storage
FEATURE_BYTES = os.environ.get("FINTORCH_FEATURE_BYTES", 4)
LABEL_BYTES = os.environ.get("FINTORCH_LABEL_PRECISION", 2)
NUMPY_FEATURE_DTYPE = np.dtype(f'f{FEATURE_BYTES}')
NUMPY_LABEL_DTYPE = np.dtype(f'f{LABEL_BYTES}')
FILE_COMPRESS_FACTOR = 128

# proxies
HTTP_PROXY = os.environ.get("FINTORCH_HTTP_PROXY", "http://tracker:nlOv5rC7cL3q3bYR@95.216.41.71:3128")
HTTPS_PROXY = os.environ.get("FINTORCH_HTTPS_PROXY", "http://tracker:nlOv5rC7cL3q3bYR@95.216.41.71:3128")
SOCKS5_PROXY = os.environ.get("FINTORCH_SOCKS5_PROXY", "dante-user:IiS8v39yGHyEMHeuuhQPOA43jryzeuT0@95.216.41.71:1080")
# PROXIES = {"http": HTTP_PROXY, "https": HTTPS_PROXY, "socks5": SOCKS5_PROXY}
PROXIES = None

# Exchange
EXCHANGE_NAME = os.environ.get("FINTORCH_EXCHANGE_NAME", "binance")

# intervals
TRADING_INTERVAL = os.environ.get("FINTORCH_TRADING_INTERVAL", TimeFrame.MIN15)
SAMPLING_INTERVAL = os.environ.get("FINTORCH_TRADING_INTERVAL", TimeFrame.MIN15)

# Data
BASE_TIME_FRAME = os.environ.get("FINTORCH_BASE_TIME_FRAME", 900)
CANDLE_COUNTS = os.environ.get("FINTORCH_CANDLE_COUNTS", None)

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
