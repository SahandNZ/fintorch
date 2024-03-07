import os
import pathlib

import numpy as np
from rich.progress import SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn, \
    TimeElapsedColumn, TimeRemainingColumn
from torch import nn

from fintorch.enum import TimeFrame

# Directories
USER_HOME_DIR = os.path.expanduser("~")
FINTORCH_ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
FINTORCH_DATA_DIR = os.environ.get("FINTORCH_DATA_DIR", os.path.join(USER_HOME_DIR, "Data"))
FINTORCH_TEMP_DIR = os.path.join(FINTORCH_ROOT_DIR, "/tmp")
TRANSFORM_DIR = os.path.join(FINTORCH_DATA_DIR, "transform")
MODULE_DIR = os.path.join(FINTORCH_DATA_DIR, "module")
CONFIG_DIR = os.path.join(FINTORCH_ROOT_DIR, "config")

# Storage
FEATURE_BYTES = os.environ.get("FINTORCH_FEATURE_BYTES", 4)
LABEL_BYTES = os.environ.get("FINTORCH_LABEL_PRECISION", 2)
NUMPY_FEATURE_DTYPE = np.dtype(f'f{FEATURE_BYTES}')
NUMPY_LABEL_DTYPE = np.dtype(f'f{LABEL_BYTES}')

# proxies
HTTP_PROXY = os.environ.get("FINTORCH_HTTP_PROXY", "http://tracker:nlOv5rC7cL3q3bYR@95.216.41.71:3128")
HTTPS_PROXY = os.environ.get("FINTORCH_HTTPS_PROXY", "http://tracker:nlOv5rC7cL3q3bYR@95.216.41.71:3128")
SOCKS5_PROXY = os.environ.get("FINTORCH_SOCKS5_PROXY", "dante-user:IiS8v39yGHyEMHeuuhQPOA43jryzeuT0@95.216.41.71:1080")
PROXIES = {"http": HTTP_PROXY, "https": HTTPS_PROXY, "socks5": SOCKS5_PROXY}

# Exchange
SYMBOL = "BTC-USDT"
TIME_FRAME = TimeFrame.DAY1
EXCHANGE_NAME = os.environ.get("FINTORCH_EXCHANGE_NAME", "binance")

# intervals
INTERVAL = os.environ.get("FINTORCH_TRADING_INTERVAL", TimeFrame.MIN15)

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

# Model hyper-parameters
DIM_SEQUENCE = 32
DIM_FEATURE = 4
DIM_OUTPUT = 2
NUM_HIDDEN_LAYERS = 2
DROPOUT = 0.5
BATCH_NORM = True
ACTIVATION_FN = nn.Softmax(dim=-1)

# Trainer hyper-parameters
LR = 1e-3
SHUFFLE = True
BATCH_SIZE = 128
EPOCHS_COUNT = 10
WEIGHT_DECAY = 1e-2
GRADIENT_CLIPPING_THRESHOLD = None

# Transform kwargs
TRANSFORM_KWARGS = {
    "symbol": SYMBOL,
    "time_frame": TIME_FRAME,
    "dim_sequence": DIM_SEQUENCE
}

# Model kwargs
MODEL_KWARGS = {
    "dim_sequence": DIM_SEQUENCE,
    "dim_feature": DIM_FEATURE,
    "dim_output": DIM_OUTPUT,
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
    "batch_norm": BATCH_NORM,
    "dropout": DROPOUT,
    "activation_fn": ACTIVATION_FN
}
