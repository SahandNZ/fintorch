import os

import torch
from rich.progress import *
from torch import nn

from .deep.criterion import *
from .deep.model import *
from .deep.transform.feature import *
from .deep.transform.label.trend import *
from .enum import TimeFrame
from .strategy.deep import *

# config files
API_CONFIG = "api"
STF_CONFIG = "btc-1d"
LOGGER_CONFIG = "debug"

MAX_WORKERS = os.cpu_count()

# API and Exchanges
EXCHANGE_NAME = "binance"

# Date
START_DATE = "2019-01-01"
STOP_DATE = "2025-01-01"

# Simulation
INITIAL_CAPITAL = 1000
INTERVAL = TimeFrame.MIN15

# types
FEATURE_TRANSFORM_TYPES = [
    # PreviousFractalsFeatureTransform,
    # PreviousFractalsWithFundingRateFeatureTransform,
    RollingMeanStdTrRocFeatureTransform,
    # SquareLogTrRocFeatureTransform,
    StftTrRocFeatureTransform,
    # TechnicalTrendIndicatorsFeatureTransform,
]

LABEL_TRANSFORM_TYPES = [
    # BackwardForwardFftLabelTransform,
    BackwardForwardMeanLabelTransform,
    BackwardForwardMinimumLabelTransform,
    # ForwardRocLabelTransform,
    # NextFractalLabelTransform,
    # NextFractalSideLabelTransform,
    # TripleBarrierLabelTransform,
    # UpDownLabelTransform,
    # UpDownHeikinAshiLabelTransform
]

TRANSFORM_TYPES = FEATURE_TRANSFORM_TYPES + LABEL_TRANSFORM_TYPES

MODEL_TYPES = [
    # FeedForward,
    # GRU,
    LSTM,
    # Hybrid,
    ResNet1D,
    # Transformer
]

DEEP_STRATEGY_TYPES = [
    DynamicAtrTpSlDeepStrategy,
    DynamicRocTpSlDeepStrategy,
    SimpleDeepStrategy,
    StaticTpSlDeepStrategy,
]

# Model hyper-parameters
DIM_INPUT_SEQUENCE = 16
DIM_OUTPUT_SEQUENCE = 1
NUM_HIDDEN_LAYERS = 1
DROPOUT = 0.5
BATCH_NORM = True
ACTIVATION_FN = nn.Softmax(dim=-1)

# Trainer hyper-parameters
BATCH_SIZE = 16

# Optimizer
OPTIM_TYPE = torch.optim.Adam
OPTIM_LR = 0.001
OPTIM_WEIGHT_DECAY = 0.01

# Learning Rate Scheduler
LRS_TYPE = torch.optim.lr_scheduler.StepLR
LRS_STEP_SIZE = 1000
LRS_GAMMA = 0.5

TRAIN_PERIOD = TimeFrame.MONTH1 * 12
VAL_PERIOD = TimeFrame.MONTH1 * 1
TEST_PERIOD = TimeFrame.MONTH1 * 1

CRITERION = FocalMSE()
SHUFFLE = True
EPOCHS_COUNT = 10
GRADIENT_CLIPPING_THRESHOLD = None

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
