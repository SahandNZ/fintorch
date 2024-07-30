import torch
from torch import nn

from ..deep.model.tsf import *
from ..deep.criterion import *
from ..deep.dataset import TimeFrameSequenceFeatureDataset
from ..deep.transform.feature import *
from ..deep.transform.label.trend import *
from ..enum import TimeFrame


FEATURE_TIME_FRAMES = [
    TimeFrame.HOUR4,
    TimeFrame.HOUR1,
    TimeFrame.MIN30
]

LABEL_TIME_FRAMES = [
    TimeFrame.HOUR4,
    TimeFrame.HOUR1
]

# types
FEATURE_TRANSFORM_TYPES = [
    PreviousFractalsFeatureTransform,
    RollingMeanStdTrRocFeatureTransform,
    TechnicalTrendIndicatorsFeatureTransform,
    
    # these feature transforms will not include in final product
    # SquareLogTrRocFeatureTransform,
    # StftTrRocFeatureTransform,
    # PreviousFractalsWithFundingRateFeatureTransform,
]

LABEL_TRANSFORM_TYPES = [
    BackwardForwardAverageLabelTransform,
    BackwardForwardMinimumLabelTransform,
    UpDownHeikinAshiLabelTransform
    
    ## these label transforms will not include in final product
    # BackwardForwardFftLabelTransform,
    # ForwardIchimokuLabelTransform,
    # ForwardRocLabelTransform,
    # NextFractalLabelTransform,
    # NextFractalSideLabelTransform,
    # TripleBarrierLabelTransform,
    # UpDownLabelTransform,
]

DATASET_TYPE = TimeFrameSequenceFeatureDataset

MODEL_TYPES = [
    GRU,
    LSTM,
    ResNet1D,
    
    ## these models will not include in final product
    # FeedForward,
    # Hybrid,
    # Transformer
]

# Model hyper-parameters
DROPOUT = 0.5
BATCH_NORM = True
ACTIVATION_FN = nn.Softmax(dim=-1)

DIM_INPUT_SEQUENCE = 2 ** 5
DIM_LATENT_SEQUENCE = 2 ** 4
DIM_LATENT_FEATURE = 2 ** 4
DIM_OUTPUT_SEQUENCE = 1

NUM_HIDDEN_LAYERS = 2

# Trainer hyperparameters
BATCH_SIZE = 2 ** 10
BATCH_COUNT = 2 ** 3

# Optimizer
OPTIM_TYPE = torch.optim.Adam
OPTIM_LR = 1e-4
OPTIM_WEIGHT_DECAY = 1e-6

# Learning Rate Scheduler
LRS_TYPE = torch.optim.lr_scheduler.StepLR
LRS_STEP_SIZE = 100
LRS_GAMMA = 0.1

TRAIN_PERIOD = TimeFrame.MONTH1 * 12
VAL_PERIOD = TimeFrame.MONTH1 * 6
TEST_PERIOD = TimeFrame.MONTH1 * 6

# Criterion 
LABEL_SMOOTHING = 0.0
CRITERION = CE(label_smoothing=LABEL_SMOOTHING)

SHUFFLE = True
EPOCHS_COUNT = 10
GRADIENT_CLIPPING_THRESHOLD = None
