import os
import pathlib

from fintorch.enum import TimeFrame

# Directories
USER_HOME_DIR = os.path.expanduser("~")
FINTORCH_ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
FINTORCH_DATA_DIR = os.environ.get("FINTORCH_DATA_DIR", os.path.join(USER_HOME_DIR, "Data"))
FINTORCH_TEMP_DIR = os.path.join(FINTORCH_ROOT_DIR, "/tmp")
TRANSFORM_DIR = os.path.join(FINTORCH_DATA_DIR, "transform")
MODULE_DIR = os.path.join(FINTORCH_DATA_DIR, "module")
CONFIG_DIR = os.path.join(FINTORCH_ROOT_DIR, "fintorch/config")

CANDLES_COUNT = -1
BASE_TIME_FRAME = TimeFrame.MIN15
