import os
import pathlib

from fintorch.enum import TimeFrame

# Directories
USER_HOME_DIR = os.path.expanduser("~")
FINTORCH_ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
FINTORCH_DATA_DIR = os.environ.get("FINTORCH_DATA_DIR", os.path.join(USER_HOME_DIR, "Data"))
FINTORCH_TEMP_DIR = os.path.join(FINTORCH_ROOT_DIR, "tmp")
FINTORCH_DEEP_DIR = os.path.join(FINTORCH_DATA_DIR, "deep")
FINTORCH_SIMULATION_DIR = os.path.join(FINTORCH_DATA_DIR, "engine")
FINTORCH_TRANSFORM_DIR = os.path.join(FINTORCH_DEEP_DIR, "transform")
FINTORCH_MODULE_DIR = os.path.join(FINTORCH_DEEP_DIR, "module")
FINTORCH_ENGINE_DIR = os.path.join(FINTORCH_SIMULATION_DIR, "engine")
CONFIG_DIR = os.path.join(FINTORCH_ROOT_DIR, "fintorch/config")

CANDLES_COUNT = -1
BASE_TIME_FRAME = TimeFrame.MIN15
