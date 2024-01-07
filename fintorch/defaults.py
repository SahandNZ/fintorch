import os

import numpy as np
from rich.progress import TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, \
    SpinnerColumn, MofNCompleteColumn

# Directories
HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HOME_DIR, "Data"))
TRANSFORM_DIR = os.path.join(DATA_DIR, "transform")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")

# Default properties
NUMPY_FEATURE_DTYPE = np.float16
NUMPY_LABEL_DTYPE = np.int8

# Rich progress bar
RICH_PROGRESS_COLUMNS = [
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(show_speed=True),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
]
