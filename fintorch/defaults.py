import os

from rich.progress import TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, \
    SpinnerColumn, MofNCompleteColumn

# directories
DATA_DIR = os.environ.get("DATA_DIR", "./data")
TRANSFORM_DIR = os.path.join(DATA_DIR, "transform")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")

Label_TRANSFORM_DIR = os.path.join(DATA_DIR, "transform/label")

# rich progress bar
RICH_PROGRESS_COLUMNS = [
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(show_speed=True),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
]
