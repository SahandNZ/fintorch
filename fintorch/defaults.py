import os

from rich.progress import TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, \
    SpinnerColumn, MofNCompleteColumn

# directories
HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HOME_DIR, "Data"))
TRANSFORM_DIR = os.path.join(DATA_DIR, "transform")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")

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
