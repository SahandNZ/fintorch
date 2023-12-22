import argparse
import itertools
import json
import os
import warnings

from rich.progress import Progress
from rich.progress import SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn, \
    TimeElapsedColumn, TimeRemainingColumn

from fintorch.module import Module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modules-path", action="store", type=str, required=False, default="config/modules.json")
    args = parser.parse_args()

    progress_bar_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    # load symbols and time_frames
    with open(args.modules_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # find modules paths
    modules_paths = []
    for symbol, time_frame in itertools.product(symbols, time_frames):
        data_root = os.environ.get("DATA_ROOT", "./data")
        module_experiment_root = os.path.join(data_root, "module/experiment", symbol, str(time_frame))
        if os.path.exists(module_experiment_root):
            file_names = os.listdir(module_experiment_root)
            for file_name in file_names:
                modules_paths.append(os.path.join(module_experiment_root, file_name))

    # load and store modules in deployment mode
    with Progress(*progress_bar_columns) as progress:
        task = progress.add_task(description="Prepare modules", total=len(modules_paths))

        for module_experiment_path in modules_paths:
            module_deployment_path = module_experiment_path.replace("experiment", "deployment")
            if not os.path.exists(module_deployment_path):
                module = Module.load(path=module_experiment_path, auto_cuda=False)
                module.save(mode="deployment")

            progress.update(task, advance=1)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
