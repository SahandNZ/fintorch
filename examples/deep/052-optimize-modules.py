import argparse
import itertools
import json
import time

from multiprocessing import Process, Queue
from typing import Dict, List, Type

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from torch import nn

from examples.args import add_default_args_and_parse
from fintorch.deep.dtype import SfDataset
from fintorch.deep.model import FeedForward, GRU, Hybrid, LSTM, Model, ResNet1D, Transformer
from fintorch.deep.module import Module
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.enum import TimeFrame


def target(module: Module, queue: Queue):
    try:
        for fold in module.optimize_and_store():
            queue.put(str(fold))
        queue.put(-1)
    except Exception as e:
        queue.put(str(e))
        time.sleep(10)
        queue.put(-1)


def create_process(args, items: List, model_params: Dict):
    for symbol, time_frame, feature_transform, label_transform, model_type in items:
        # define dataset
        dataset = SfDataset(
            start_date=args.start_date,
            stop_date=args.stop_date,
            interval=args.interval,
            symbol=symbol,
            time_frame=time_frame,
            sequence_length=args.dim_sequence,
            feature_transform=feature_transform,
            label_transform=label_transform
        )

        # define model
        model = model_type(**model_params)

        # define module
        module = Module(
            dataset=dataset,
            model=model,
        )

        # create process
        process_args = (module, Queue())
        process = Process(target=target, args=process_args)

        yield process, process_args


def run_multi_process(args, symbols: List[str], time_frames: List[TimeFrame],
                      feature_transforms: List[FeatureTransform], label_transforms: List[LabelTransform],
                      model_types: List[Type[Model]], model_params: Dict):
    # create rich main layout
    rows, columns = 6, 4
    main_layout = Layout()
    main_layout.split_column(*[Layout(name=f"row-{i}") for i in range(rows)])
    free_layouts: List[Layout] = []
    for i in range(rows):
        layouts = [Layout(Panel("Pending..."), name=f"col-{j}") for j in range(columns)]
        main_layout[f"row-{i}"].split_row(*layouts)
        free_layouts.extend(layouts)

    # create process generator to avoid from too many open files issue
    items = list(itertools.product(symbols, time_frames, feature_transforms, label_transforms, model_types))
    process_generator = create_process(args, items, model_params)

    # create terminal layout to track processes
    process_to_args = {}
    process_to_layout = {}
    running_process_set = set()
    done_process_set = set()
    with Live(main_layout, refresh_per_second=2):
        while len(done_process_set) < len(items):
            # start process if there is free worker (processor)
            if len(running_process_set) < args.max_workers:
                process, process_args = next(process_generator)
                process_to_args[process] = process_args
                process_to_layout[process] = free_layouts.pop(0)
                running_process_set.add(process)
                process.start()

            # update terminal live display
            for process in running_process_set:
                module, queue = process_to_args[process]
                layout = process_to_layout[process]
                if not queue.empty():
                    message = queue.get()
                    if isinstance(message, str):
                        layout.update(Panel(message, title=f"[blue]{module}"))
                    elif isinstance(message, int) and -1 == message:
                        layout.update(Panel("Pending..."))
                        free_layouts.append(layout)
                        done_process_set.add(process)

            # remove done processes from running set
            running_process_set = running_process_set - done_process_set

            # remove args of done processes
            for process in done_process_set:
                if process in process_to_args:
                    del process_to_args[process]


def main():
    parser = argparse.ArgumentParser()
    args = add_default_args_and_parse(parser)

    # load symbols and time frames
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature transforms
    feature_transforms = [
        RollingMeanStdTrRocFeatureTransform(sequence_length=args.dim_sequence),
        StftTrRocFeatureTransform(sequence_length=args.dim_sequence),
    ]

    # define label transforms
    label_transforms = [
        ForwardBackwardMinimumLabelTransform(),
        ForwardIchimokuLabelTransform(),
        ForwardMiddleSmaLabelTransform(),
        ForwardRocLabelTransform(),
        NextFractalLabelTransform(),
        UpDownLabelTransform()
    ]

    # define model params
    model_params = {
        "dim_sequence": args.dim_sequence,
        "dim_feature": 4,
        "dim_output": 2,
        "num_hidden_layers": args.num_hidden_layers,
        "batch_norm": args.no_batch_norm,
        "dropout": args.dropout,
        "activation_fn": nn.Softmax(dim=-1)
    }

    # define model_types
    model_types = [
        FeedForward,
        GRU,
        Hybrid,
        LSTM,
        ResNet1D,
        Transformer
    ]

    run_multi_process(args=args, symbols=symbols, time_frames=time_frames, feature_transforms=feature_transforms,
                      label_transforms=label_transforms, model_types=model_types, model_params=model_params)


if __name__ == '__main__':
    main()
