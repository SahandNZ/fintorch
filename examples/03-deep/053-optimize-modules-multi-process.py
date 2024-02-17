import argparse
import itertools
import json
import time

from multiprocessing import Process, Queue
from typing import List

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from torch import nn

from examples.args import add_default_args_and_parse
from fintorch.deep.dtype import Dataset
from fintorch.deep.model import FeedForward, GRU, Hybrid, LSTM, ResNet1D, Transformer
from fintorch.deep.module import Module
from fintorch.deep.transform.feature import *
from fintorch.deep.transform.label import *
from fintorch.dtype import DataCollection
from fintorch.exchange import ONLINE_EXCHANGE
from fintorch.utils.function import call_with_dict
from fintorch.utils.timestamp import create_timestamps


def target(module: Module, dc: DataCollection, queue: Queue):
    try:
        for fold in module.optimize(dc=dc):
            queue.put(str(fold))
        queue.put(-1)
    except Exception as e:
        queue.put(str(e))
        time.sleep(10)
        queue.put(-1)


def create_process(modules: List[Module], dc: DataCollection):
    for module in modules:
        # create process
        process_args = (module, dc, Queue())
        process = Process(target=target, args=process_args)

        yield process, process_args


def run_multi_process(args, modules: List[Module]):
    # create rich main layout
    rows, columns = 6, 4
    main_layout = Layout()
    main_layout.split_column(*[Layout(name=f"row-{i}") for i in range(rows)])
    free_layouts: List[Layout] = []
    for i in range(rows):
        layouts = [Layout(Panel("Pending..."), name=f"col-{j}") for j in range(columns)]
        main_layout[f"row-{i}"].split_row(*layouts)
        free_layouts.extend(layouts)

    # load data collection
    dc = ONLINE_EXCHANGE.future.data.get_data_collection(symbols=[args.symbol], time_frames=[args.time_frame])

    # create process generator to avoid from too many open files issue
    process_generator = create_process(modules=modules, dc=dc)

    # create terminal layout to track processes
    process_to_args = {}
    process_to_layout = {}
    running_process_set = set()
    done_process_set = set()
    with Live(main_layout, refresh_per_second=2):
        while len(done_process_set) < len(modules):
            # start process if there is free worker (processor)
            if len(running_process_set) < args.max_workers:
                process, (module, _, queue) = next(process_generator)
                process_to_args[process] = (module, queue)
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

    # load config
    with open(args.config_path, "r") as file:
        config_dict = json.load(file)

    # get symbols and time_frames fom config_dict
    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # define feature and label transforms
    feature_transform_types = [
        RollingMeanStdTrRocFeatureTransform,
        StftTrRocFeatureTransform
    ]

    label_transform_types = [
        ForwardBackwardMinimumLabelTransform,
        ForwardIchimokuLabelTransform,
        ForwardMiddleSmaLabelTransform,
        ForwardRocLabelTransform,
        NextFractalLabelTransform,
        UpDownLabelTransform
    ]

    # define model_types
    model_types = [
        FeedForward,
        GRU,
        Hybrid,
        LSTM,
        ResNet1D,
        Transformer
    ]

    # define model params
    model_args = {
        "dim_sequence": args.dim_sequence,
        "dim_feature": 4,
        "dim_output": 2,
        "num_hidden_layers": args.num_hidden_layers,
        "batch_norm": args.no_batch_norm,
        "dropout": args.dropout,
        "activation_fn": nn.Softmax(dim=-1)
    }

    # define modules
    modules = []
    items = itertools.product(symbols, time_frames, feature_transform_types, label_transform_types, model_types)
    for symbol, time_frame, ft_type, lt_type, model_type in items:
        # define feature and label transforms
        transform_param = {"symbol": symbol, "time_frame": time_frame, "dim_sequence": args.dim_sequence}
        feature_transform = call_with_dict(ft_type, transform_param)
        label_transform = call_with_dict(lt_type, transform_param)

        # define dataset
        dataset = Dataset(feature_transform=feature_transform, label_transform=label_transform, interval=args.interval)

        # define module
        module = Module(dataset=dataset, model_type=model_type, model_kwargs=model_args)
        modules.append(module)

    run_multi_process(args=args, modules=modules)


if __name__ == '__main__':
    main()
