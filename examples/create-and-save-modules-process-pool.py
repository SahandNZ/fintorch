import argparse
import itertools
import json
import os.path
from concurrent.futures import ProcessPoolExecutor
from typing import Type

import torch
from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local import load_dataframes_dict
from torch import nn
from tqdm.rich import tqdm

from fintorch.criterion.ce import CELoss
from fintorch.cross_validation.sliding_window import SlidingWindowCrossValidation
from fintorch.data import Data
from fintorch.data_loader.data_loader import DataLoader
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.lr_scheduler import LRScheduler
from fintorch.model.feed_forward.feed_forward import FeedForward
from fintorch.model.gru.gru import GRU
from fintorch.model.hybrid.hybrid import Hybrid
from fintorch.model.lstm.lstm import LSTM
from fintorch.model.model import Model
from fintorch.model.resnet1d.residual1d import ResidualBlock1D
from fintorch.model.resnet1d.resnet1d import ResNet1D
from fintorch.model.transformer.transformer import Transformer
from fintorch.module import Module
from fintorch.optimizer import Optimizer
from fintorch.trainer import Trainer
from fintorch.transform.feature.rmstd_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.feature.stft_tr_roc import StftTrRocFeatureTransform
from fintorch.transform.label.classification.trend.forward_backward_min import ForwardBackwardMinimumLabelTransform
from fintorch.transform.label.classification.trend.forward_ichimoku import ForwardIchimokuLabelTransform
from fintorch.transform.label.classification.trend.forward_middle_sma import ForwardMiddleSmaLabelTransform
from fintorch.transform.label.classification.trend.forward_roc import ForwardRocLabelTransform
from fintorch.transform.label.classification.trend.next_fractal import NextFractalLabelTransform
from fintorch.transform.label.classification.trend.up_down import UpDownLabelTransform
from fintorch.utils.function import call_with_dict

exchange: str = None
show_progress_bar: bool = None
print_classification_logs: bool = None


def work(
        symbol: str,
        time_frame: TimeFrame,
        feature_transform_cls: Type,
        label_transform_cls: Type,
        model: Model,
):
    # load candlestick date
    symbols = [symbol]
    time_frames = [time_frame]
    df_dict = load_dataframes_dict(exchange=exchange, symbols=symbols, time_frames=time_frames, update=False,
                                   show_progress_bar=False)
    data = Data(df_dict)

    # create dataset
    feature_transform = feature_transform_cls(symbols=symbols, time_frames=time_frames)
    label_transform = label_transform_cls(symbol=symbol, time_frame=time_frame)
    dataset = BsfDataset(feature_transform=feature_transform, label_transform=label_transform)

    # create trainer
    cross_validation = SlidingWindowCrossValidation(window_size=5000, train_percentage=0.8, dev_percentage=0.1)
    data_loader = DataLoader(batch_size=2 ** 10)
    criterion = CELoss()
    optimizer = Optimizer(cls=torch.optim.Adam, lr=1e-3, weight_decay=5e-3)
    scheduler = LRScheduler(cls=torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.9)
    trainer = Trainer(
        epochs=50,
        cross_validation=cross_validation,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        print_logs=print_classification_logs,
        show_progress_bar=show_progress_bar,
        print_memory_status_logs=False,
        show_learning_curve_plot=False,
        print_classification_logs=print_classification_logs,
    )

    # create and save trained module
    module = Module(trainer=trainer, dataset=dataset, model=model)
    print("{}{:^12}-{:^8}-{:^32}{}".format("*" * 32, symbol, time_frame, module.short_name, "*" * 32))
    if not os.path.exists(module.path(mode="experiment")):
        dataset.prepare(data=data, show_progress_bar=show_progress_bar)
        module.optimize()
        module.save(mode="experiment")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", action="store_true", required=False)
    parser.add_argument("--show-progress-bar", action="store_true", required=False)
    parser.add_argument("--print-classification-logs", action="store_true", required=False)
    parser.add_argument("--max-workers", action="store", type=int, required=False, default=32)
    parser.add_argument("--exchange", action="store", type=str, required=False, default="binance")
    parser.add_argument("--modules-path", action="store", type=str, required=False, default="modules.json")
    args = parser.parse_args()

    # set global variables
    global exchange, show_progress_bar, print_classification_logs
    exchange = args.exchange
    show_progress_bar = args.show_progress_bar
    print_classification_logs = args.print_classification_logs

    # load symbols and time_frames
    with open(args.modules_path, "r") as file:
        config_dict = json.load(file)

    symbols = config_dict["symbols"]
    time_frames = config_dict["time-frames"]

    # create feature transforms
    feature_transforms_cls = [
        RollingMeanStdTrRocFeatureTransform,
        StftTrRocFeatureTransform
    ]

    label_transforms_cls = [
        UpDownLabelTransform,
        ForwardRocLabelTransform,
        ForwardIchimokuLabelTransform,
        ForwardMiddleSmaLabelTransform,
        ForwardBackwardMinimumLabelTransform,
        NextFractalLabelTransform,
    ]

    # create models
    dim_sequence = 32
    dim_feature = 4
    dim_input = dim_sequence * dim_feature
    dim_output = 2

    model_params_dict = {
        'num_head': 2,
        'dropout': 0.5,
        'num_layer': 2,
        'hidden_size': 16,
        'block': ResidualBlock1D,
        'activation_fn': nn.Softmax(dim=-1),
        'dim_sequence': dim_sequence,
        'dim_feature': dim_feature,
        'dim_input': dim_input,
        'dim_output': dim_output,
        'layers': [dim_input, dim_input // 2, dim_input // 4, dim_output]
    }

    models = []
    model_classes = [GRU, LSTM, Hybrid, ResNet1D, Transformer, FeedForward]
    for model_cls in model_classes:
        model = call_with_dict(model_cls, model_params_dict)
        models.append(model)

    # run jobs
    items = list(itertools.product(symbols, time_frames, feature_transforms_cls, label_transforms_cls, models))
    bar = tqdm(items, desc="Create and saving modules")

    if args.sequential:
        for symbol, time_frame, ft_cls, lt_cls, model in bar:
            work(symbol, time_frame, ft_cls, lt_cls, model)
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for symbol, time_frame, ft_cls, lt_cls, model in items:
                future = executor.submit(work, exchange, symbol, time_frame, ft_cls, lt_cls, model, show_progress_bar)
                futures.append(future)

            for future in futures:
                future.result()
                bar.update(1)


if __name__ == '__main__':
    main()
