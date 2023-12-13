import itertools

import torch
from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local import load_dataframes_dict
from torch import nn
from tqdm import tqdm

from fintorch.criterion.ce import CELoss
from fintorch.cross_validation.sliding_window import SlidingWindowCrossValidation
from fintorch.data import Data
from fintorch.data_loader.data_loader import DataLoader
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.lr_scheduler import LRScheduler
from fintorch.model.feed_forward.feed_forward import FeedForward
from fintorch.model.resnet1d.residual1d import ResidualBlock1D
from fintorch.model.resnet1d.resnet1d import ResNet1D
from fintorch.module import Module
from fintorch.optimizer import Optimizer
from fintorch.trainer import Trainer
from fintorch.transform.feature.rmstd_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.label.classification.trend.up_down import UpDownLabelTransform
from fintorch.utils.function import call_with_dict


def main():
    exchange = 'binance'
    symbol = 'BTC-USDT'
    label_time_frame = TimeFrame.DAY1
    feature_time_frame = TimeFrame.HOUR1
    time_frames = [label_time_frame, feature_time_frame]

    # preparing data
    df_dict = load_dataframes_dict(exchange=exchange, symbols=[symbol], time_frames=time_frames, update=False)
    data = Data(df_dict)

    # create datasets
    feature_transforms = [RollingMeanStdTrRocFeatureTransform(symbols=[symbol], time_frames=[feature_time_frame])]
    label_transforms = [UpDownLabelTransform(symbol=symbol, time_frame=label_time_frame)]

    datasets = []
    for feature_transform, label_transform in itertools.product(feature_transforms, label_transforms):
        dataset = BsfDataset(feature_transform=feature_transform, label_transform=label_transform)
        dataset.prepare(data=data, samples_count=1000, show_progress_bar=True)
        datasets.append(dataset)
        print(dataset.short_name)

    # create models
    dim_sequence = dataset.feature_transform.sequence_length
    dim_feature = len(dataset.feature_transform.symbols) * \
                  len(dataset.feature_transform.time_frames) * \
                  len(dataset.feature_transform.features)
    dim_input = dim_sequence * dim_feature
    dim_output = dataset.label_transform.num_classes

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
        'layers': [dim_input, dim_output]
    }

    models = []
    model_classes = [FeedForward]
    for model_cls in model_classes:
        model = call_with_dict(model_cls, model_params_dict)
        models.append(model)

    # prepare trainer
    cross_validation = SlidingWindowCrossValidation(window_size=1000, train_percentage=0.8, dev_percentage=0.1)
    data_loader = DataLoader(batch_size=2 ** 8)
    criterion = CELoss()
    optimizer = Optimizer(cls=torch.optim.Adam, lr=1e-3, weight_decay=1e-3)
    scheduler = LRScheduler(cls=torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.9)
    trainer = Trainer(
        epochs=1,
        cross_validation=cross_validation,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        print_logs=True,
        show_progress_bar=False,
        print_memory_status_logs=False,
        show_learning_curve_plot=False,
        print_classification_logs=True,
    )

    # create modules
    bar = list(itertools.product(datasets, models))[:2]
    bar = tqdm(bar, desc="Creating and saving modules")
    for dataset, model in bar:
        symbol = dataset.label_transform.symbol
        time_frame = dataset.label_transform.time_frame
        module = Module(trainer=trainer, dataset=dataset, model=model)
        print("{}{:^12}-{:^8}-{:^32}{}".format("*" * 32, symbol, time_frame, module.short_name, "*" * 32))

        if dataset.need_preparation:
            dataset.prepare(data=data, show_progress_bar=False)
        module.optimize()
        module.save(mode="experiment")

        module.overall_fold.print_classification_logs()


if __name__ == '__main__':
    main()
