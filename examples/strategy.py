import torch
from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local import load_dataframes_dict
from torch import nn

from fintorch.criterion.ce import CELoss
from fintorch.cross_validation.sliding_window import SlidingWindowCrossValidation
from fintorch.data import Data
from fintorch.data_loader.data_loader import DataLoader
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.measures import Measures
from fintorch.model.lstm.lstm import LSTM
from fintorch.module import Module
from fintorch.position import Position
from fintorch.strategy.trend import TrendStrategy
from fintorch.trainer import Trainer
from fintorch.transform.feature.rms_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.label.classification.trend.forward_msma import ForwardMsmaLabelTransform


def main():
    symbols = ['BTC-USDT']
    time_frames = [TimeFrame.MIN1]

    # preparing data
    df_dict = load_dataframes_dict(exchange='binance', symbols=symbols, time_frames=time_frames, update=True)
    data = Data(df_dict)

    # creating dataset
    feature_transform = RollingMeanStdTrRocFeatureTransform(symbols=symbols, time_frames=time_frames)
    label_transform = ForwardMsmaLabelTransform(symbol=symbols[0], time_frame=time_frames[0])
    dataset = BsfDataset(samples_count=6000, feature_transform=feature_transform, label_transform=label_transform,
                         show_progress_bar=True)

    # prepare model
    dim_sequence = dataset.feature_transform.sequence_length
    dim_feature = len(dataset.feature_transform.symbols) * len(dataset.feature_transform.time_frames) * \
                  len(dataset.feature_transform.features)
    dim_output = dataset.label_transform.num_classes
    model = LSTM(dim_sequence=dim_sequence, dim_feature=dim_feature, hidden_size=8, num_layer=2, dropout=0.5,
                 dim_output=dim_output, activation_fn=nn.Softmax(dim=-1))

    # prepare trainer
    cross_validation = SlidingWindowCrossValidation(window_size=5000, train_percentage=0.8, dev_percentage=0.1)
    data_loader = DataLoader(batch_size=2 ** 10)
    criterion = CELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    trainer = Trainer(
        epochs=20,
        cross_validation=cross_validation,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        print_logs=True,
        show_progress_bar=True,
        show_learning_curve_plot=False,
        print_classification_logs=True,
    )

    # prepare module
    module = Module(trainer=trainer, dataset=dataset, model=model)
    module.optimize(data=data)

    # calculating trade measures
    strategy = TrendStrategy()
    positions = strategy.backtest(module.overall_fold)
    Position.tabulate(positions)

    measures = Measures(positions, quantities=0.01)
    measures.tabulate()


if __name__ == '__main__':
    main()
