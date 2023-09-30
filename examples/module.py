import argparse
import pickle

import torch
from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local_data import LocalData
from pyccx.interface.exchange import Exchange
from pyccx.model.candle import Candle
from torch import nn

from fintorch.criterion.bce_loss import BCELoss
from fintorch.cross_validation.cross_validation import CrossValidation
from fintorch.data_loader.data_loader import DataLoader
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.model.feed_forward.feed_forward import FeedForward
from fintorch.module import Module
from fintorch.trainer import Trainer
from fintorch.transform.feature.mean_std_tr_roc import MeanStdTrRocFeatureTransform
from fintorch.transform.label.fmsma import FMsmaLabelTransform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', action='store', type=str, required=False, default='BTC-USDT')
    parser.add_argument('--time-frame', action='store', type=int, required=False, default=TimeFrame.HOUR1)
    args = parser.parse_args()

    # prepare dataframe
    exchange = Exchange(exchange='binance')
    local_data = LocalData(exchange=exchange)
    candles = local_data.get_candles(symbol=args.symbol, time_frame=args.time_frame)
    df = Candle.to_data_frame(candles)[-10000:]

    # prepare dataset
    feature_transform = MeanStdTrRocFeatureTransform(look_back=4, sequence_length=32)
    label_transform = FMsmaLabelTransform()
    dataset = BsfDataset(feature_transform=feature_transform, label_transform=label_transform, show_progress_bar=True)

    # prepare model
    model = FeedForward(layers=[128, 3], dropout=0.5, activation_fn=nn.Softmax(dim=1))

    # prepare trainer
    cross_validation = CrossValidation(train_percentage=0.8, dev_percentage=0.1)
    data_loader = DataLoader(batch_size=2 ** 8)
    criterion = BCELoss()
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
        show_learning_curve_plot=False
    )

    # prepare module
    module = Module(trainer=trainer, dataset=dataset, model=model)
    module.optimize(df=df)

    # store modules
    modules = [module]
    with open('./data/pre trained/modules.pkl', 'wb+') as file:
        pickle.dump(modules, file)


if __name__ == '__main__':
    main()
