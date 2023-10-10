import pickle

import numpy as np
import torch
from fintorch.transform.label.fmsma import FMsmaLabelTransform
from pyccx.constant.time_frame import TimeFrame
from torch import nn

from fintorch.criterion.bce_loss import BCELoss
from fintorch.cross_validation.cross_validation import CrossValidation
from fintorch.data_loader.data_loader import DataLoader
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.model.feed_forward.feed_forward import FeedForward
from fintorch.module import Module
from fintorch.trainer import Trainer
from fintorch.transform.feature.rms_tr_roc import RollingMeanStdTrRocFeatureTransform


def main():
    symbols = ['BTC-USDT']
    time_frames = [TimeFrame.MIN5, TimeFrame.MIN15]

    # preparing data
    data = Data()
    for symbol, time_frame in itertools.product(symbols, time_frames):
        df = load_dataframe(exchange='binance', symbol=symbol, time_frame=time_frame)
        data[(symbol, time_frame)] = df

    # creating dataset
    feature_transform = RollingMeanStdTrRocFeatureTransform(symbols=symbols, time_frames=time_frames)
    label_transform = FMsmaLabelTransform(symbol=symbols[0], time_frame=time_frames[0])
    dataset = BsfDataset(samples_count=10000, feature_transform=feature_transform, label_transform=label_transform,
                         show_progress_bar=True)
    dataset.prepare(data=data)

    # prepare model
    input_dim = np.array(list(dataset.x.shape)[1:]).prod()
    output_dim = dataset.label_transform.num_classes
    model = FeedForward(layers=[input_dim, output_dim], dropout=0.5, activation_fn=nn.Softmax(dim=-1))

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
