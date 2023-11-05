import pickle

import torch
from matplotlib import pyplot as plt
from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local import load_dataframes_dict
from torch import nn

from fintorch.criterion.bce_loss import BCELoss
from fintorch.cross_validation.sliding_window import SlidingWindowCrossValidation
from fintorch.data import Data
from fintorch.data_loader.data_loader import DataLoader
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.model.feed_forward.feed_forward import FeedForward
from fintorch.module import Module
from fintorch.trainer import Trainer
from fintorch.transform.feature.rms_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.label.classification.trend.forward_msma import ForwardMsmaLabelTransform


def main():
    symbols = ['BTC-USDT']
    time_frames = [TimeFrame.MIN5]

    # preparing data
    df_dict = load_dataframes_dict(exchange='binance', symbols=symbols, time_frames=time_frames, update=True)
    data = Data(df_dict)

    # creating dataset
    feature_transform = RollingMeanStdTrRocFeatureTransform(symbols=symbols, time_frames=time_frames)
    label_transform = ForwardMsmaLabelTransform(symbol=symbols[0], time_frame=time_frames[0])
    dataset = BsfDataset(samples_count=6000, feature_transform=feature_transform, label_transform=label_transform,
                         show_progress_bar=True)

    # prepare model
    input_dim = len(dataset.feature_transform.symbols) * \
                len(dataset.feature_transform.time_frames) * \
                dataset.feature_transform.sequence_length * \
                len(dataset.feature_transform.features)
    output_dim = dataset.label_transform.num_classes
    model = FeedForward(layers=[input_dim, output_dim], dropout=0.5, activation_fn=nn.Softmax(dim=-1))

    # prepare trainer
    cross_validation = SlidingWindowCrossValidation(window_size=5000, train_percentage=0.8, dev_percentage=0.1)
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
        show_learning_curve_plot=False,
        print_classification_logs=True,
    )

    # prepare module
    module = Module(trainer=trainer, dataset=dataset, model=model)
    module.optimize(data=data)

    # store modules
    modules = [module]
    with open(f'./data/pre-trained-dlm/data.pkl', 'wb+') as file:
        pickle.dump(modules, file)

    # make prediction
    df_dict = load_dataframes_dict(exchange='binance', symbols=symbols, time_frames=time_frames, update=True)
    data = Data(df_dict)

    df = data[module.dataset.label_transform.symbol, module.dataset.label_transform.time_frame]
    timestamps = df.index.to_list()[-500:]
    y_hat = module.predict(data=data, timestamps=timestamps, show_progress_bar=True)
    prediction = y_hat.argmax(-1)
    print(y_hat, y_hat.shape)

    # draw ohlcv plot
    pdf = df[timestamps[0] <= df.index.to_series()]
    label_transform.draw_ohlcv_plot(df=pdf, prediction=prediction)
    plt.show()


if __name__ == '__main__':
    main()
