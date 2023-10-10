import itertools

import numpy as np
import torch
from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local_data import load_dataframe
from torch import nn

from fintorch.criterion.bce_loss import BCELoss
from fintorch.cross_validation.cross_validation import CrossValidation
from fintorch.data_loader.data_loader import DataLoader
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.dataset.data import Data
from fintorch.model.feed_forward.feed_forward import FeedForward
from fintorch.trainer import Trainer
from fintorch.transform.feature.rms_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.label.classification.trend.fmsma import FMsmaLabelTransform


def print_metrics_logs(fold):
    for name, best_dataset_metrics in zip(["Dev set", "Test set"], [fold.best_dev_metrics, fold.best_test_metrics]):
        print(name)
        print("\t{:<20}{}".format("MAE", best_dataset_metrics.mae_loss))
        print("\t{:<20}{}".format("MSE", best_dataset_metrics.mse_loss))
        print("\t{:<20}{}".format("Loss", best_dataset_metrics.objective))
        print("\t{:<20}{}\n".format("Accuracy", best_dataset_metrics.accuracy))
        print("\t{:<20}{:<20}{:<20}{:<20}".format("Label \ Measure", "Precision", "Recall", "F1-score"))
        for label in range(fold.dev_set.label_transform.num_classes):
            p = best_dataset_metrics.precision(label=label)
            r = best_dataset_metrics.recall(label=label)
            f1 = best_dataset_metrics.f1(label=label)
            print("\t{:<20}{:<20}{:<20}{:<20}".format(label, p, r, f1))
        print()


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
    model = FeedForward(layers=[input_dim, output_dim], dropout=0.5, activation_fn=nn.Softmax())

    # prepare trainer
    cross_validation = CrossValidation(train_percentage=0.8, dev_percentage=0.1)
    data_loader = DataLoader(batch_size=2 ** 8)
    criterion = BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # prepare trainer
    trainer = Trainer(
        epochs=50,
        cross_validation=cross_validation,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        print_logs=True,
        show_progress_bar=False,
        show_learning_curve_plot=True
    )

    folds = trainer.optimize(dataset=dataset, model=model)
    print_metrics_logs(folds[-1])


if __name__ == '__main__':
    main()
