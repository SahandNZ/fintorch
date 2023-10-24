import torch
from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local import load_dataframes_dict
from torch import nn

from fintorch.criterion.bce_loss import BCELoss
from fintorch.cross_validation.cross_validation import CrossValidation
from fintorch.data import Data
from fintorch.data_loader.data_loader import DataLoader
from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.model.feed_forward.feed_forward import FeedForward
from fintorch.trainer import Trainer
from fintorch.transform.feature.rms_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.label.classification.trend.forward_msma import ForwardMsmaLabelTransform


def main():
    symbols = ['BTC-USDT']
    time_frames = [TimeFrame.MIN5, TimeFrame.MIN15]

    # preparing data
    df_dict = load_dataframes_dict(exchange='binance', symbols=symbols, time_frames=time_frames)
    data = Data(df_dict)

    # creating dataset
    feature_transform = RollingMeanStdTrRocFeatureTransform(symbols=symbols, time_frames=time_frames)
    label_transform = ForwardMsmaLabelTransform(symbol=symbols[0], time_frame=time_frames[0])
    dataset = BsfDataset(samples_count=10000, feature_transform=feature_transform, label_transform=label_transform,
                         show_progress_bar=True)
    dataset.prepare(data=data)

    # prepare model
    input_dim = len(dataset.feature_transform.symbols) * \
                len(dataset.feature_transform.time_frames) * \
                dataset.feature_transform.sequence_length * \
                len(dataset.feature_transform.features)
    output_dim = dataset.label_transform.num_classes
    model = FeedForward(layers=[input_dim, output_dim], dropout=0.5, activation_fn=nn.Softmax(dim=-1))

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
        show_progress_bar=True,
        show_learning_curve_plot=True,
        print_classification_logs=True
    )

    trainer.optimize(dataset=dataset, model=model)


if __name__ == '__main__':
    main()
