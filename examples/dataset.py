import argparse

from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local_data import LocalData
from pyccx.interface.exchange import Exchange
from pyccx.model.candle import Candle

from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.transform.feature.mean_std_tr_roc import MeanStdTrRocFeatureTransform
from fintorch.transform.label.fmsma import FMsmaLabelTransform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', action='store', type=str, required=False, default='BTC-USDT')
    parser.add_argument('--time-frame', action='store', type=int, required=False, default=TimeFrame.HOUR1)
    args = parser.parse_args()

    exchange = Exchange(exchange='binance')
    local_data = LocalData(exchange=exchange)
    candles = local_data.get_candles(symbol=args.symbol, time_frame=args.time_frame)
    df = Candle.to_data_frame(candles)[-1000:]

    feature_transform = MeanStdTrRocFeatureTransform(look_back=4, sequence_length=32)
    label_transform = FMsmaLabelTransform()
    dataset = BsfDataset(feature_transform=feature_transform, label_transform=label_transform)
    dataset.prepare(df=df)


if __name__ == '__main__':
    main()
