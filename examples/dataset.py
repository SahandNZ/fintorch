import itertools

from pyccx.constant.time_frame import TimeFrame
from pyccx.data.local_data import LocalData, load_dataframe
from pyccx.interface.exchange import Exchange

from fintorch.dataset.bsf_dataset import BsfDataset
from fintorch.dataset.data import Data
from fintorch.transform.feature.rms_tr_roc import RollingMeanStdTrRocFeatureTransform
from fintorch.transform.label.fmsma import FMsmaLabelTransform


def main():
    symbols = ['BTC-USDT', 'ETH-USDT']
    time_frames = [TimeFrame.MIN5, TimeFrame.HOUR1, TimeFrame.HOUR4]

    exchange = Exchange(exchange='binance')
    local_data = LocalData(exchange=exchange, candles_count=100000)
    for symbol in symbols:
        local_data.download_candles(symbol=symbol, time_frame=TimeFrame.MIN1)

    # preparing data
    data = Data()
    for symbol, time_frame in itertools.product(symbols, time_frames):
        df = load_dataframe(exchange='binance', symbol=symbol, time_frame=time_frame)
        data[(symbol, time_frame)] = df

    # creating dataset
    feature_transform = RollingMeanStdTrRocFeatureTransform(symbols=symbols, time_frames=time_frames)
    label_transform = FMsmaLabelTransform(symbol=symbols[0], time_frame=time_frames[1])
    dataset = BsfDataset(samples_count=1000, feature_transform=feature_transform, label_transform=label_transform,
                         show_progress_bar=True)
    dataset.prepare(data=data)

    a = 0


if __name__ == '__main__':
    main()
