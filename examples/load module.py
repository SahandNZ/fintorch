from pyccx.data.local import load_dataframes_dict

from fintorch.data import Data
from fintorch.utils.unpickler import Unpickler


def main():
    # load modules
    with open(f'./data/pre-trained-dlm/data.pkl', "rb") as file:
        unpickler = Unpickler(file)
        module = unpickler.load()[0]

    # make prediction
    symbols = module.dataset.feature_transform.symbols
    time_frames = module.dataset.feature_transform.time_frames

    df_dict = load_dataframes_dict(exchange='binance', symbols=symbols, time_frames=time_frames, update=True)
    data = Data(df_dict)

    df = data[module.dataset.label_transform.symbol, module.dataset.label_transform.time_frame]
    timestamps = df.index.to_list()[-500:]
    y_hat = module.predict(data=data, timestamps=timestamps, show_progress_bar=True)
    print(y_hat, y_hat.shape)


if __name__ == '__main__':
    main()
