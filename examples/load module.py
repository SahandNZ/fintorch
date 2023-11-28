import os

from pyccx.data.local import load_dataframes_dict

from fintorch.data import Data
from fintorch.module import Module


def main():
    symbol = "BTC-USDT"
    time_frame = 900

    # define path
    data_root = os.environ.get("DATA_ROOT", "./data")
    module_root = os.path.join(data_root, "module/experiment", symbol, str(time_frame))
    modules_file_name = os.listdir(module_root)
    module_path = os.path.join(module_root, modules_file_name[0])

    # load module
    module = Module.load(module_path)

    # save module in deployment mode
    module.save(mode="deployment")

    # print overall fold classification logs
    module.overall_fold.print_classification_logs()

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
