import os
import math
from datetime import datetime

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

os.environ.setdefault("FINTORCH_BASE_TIME_FRAME", "60")

from fintorch.app import Application, Context
from fintorch.utils.args import DefaultArgumentParser, DefaultNamespace
from fintorch.utils.preprocess import remove_price_dependency

def callback(context: Context, args: DefaultNamespace):
    LOOK_AHEAD = 3
    SEQENCE_LENGTH = 64
    
    TEST_LENGTH = LOOK_AHEAD
    TRAIN_LENGTH = 10
    WINDOW_LENGTH = TRAIN_LENGTH + TEST_LENGTH
    
    # get candles df
    sdf = context.online_data.get_candles_dataframe(symbol=args.symbol, time_frame=args.time_frame).copy()
    
    # add label
    sdf["sma"] = sdf.close.rolling(window=LOOK_AHEAD, center=True).mean()
    sdf["fsma"] = sdf.sma.shift(periods=-LOOK_AHEAD // 2)
    sdf["label"] = np.where(np.isnan(sdf.sma) | np.isnan(sdf.fsma), np.nan, sdf.sma < sdf.fsma)
    
    # add features
    df = remove_price_dependency(df=sdf)
    features = ["mean-close", "std-close"]
    df[f"mean-close"] = df.close.rolling(window=5).mean()
    df[f"std-close"] = df.close.rolling(window=5).std()
        
    # create samples
    x, y = [], []
    for index in range(len(df) - WINDOW_LENGTH, len(df)):
        fdf = df[features].iloc[index - SEQENCE_LENGTH: index]
        ndf = (fdf - fdf.min()) / (fdf.max() - fdf.min())
        label = sdf.label.iloc[index]

        x.append(ndf.to_numpy())
        y.append(label)
    
    # convert to np.array
    x = np.array(x).reshape(-1, SEQENCE_LENGTH * len(features))
    y = np.array(y)
    # print(x.shape, y.shape)
    
    # split train and test x, y
    train_x, train_y = x[:TRAIN_LENGTH], y[:TRAIN_LENGTH]
    test_x, test_y = x[TRAIN_LENGTH:], y[TRAIN_LENGTH:]
    # print(train_y.shape, train_y.shape)
    # print(test_x.shape, test_y.shape)
    
    # fit and transform polynomial transformer
    transformer = PolynomialFeatures(degree=2, include_bias=True)
    train_x_hat = transformer.fit_transform(train_x)

    model = LinearRegression(fit_intercept=False).fit(train_x_hat, train_y)
    
    test_x_hat = transformer.transform(test_x)
    test_y_hat = np.round(model.predict(test_x_hat), 2)
    test_prediction = 0.5 < test_y_hat
    
    print(datetime.fromtimestamp(sdf.index[-1]))
    print(datetime.fromtimestamp(df.index[-1]))
    print("Train y:          {}".format(train_y))
    print("Test y:           {}".format(test_y))
    print("Test y hat:       {}".format(test_y_hat))
    print("Test prediction: ", test_prediction)
    print("=" * 64)
    print()


def main():
    string_args = [
        "--stf-config", "btc-1m",
        "--symbol", "BTC-USDT",
        "--time-frame", "60"
    ]
    args = DefaultArgumentParser.parse(args=string_args)
    
    app = Application.from_dict(dct=args.app_kwargs)
    app.job_queue.run_once(
        callback=callback,
        kwargs={"args": args},
        misfire_grace_time=10
    )
    app.job_queue.run_repeating(
        callback=callback,
        kwargs={"args": args},
        interval=args.time_frame,
        when='open',
        misfire_grace_time=10
    )
    app.start()


if __name__ == '__main__':
    main()
