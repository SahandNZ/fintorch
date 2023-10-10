import numpy as np
import torch
from tqdm.auto import tqdm

from fintorch.dataset.data import Data
from fintorch.dataset.dataset import Dataset
from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform


class BsatfDataset(Dataset):
    def __init__(self, samples_count: int, feature_transform: FeatureTransform, label_transform: LabelTransform,
                 show_progress_bar: bool = False):
        super().__init__(samples_count=samples_count, feature_transform=feature_transform,
                         label_transform=label_transform, show_progress_bar=show_progress_bar)

    def prepare(self, data: Data):
        # create dataframes
        feature_data = self.feature_transform.fit(data)
        label_dataframe = self.label_transform.fit(data)

        # create iteration range
        bar = label_dataframe.index.to_list()[-self.sample_count:]
        if self.show_progress_bar:
            bar = tqdm(list(bar))
            bar.set_description_str(f"Creating ({self.short_name}) BSATF dataset")

        # create samples
        x, y = [], []
        start_timestamp, stop_timestamp = None, None
        for timestamp in bar:
            feature = self.feature_transform.transform(data=feature_data, timestamp=timestamp)
            label = self.label_transform.transform(df=label_dataframe, timestamp=timestamp)

            if feature is not None and label is not None:
                if start_timestamp is None or timestamp < start_timestamp:
                    start_timestamp = timestamp
                if stop_timestamp is None or stop_timestamp < timestamp:
                    stop_timestamp = timestamp

                x.append(feature)
                y.append(label)

        # convert x and y to tensor and slice dataframe
        x = torch.from_numpy(np.array(x)).float()
        y = torch.from_numpy(np.array(y)).float()
        df = label_dataframe[(start_timestamp <= label_dataframe.index.to_series()) &
                             (label_dataframe.index.to_series() <= stop_timestamp)]

        # set x, y, and df properties
        self.preset(x=x, y=y, df=df)

    def preprocess(self, data: Data, timestamp: int) -> torch.Tensor:
        feature_data = self.feature_transform.fit(data=data)
        feature = self.feature_transform.transform(data=feature_data, timestamp=timestamp)
        x = torch.from_numpy(feature).unsqueeze(0).float()

        return x
