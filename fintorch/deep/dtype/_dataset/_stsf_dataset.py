from typing import List, Union

import torch
from rich.progress import Progress

from ._dataset import Dataset
from .._sample import Sample
from ...transform.feature import FeatureTransform
from ...transform.label import LabelTransform
from ....dtype import Data
from ....enum import TimeFrame


class StsfDataset(Dataset):
    def __init__(self, start_date: str, stop_date: str, interval: TimeFrame, symbols: List[str],
                 time_frames: List[TimeFrame], sequence_length: int, feature_transform: FeatureTransform,
                 label_transform: LabelTransform):
        super().__init__(start_date, stop_date, interval, sequence_length, feature_transform, label_transform)
        self.__symbols: List[str] = symbols
        self.__time_frames: List[TimeFrame] = time_frames

    @property
    def symbols(self) -> List[str]:
        return self.__symbols

    @property
    def time_frames(self) -> List[TimeFrame]:
        return self.__time_frames

    def prepare(self, data: Data, progress: Progress = None) -> None:
        # create new task in rich progress bar for creating samples
        if progress is not None:
            task = progress.add_task(description="[green]Creating Dataset samples", total=len(self.timestamps))

        # create samples
        for timestamp in self.timestamps:
            self.feature_transform.load_or_transform_stsf(data, timestamp, self.symbols, self.time_frames)
            self.label_transform.load_or_transform_stsf(data, timestamp, self.symbols, self.time_frames)

            if progress is not None:
                progress.update(task, advance=1)

    def preprocess(self, data: Data, timestamps: List[int]) -> torch.Tensor:
        x = []
        for timestamp in timestamps:
            feature = self.feature_transform.load_or_transform_stsf(data, timestamp, self.symbols, self.time_frames)
            feature = torch.from_numpy(feature).unsqueeze(0)
            x.append(feature)

        x = torch.cat(x, dim=0).float()
        return x

    def _load_samples(self, timestamps: List[int]) -> List[Sample]:
        samples = []
        for timestamp in timestamps:
            feature = self.feature_transform.load_stsf(timestamp, self.symbols, self.time_frames)
            label = self.label_transform.load_stsf(timestamp, self.symbols, self.time_frames)
            sample = Sample(timestamp=timestamp, feature=feature, label=label)
            samples.append(sample)

        return samples
