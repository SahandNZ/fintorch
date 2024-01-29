from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from rich.progress import Progress

from fintorch.deep.dtype._sample import Sample
from fintorch.deep.transform.feature import FeatureTransform
from fintorch.deep.transform.label import LabelTransform
from fintorch.dtype import Data
from fintorch.enum import TimeFrame
from fintorch.utils.timestamp import create_timestamps


class Dataset:
    def __init__(self, start_date: str, stop_date: str, sampling_time_frame: TimeFrame,
                 symbols: List[str], time_frames: List[TimeFrame], sequence_length: int,
                 feature_transform: FeatureTransform, label_transform: LabelTransform):
        self.__start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.__stop_date = datetime.strptime(stop_date, "%Y-%m-%d")
        self.__sampling_time_frame: TimeFrame = sampling_time_frame
        self.__timestamps: List[int] = create_timestamps(self.start_date, self.stop_date, self.sampling_time_frame)

        self.__symbols: List[str] = symbols
        self.__time_frames: List[TimeFrame] = time_frames
        self.__sequence_length: int = sequence_length
        self.__feature_transform: FeatureTransform = feature_transform
        self.__label_transform: LabelTransform = label_transform

    @property
    def start_date(self) -> datetime:
        return self.__start_date

    @property
    def stop_date(self) -> datetime:
        return self.__stop_date

    @property
    def sampling_time_frame(self) -> TimeFrame:
        return self.__sampling_time_frame

    @property
    def timestamps(self) -> List[int]:
        return self.__timestamps

    @property
    def symbols(self) -> List[str]:
        return self.__symbols

    @property
    def time_frames(self) -> List[TimeFrame]:
        return self.__time_frames

    @property
    def sequence_length(self) -> int:
        return self.__sequence_length

    @property
    def feature_transform(self) -> FeatureTransform:
        return self.__feature_transform

    @property
    def label_transform(self) -> LabelTransform:
        return self.__label_transform

    def prepare(self, data: Data, progress: Progress = None) -> None:
        # create new task in rich progress bar for creating samples
        if progress is not None:
            task = progress.add_task(description="[green]Creating Dataset samples", total=len(self.timestamps))

        # create samples
        for timestamp in self.timestamps:
            self.feature_transform.load_or_transform(data, timestamp, self.symbols, self.time_frames)
            self.label_transform.load_or_transform(data, timestamp, self.symbols, self.time_frames)

            if progress is not None:
                progress.update(task, advance=1)

    def preprocess(self, data: Data, timestamps: List[int]) -> torch.Tensor:
        x = []
        for timestamp in timestamps:
            feature = self.feature_transform.load_or_transform(data, timestamp, self.symbols, self.time_frames)
            feature = torch.from_numpy(feature).unsqueeze(0)
            x.append(feature)

        x = torch.cat(x, dim=0).float()
        return x

    def _load_samples(self, timestamps: List[int]) -> List[Sample]:
        samples = []
        for timestamp in timestamps:
            feature = self.feature_transform.load(timestamp, self.symbols, self.time_frames)
            label = self.label_transform.load(timestamp, self.symbols, self.time_frames)
            sample = Sample(timestamp=timestamp, feature=feature, label=label)
            samples.append(sample)

        return samples

    def __len__(self):
        return int(self.stop_date.timestamp() - self.start_date.timestamp()) // self.sampling_time_frame

    def __getitem__(self, item: Union[int, List[int]]) -> List[Sample]:
        if isinstance(item, int):
            return self._load_samples(timestamps=[item])
        elif isinstance(item, list):
            return self._load_samples(timestamps=item)
        else:
            raise ValueError("item parameter must be int (single timestamp) or list of ints (multiple timestamps).")
