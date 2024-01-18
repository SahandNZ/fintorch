import os
import math
import pickle
from datetime import datetime
from typing import List, Union

import numpy as np
from rich.progress import Progress

from ._sample import Sample
from ..transform.feature import FeatureTransform
from ..transform.label import LabelTransform
from ...dtype import Data
from ...enum import TimeFrame
from ...setting import DATASET_DIR, FILE_COMPRESS_FACTOR
from ...utils.directory import create_directory
from ...utils.hash import static_list_hash
from ...utils.timestamp import create_timestamps


class Dataset:
    def __init__(self, start_date: str, stop_date: str, sampling_time_frame: TimeFrame, symbols: List[str],
                 time_frames: List[int], sequence_length: int, feature_transform: FeatureTransform,
                 label_transform: LabelTransform):
        self.__start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.__stop_date = datetime.strptime(stop_date, "%Y-%m-%d")
        self.__sampling_time_frame: TimeFrame = sampling_time_frame
        self.__timestamps: List[int] = create_timestamps(self.start_date, self.stop_date, self.sampling_time_frame)

        self.__symbols: List[str] = symbols
        self.__time_frames: List[int] = time_frames
        self.__sequence_length: int = sequence_length
        self.__feature_transform: FeatureTransform = feature_transform
        self.__label_transform: LabelTransform = label_transform

        self.__directory: str = os.path.join(DATASET_DIR, str(self.__hash__()))
        create_directory(self.__directory)

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
    def time_frames(self) -> List[int]:
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

    @property
    def directory(self) -> str:
        return self.__directory

    def prepare(self, data: Data, progress: Progress = None) -> None:
        # create new task in rich progress bar for creating samples
        if progress is not None:
            desc = "[green]Creating Dataset samples"
            task = progress.add_task(description=desc, total=len(self.timestamps))

        # fit data to feature and label transforms
        self.fit(data=data, progress=progress)

        for timestamp in self.timestamps:
            self.load_sample(timestamp=timestamp)
            if progress is not None:
                progress.update(task, advance=1)

    def preprocess(self, data: Data, timestamp: int) -> Sample:
        self.fit(data=data, progress=None)
        return self.load_sample(timestamp=timestamp)

    def fit(self, data: Data, progress: Progress = None):
        self.feature_transform.fit(data=data, progress=progress)
        self.label_transform.fit(data=data, progress=progress)

    def load_sample(self, timestamp: int, create_missing: bool = True) -> Sample:
        file_compress_factor = FILE_COMPRESS_FACTOR * self.sampling_time_frame
        file_name = math.floor(timestamp / file_compress_factor) * file_compress_factor
        file_path = os.path.join(self.directory, f"{file_name}.pkl")

        # safe load sample if exists
        if os.path.exists(file_path):
            with open(file_path, "rb+") as file:
                try:
                    timestamp_to_sample = pickle.load(file)
                except EOFError:
                    timestamp_to_sample = {}
        else:
            timestamp_to_sample = {}

        sample = timestamp_to_sample[timestamp] if timestamp in timestamp_to_sample else None

        # create feature and labels
        if create_missing:
            if sample is None:
                feature = self.feature_transform.transform(timestamp=timestamp)
            else:
                feature = sample.feature

            if sample is None or sample.label is None:
                label = self.label_transform.transform(timestamp=timestamp)
            else:
                label = sample.label

            # update sample on storage if it's none or its label was none
            if sample is None or sample.label is None:
                sample = Sample(timestamp=timestamp, feature=feature, label=label)
                timestamp_to_sample[timestamp] = sample
                with open(file_path, "wb+") as file:
                    pickle.dump(timestamp_to_sample, file)

        return sample

    def _load_samples(self, timestamps: List[int]) -> List[Sample]:
        samples = []
        for timestamp in timestamps:
            sample = self.load_sample(timestamp=timestamp, create_missing=False)
            samples.append(sample)

        return samples

    def __len__(self):
        samples = int(self.stop_date.timestamp() - self.start_date.timestamp()) // self.sampling_time_frame
        return samples

    def __getitem__(self, item: Union[int, List[int]]) -> List[Sample]:
        if isinstance(item, int):
            return self._load_samples(timestamps=[item])
        elif isinstance(item, list):
            return self._load_samples(timestamps=item)
        else:
            raise ValueError("item parameter must be int (single timestamp) or list of ints (multiple timestamps).")

    def __hash__(self):
        hash_values = [
            static_list_hash(self.symbols),
            static_list_hash(self.time_frames),
            self.sequence_length,
            self.sampling_time_frame,
            self.feature_transform.name,
            self.label_transform.name
        ]

        total_hash = static_list_hash(hash_values)
        return total_hash
