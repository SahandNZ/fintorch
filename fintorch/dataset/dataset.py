import math
import os
import pickle
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from rich.progress import Progress

from fintorch.data import Data
from fintorch.dataset.sample import Sample
from fintorch.defaults import DATASET_DIR
from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform
from fintorch.utils.directory import create_directory
from fintorch.utils.hash import static_list_hash


class Dataset:
    def __init__(self, symbols: str, time_frames: List[int], sequence_length: int, sampling_time_frame: int,
                 feature_transforms: List[FeatureTransform], label_transforms: List[LabelTransform]):

        self.__symbols: List[str] = symbols
        self.__time_frames: List[int] = time_frames
        self.__sequence_length: int = sequence_length
        self.__sampling_time_frame: int = sampling_time_frame
        self.__feature_transforms: List[FeatureTransform] = feature_transforms
        self.__label_transforms: List[LabelTransform] = label_transforms

        self.__directory: str = os.path.join(DATASET_DIR, str(self.__hash__()))
        create_directory(self.__directory)

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
    def sampling_time_frame(self) -> int:
        return self.__sampling_time_frame

    @property
    def feature_transforms(self) -> List[FeatureTransform]:
        return self.__feature_transforms

    @property
    def label_transforms(self) -> List[LabelTransform]:
        return self.__label_transforms

    @property
    def directory(self) -> str:
        return self.__directory

    def prepare(self, data: Data, start_date: str, stop_date: str, progress: Progress = None) -> None:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(start_date, str):
            stop_date = datetime.strptime(stop_date, "%Y-%m-%d")

        # create timestamps
        start_timestamp = math.ceil(start_date.timestamp() / self.sampling_time_frame) * self.sampling_time_frame
        stop_timestamp = math.floor(stop_date.timestamp() / self.sampling_time_frame) * self.sampling_time_frame
        timestamps = range(start_timestamp, stop_timestamp, self.sampling_time_frame)

        # create new task in rich progress bar for creating samples
        if progress is not None:
            desc = "[green]Creating Dataset samples"
            task = progress.add_task(description=desc, total=len(timestamps))

        # fit data to feature and label transforms
        self.fit(data=data, progress=progress)

        for timestamp in timestamps:
            self.create_sample(timestamp)
            if progress is not None:
                progress.update(task, advance=1)

    def preprocess(self, data: Data, timestamp: int) -> Sample:
        self.fit(data=data, progress=None)
        return self.create_sample(timestamp=timestamp)

    def fit(self, data: Data, progress: Progress = None):
        # fitting data to feature transforms
        for feature_transform in self.feature_transforms:
            feature_transform.fit(data=data, progress=progress)

        # fitting data to label transforms
        for label_transform in self.label_transforms:
            label_transform.fit(data=data, progress=progress)

    def create_sample(self, timestamp: int) -> Sample:
        file_path = os.path.join(self.directory, f"{timestamp}.pkl")

        # safe load timestamp_to_sample and if exists
        sample = None
        if os.path.exists(file_path):
            with open(file_path, "rb+") as file:
                try:
                    sample = pickle.load(file)
                except EOFError:
                    pass
        if sample is None or sample.feature is None or sample.label is None:
            # create feature and label numpy arrays
            feature_array = self._create_feature(timestamp=timestamp)
            label_array = self._create_label(timestamp=timestamp)

            # convert to tensor
            feature_tensor = torch.from_numpy(feature_array).float() if feature_array is not None else None
            label_tensor = torch.from_numpy(label_array).float() if label_array is not None else None
            sample = Sample(timestamp=timestamp, feature=feature_tensor, label=label_tensor)

            # save sample if it's label isn't none
            if sample.label is not None:
                with open(file_path, "wb+") as file:
                    pickle.dump(sample, file)

        return sample

    def _create_feature(self, timestamp: int) -> Union[np.array, None]:
        matsf = []  # dimensions (feature transform method, asset, time frame, sequence, feature)
        for feature_transform in self.feature_transforms:
            atsf = feature_transform.transform(timestamp=timestamp)
            if atsf is None:
                return None
            matsf.append(atsf)
        return np.array(matsf)

    def _create_label(self, timestamp: int) -> Union[np.array, None]:
        matl = []  # dimensions (label transform method, asset, time frame, one hot encoded label)
        for label_transform in self.label_transforms:
            atl = label_transform.transform(timestamp=timestamp)
            if atl is None:
                return None
            matl.append(atl)

        return np.array(matl)

    def __getitem__(self, item: Union[int, List[int]]) -> List[Sample]:
        if isinstance(item, int):
            samples = []
            for timestamp in item:
                sample = self.create_sample(timestamp=timestamp)
                samples.append(sample)
            return samples
        elif isinstance(item, list):
            sample = self.create_sample(timestamp=item)
            return sample
        else:
            raise ValueError("item parameter must be int (single timestamp) or list of ints (multiple timestamps).")

    def __hash__(self):
        hash_values = [
            static_list_hash(self.symbols),
            static_list_hash(self.time_frames),
            self.sequence_length,
            self.sampling_time_frame,
            static_list_hash(sorted([ft.short_name for ft in self.feature_transforms])),
            static_list_hash(sorted([lt.short_name for lt in self.label_transforms]))
        ]

        total_hash = static_list_hash(hash_values)
        return total_hash
