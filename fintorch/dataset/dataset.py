import math
import os
import pickle
from typing import List, Union

import numpy as np
import torch
from rich.progress import Progress

from fintorch.data import Data
from fintorch.dataset.smaple import Sample
from fintorch.defaults import DATA_DIR
from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.label.transform import LabelTransform
from fintorch.utils.directory import create_directory
from fintorch.utils.hash import list_hash


class Dataset:
    def __init__(self, symbols: str, time_frames: List[int], sequence_length: int, sampling_time_frame: int,
                 feature_transforms: List[FeatureTransform], label_transforms: List[LabelTransform]):

        self.__symbols: List[str] = symbols
        self.__time_frames: List[int] = time_frames
        self.__sequence_length: int = sequence_length
        self.__sampling_time_frame: int = sampling_time_frame
        self.__feature_transforms: List[FeatureTransform] = feature_transforms
        self.__label_transforms: List[LabelTransform] = label_transforms

        self.__directory: str = os.path.join(DATA_DIR, "dataset", str(self.__hash__()))
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

    def prepare(self, data: Data, progress: Progress = None):
        self._fit_data(data=data, progress=progress)

        start_timestamp = max(max(df.index.min() for df in ft.data.dataframes) for ft in self.feature_transforms)
        stop_timestamp = min(min(df.index.max() for df in lt.data.dataframes) for lt in self.label_transforms)

        start_timestamp = start_timestamp + self.sequence_length * max(self.time_frames)
        start_timestamp = math.ceil(start_timestamp / self.sampling_time_frame) * self.sampling_time_frame
        stop_timestamp = math.floor(stop_timestamp / self.sampling_time_frame) * self.sampling_time_frame
        timestamps = range(start_timestamp, stop_timestamp + 1, self.sampling_time_frame)

        self._create_samples(timestamps=timestamps, progress=progress)

    def preprocess(self, data: Data, timestamp: int) -> Sample:
        self._fit_data(data=data, progress=None)
        return self._create_sample(timestamp=timestamp)

    def _fit_data(self, data: Data, progress: Progress):
        # create new task in rich progress bar for fitting data to feature transforms
        if progress is not None:
            desc = "[green]Fit data to feature transforms"
            task = progress.add_task(description=desc, total=len(self.feature_transforms))

        # fitting data to feature transforms
        for feature_transform in self.feature_transforms:
            feature_transform.fit(data=data)
            if progress is not None:
                progress.update(task, advance=1)

        # create new task in rich progress bar for fitting data to label transforms
        if progress is not None:
            desc = "[green]Fit data to label transforms"
            task = progress.add_task(description=desc, total=len(self.label_transforms))

        # fitting data to label transforms
        for label_transform in self.label_transforms:
            label_transform.fit(data=data)
            if progress is not None:
                progress.update(task, advance=1)

    def _create_samples(self, timestamps: List[int], progress: Progress):
        # create new task in rich progress bar for creating samples
        if progress is not None:
            desc = "[green]Creating Dataset samples"
            task = progress.add_task(description=desc, total=len(timestamps))

        # iterate over timestamps to create and save missing samples
        for timestamp in timestamps:
            file_path = os.path.join(self.directory, str(timestamp) + ".pkl")
            if not os.path.exists(file_path) or 0 == os.path.getsize(file_path):
                sample = self._create_sample(timestamp=timestamp)
                if sample.feature is not None and sample.label is not None:
                    with open(file_path, "wb+") as file:
                        pickle.dump(sample, file)

            if progress is not None:
                progress.update(task, advance=1)

    def _create_sample(self, timestamp: int) -> Sample:
        feature_array = self._create_feature(timestamp=timestamp)
        label_array = self._create_label(timestamp=timestamp)

        # convert to tensor
        feature_tensor = torch.from_numpy(feature_array).float() if feature_array is not None else None
        label_tensor = torch.from_numpy(label_array).float() if label_array is not None else None

        return Sample(timestamp=timestamp, feature=feature_tensor, label=label_tensor)

    def _create_feature(self, timestamp: int) -> Union[np.array, None]:
        matsf = []  # dimensions (feature transform method, asset, time frame, sequence, feature)
        for feature_transform in self.feature_transforms:
            atsf = []  # dimensions (asset, time frame, sequence, feature)
            for symbol in self.symbols:
                tsf = []  # dimensions (time frame, sequence, feature)
                for time_frame in self.time_frames:
                    sf = feature_transform.transform(timestamp, symbol, time_frame, self.sequence_length)
                    if sf is None:
                        return None

                    tsf.append(sf)
                atsf.append(tsf)
            matsf.append(atsf)

        return np.array(matsf)

    def _create_label(self, timestamp: int) -> Union[np.array, None]:
        matl = []  # dimensions (label transform method, asset, time frame, one hot encoded label)
        for label_transform in self.label_transforms:
            atl = []  # dimensions (asset, time frame, one hot encoded label)
            for symbol in self.symbols:
                tl = []  # dimensions (time frame, one hot encoded label)
                for time_frame in self.time_frames:
                    label = label_transform.transform(timestamp=timestamp, symbol=symbol, time_frame=time_frame)
                    if label is None:
                        return None

                    tl.append(label)
                atl.append(tl)
            matl.append(atl)

        return np.array(matl)

    def _load_samples(self, timestamps: List[int]) -> List[Sample]:
        samples = []
        for timestamp in timestamps:
            file_path = os.path.join(self.directory, str(timestamp) + ".pkl")
            with open(file_path, "rb") as file:
                sample = pickle.load(file)
            samples.append(sample)

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
            list_hash(self.symbols),
            list_hash(self.time_frames),
            self.sequence_length,
            list_hash(sorted([ft.short_name for ft in self.feature_transforms])),
            list_hash(sorted([lt.short_name for lt in self.label_transforms]))
        ]

        total_hash = list_hash(hash_values)
        return total_hash
