import math
import os
import pickle
import time
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from rich.progress import Progress

from fintorch.data import Data
from fintorch.dataset.sample import Sample
from fintorch.defaults import DATA_DIR
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

    def prepare(self, data: Data, start_date: str, progress: Progress = None) -> None:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        self._fit_data(data=data, progress=progress)

        start_timestamp = max(max(df.index.min() for df in ft.data.dataframes) for ft in self.feature_transforms)
        stop_timestamp = min(min(df.index.max() for df in lt.data.dataframes) for lt in self.label_transforms)

        start_timestamp = max(start_date.timestamp(), start_timestamp + self.sequence_length * max(self.time_frames))
        start_timestamp = math.ceil(start_timestamp / self.sampling_time_frame) * self.sampling_time_frame
        stop_timestamp = math.floor(stop_timestamp / self.sampling_time_frame) * self.sampling_time_frame

        timestamps = range(start_timestamp, stop_timestamp + 1, self.sampling_time_frame)

        # create new task in rich progress bar for creating samples
        if progress is not None:
            desc = "[green]Creating Dataset samples"
            task = progress.add_task(description=desc, total=len(timestamps))

        for timestamp in timestamps:
            self._create_sample(timestamp)
            if progress is not None:
                progress.update(task, advance=1)

    def preprocess(self, data: Data, timestamp: int) -> Sample:
        self._fit_data(data=data, progress=None)
        return self._create_sample(timestamp=timestamp)

    def _fit_data(self, data: Data, progress: Progress):
        # fitting data to feature transforms
        for feature_transform in self.feature_transforms:
            feature_transform.fit(data=data, progress=progress)

        # fitting data to label transforms
        for label_transform in self.label_transforms:
            label_transform.fit(data=data, progress=progress)

    def _create_sample(self, timestamp: int) -> Sample:
        file_path = os.path.join(self.directory, str(timestamp) + ".pkl")

        # load sample and return it if exists
        if os.path.exists(file_path) and 0 < os.path.getsize(file_path):
            with open(file_path, "rb") as file:
                sample = pickle.load(file)
            return sample

        # create feature and label arrays
        start_time = time.time()
        feature_array = self._create_feature(timestamp=timestamp)
        print("creating feature takes: {:.3}".format(time.time() - start_time))
        start_time = time.time()
        label_array = self._create_label(timestamp=timestamp)
        print("creating label takes: {:.3}".format(time.time() - start_time))

        # convert to tensor
        feature_tensor = torch.from_numpy(feature_array).float() if feature_array is not None else None
        label_tensor = torch.from_numpy(label_array).float() if label_array is not None else None
        sample = Sample(timestamp=timestamp, feature=feature_tensor, label=label_tensor)

        # store and return sample
        with open(file_path, "wb+") as file:
            pickle.dump(sample, file)

        return sample

    def _create_feature(self, timestamp: int) -> Union[np.array, None]:
        matsf_start_time = time.time()

        matsf = []  # dimensions (feature transform method, asset, time frame, sequence, feature)
        for feature_transform in self.feature_transforms:

            atsf_start_time = time.time()
            atsf = []  # dimensions (asset, time frame, sequence, feature)
            for symbol in self.symbols:
                tsf_start_time = time.time()

                tsf = []  # dimensions (time frame, sequence, feature)
                for time_frame in self.time_frames:
                    sf_start_time = time.time()
                    sf = feature_transform.transform(timestamp, symbol, time_frame, self.sequence_length)
                    print("creating sf takes: {:.3f}".format(time.time() - sf_start_time))
                    if sf is None:
                        return None

                    tsf.append(sf)

                print("creating tsf takes: {:.3f}".format(time.time() - tsf_start_time))

                atsf.append(tsf)

            print("creating atsf takes: {:.3f}".format(time.time() - atsf_start_time))

            matsf.append(atsf)

        print("creating matsf takes: {:.3f}".format(time.time() - matsf_start_time))

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
            static_list_hash(self.symbols),
            static_list_hash(self.time_frames),
            self.sequence_length,
            self.sampling_time_frame,
            static_list_hash(sorted([ft.short_name for ft in self.feature_transforms])),
            static_list_hash(sorted([lt.short_name for lt in self.label_transforms]))
        ]

        total_hash = static_list_hash(hash_values)
        return total_hash
