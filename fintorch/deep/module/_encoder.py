import os.path
import pickle
from typing import List, Tuple, Type

import torch
from rich.progress import Progress

from ._module import Module
from ..criterion import MSE
from ..cross_validation import CrossValidation
from ..data_loader import DataLoader
from ..dtype import Dataset
from ..lr_scheduler import StepLR
from ..model import FeedForward, Model
from ..model import AutoEncoder
from ..optimizer import Adam
from ..trainer import Trainer
from ...setting import MODULE_DIR
from ...utils.directory import create_directory
from ...utils.hash import static_list_hash


class EncoderModule(Module):
    def __init__(self, dataset: Dataset, dim_latent: int = 128, num_hidden_layers: int = 2, batch_norm: bool = True,
                 encoder: Type[Model] = FeedForward):
        dim_flat_feature = len(dataset.symbols) * len(dataset.time_frames) * len(dataset.feature_transform.features)
        model = AutoEncoder(
            dim_sequence=dataset.feature_transform.sequence_length,
            dim_feature=dim_flat_feature,
            dim_output=dim_latent,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            encoder=encoder,
        )

        trainer = Trainer(
            cross_validation=CrossValidation(train_percentage=60, dev_percentage=20),
            data_loader=DataLoader(post_load_fn=self.__post_load_fn),
            criterion=MSE(),
            optimizer=Adam(lr=1e-3, weight_decay=1e-4),
            lr_scheduler=StepLR(step_size=1, gamma=0.9),
            gradient_clipping_threshold=1
        )

        super().__init__(dataset=dataset, model=model, trainer=trainer)
        self.__auto_encoder: AutoEncoder = model
        self.__dim_latent: int = dim_latent

    @property
    def dim_symbol(self) -> int:
        return len(self.dataset.symbols)

    @property
    def dim_time_frame(self) -> int:
        return len(self.dataset.time_frames)

    @property
    def dim_sequence(self) -> int:
        return self.dataset.sequence_length

    @property
    def dim_feature(self) -> int:
        return len(self.dataset.feature_transform.features)

    @property
    def auto_encoder(self) -> AutoEncoder:
        return self.__auto_encoder

    @property
    def dim_latent(self) -> int:
        return self.__dim_latent

    @property
    def directory(self) -> str:
        symbols_hash = static_list_hash(self.dataset.symbols)
        time_frames_hash = static_list_hash(self.dataset.time_frames)
        dataset_hash = static_list_hash([symbols_hash, time_frames_hash, self.dim_sequence, self.dim_feature])

        directory = os.path.join(MODULE_DIR, "encoder", self.auto_encoder.encoder.short_name,
                                 self.dataset.feature_transform.short_name, str(dataset_hash))
        create_directory(directory)

        return directory

    @property
    def path(self) -> str:
        return os.path.join(self.directory, f"dim-latent-{self.dim_latent}.pkl")

    def optimize_and_store(self, epochs: int, batch_size: int, progress: Progress = None):
        folds = self.trainer.optimize(dataset=self.dataset, model=self.model, epochs=epochs, batch_size=batch_size,
                                      progress=progress)

        model_state_dict = folds[0].best_validation_model_state_dict
        with open(self.path, "wb+") as file:
            pickle.dump(model_state_dict, file)

    def predict(self, timestamps: List[int]):
        with open(self.path, "rb") as file:
            model_state_dict = pickle.load(file)
        self.model.load_state_dict(model_state_dict)

    def __post_load_fn(self, x: torch.tensor, y: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        bsatf = x.permute(0, 3, 1, 2, 4).contiguous()  # dims (Batch, Sequence, Asset, Time frame, Feature)
        bsf = bsatf.view(bsatf.shape[0], bsatf.shape[1], -1)  # dims (Batch, Sequence, Asset * Time frame * Feature)
        return bsf, bsf
