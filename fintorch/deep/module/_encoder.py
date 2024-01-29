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
from ...dtype import Data
from ...setting import MODULE_DIR
from ...utils.directory import create_directory
from ...utils.hash import static_list_hash


class EncoderModule(Module):
    def __init__(self, dataset: Dataset, dim_latent: int = 128, num_hidden_layers: int = 2, batch_norm: bool = True,
                 dropout: float = 0.5, encoder: Type[Model] = FeedForward):
        dim_flat_feature = len(dataset.symbols) * len(dataset.time_frames) * len(dataset.feature_transform.features)
        auto_encoder = AutoEncoder(
            dim_sequence=dataset.feature_transform.sequence_length,
            dim_feature=dim_flat_feature,
            dim_output=dim_latent,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            dropout=dropout,
            encoder=encoder,
        )

        trainer = Trainer(
            cross_validation=CrossValidation(train_percentage=60, dev_percentage=20),
            data_loader=DataLoader(post_load_fn=self.__post_load_fn),
            criterion=MSE(),
            optimizer=Adam(lr=1e-3, weight_decay=1e-3),
            lr_scheduler=StepLR(step_size=1, gamma=0.9),
            gradient_clipping_threshold=1
        )

        super().__init__(dataset=dataset, model=auto_encoder, trainer=trainer)
        self.__auto_encoder: AutoEncoder = auto_encoder
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

    def predict(self, data: Data, timestamps: List[int]) -> torch.Tensor:
        # load model and set it to eval mode
        with open(self.path, "rb") as file:
            model_state_dict = pickle.load(file)
        self.auto_encoder.load_state_dict(model_state_dict)
        self.auto_encoder.eval()

        # load dataset and predict values
        latent_x = []
        for timestamp in timestamps:
            feature = self.dataset.preprocess(data=data, timestamp=timestamp)
            x = torch.from_numpy(feature).unsqueeze(0)
            flat_x = self.__flatten_x(x=x)
            with torch.no_grad():
                encoded_x = self.auto_encoder.encode(flat_x)
            latent_x.append(encoded_x)

        latent_x = torch.cat(latent_x, dim=0)
        return latent_x

    def __post_load_fn(self, x: torch.tensor, y: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        bsf = self.__flatten_x(x)
        return bsf, bsf

    def __flatten_x(self, x: torch.Tensor) -> torch.Tensor:
        bsatf = x.permute(0, 3, 1, 2, 4).contiguous()  # dims (Batch, Sequence, Asset, Time frame, Feature)
        bsf = torch.flatten(bsatf, start_dim=2)  # dims (Batch, Sequence, Asset * Time frame * Feature)

        return bsf
