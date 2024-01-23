from datetime import datetime
from typing import List, Tuple, Type

import torch
from rich.progress import Progress

from ._module import Module
from ..criterion import MSE
from ..cross_validation import CrossValidation
from ..data_loader import DataLoader
from ..dtype import Dataset, Fold
from ..lr_scheduler import StepLR
from ..model import FeedForward, Model
from ..model import AutoEncoder
from ..optimizer import Adam
from ..trainer import Trainer


class EncoderModule(Module):
    def __init__(self, dim_symbol: int, dim_time_frame: int, dim_sequence: int, dim_feature, dim_latent: int,
                 num_hidden_layers: int = 4, batch_norm: bool = False, model: Type[Model] = FeedForward):
        trainer = Trainer(
            cross_validation=CrossValidation(train_percentage=60, dev_percentage=20),
            data_loader=DataLoader(post_load_fn=self.__post_load_fn),
            criterion=MSE(),
            optimizer=Adam(lr=1e-3, weight_decay=1e-4),
            lr_scheduler=StepLR(step_size=1, gamma=0.5),
            gradient_clipping_threshold=1,
            log_fn=self.__log_fn
        )
        super().__init__(trainer=trainer)

        self.__dim_symbol: int = dim_symbol
        self.__dim_time_frame: int = dim_time_frame
        self.__dim_sequence: int = dim_sequence
        self.__dim_feature: int = dim_feature
        self.__dim_latent: int = dim_latent

        dim_flat_feature = dim_symbol * dim_time_frame * dim_feature
        self.__auto_encoder: AutoEncoder = AutoEncoder(
            dim_sequence=dim_sequence,
            dim_feature=dim_flat_feature,
            dim_output=dim_latent,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            encoder=model,
        )

    @property
    def dim_symbol(self) -> int:
        return self.__dim_symbol

    @property
    def dim_time_frame(self) -> int:
        return self.__dim_time_frame

    @property
    def dim_sequence(self) -> int:
        return self.__dim_sequence

    @property
    def dim_feature(self) -> int:
        return self.__dim_feature

    @property
    def dim_latent(self) -> int:
        return self.__dim_latent

    @property
    def auto_encoder(self) -> AutoEncoder:
        return self.__auto_encoder

    def forward(self, x: torch.tensor) -> torch.tensor:
        latent_x = self.auto_encoder.encode(x)  # dims (Batch, Latent Feature)
        return latent_x

    def optimize(self, dataset: Dataset, epochs: int, batch_size: int, progress: Progress = None):
        self.trainer.optimize(dataset, self.auto_encoder, epochs, batch_size, progress)

    def predict(self, dataset: Dataset, timestamps: List[int]) -> List:
        pass

    def __post_load_fn(self, x: torch.tensor, y: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        bsatf = x.permute(0, 3, 1, 2, 4).contiguous()  # dims (Batch, Sequence, Asset, Time frame, Feature)
        bsf = bsatf.view(bsatf.shape[0], bsatf.shape[1], -1)  # dims (Batch, Sequence, Asset * Time frame * Feature)
        return bsf, bsf

    def __log_fn(self, fold: Fold, epoch: int, epochs: int, epoch_time: float) -> None:
        elapsed_time = epoch_time * epoch
        total_time = elapsed_time * (epochs / epoch)
        remaining_time = total_time - elapsed_time
        elapsed_time_str = datetime.strftime(datetime.utcfromtimestamp(elapsed_time), '%H:%M:%S')
        remaining_time_str = datetime.strftime(datetime.utcfromtimestamp(remaining_time), '%H:%M:%S')
        total_time_str = datetime.strftime(datetime.utcfromtimestamp(total_time), '%H:%M:%S')

        # epoch metrics
        train_metrics = fold.epoch_to_train_metrics[epoch]
        val_metrics = fold.epoch_to_validation_metrics[epoch]
        test_metrics = fold.epoch_to_test_metrics[epoch]

        # best metrics
        best_train_metrics = fold.best_train_metrics
        best_val_metrics = fold.best_validation_metrics
        best_test_metrics = fold.best_test_metrics

        # features
        train_x = train_metrics.y
        val_x = val_metrics.y
        test_x = test_metrics.y

        # mean std features
        train_mstd_feature = torch.mean(torch.std(train_x, dim=0))
        val_mstd_feature = torch.mean(torch.std(val_x, dim=0))
        test_mstd_feature = torch.mean(torch.std(test_x, dim=0))

        # latents
        train_latent = self.auto_encoder.encode(x=train_x)
        val_latent = self.auto_encoder.encode(x=val_x)
        test_latent = self.auto_encoder.encode(x=test_x)

        # mean std latents
        train_mstd_latent = torch.mean(torch.std(train_latent, dim=0))
        val_mstd_latent = torch.mean(torch.std(val_latent, dim=0))
        test_mstd_latent = torch.mean(torch.std(test_latent, dim=0))

        # logs
        print("Epoch ({}/{}) (elapsed: {} remaining: {} total: {})"
              .format(epoch, epochs, elapsed_time_str, remaining_time_str, total_time_str))
        print("\t- {:<12} MSE: {:.4f}({:.4f}) | MSTD feature: {:.4f} | MSTD latent: {:.4f}"
              .format("Train", train_metrics.mse_loss, best_train_metrics.mse_loss,
                      train_mstd_feature, train_mstd_latent))
        print("\t- {:<12} MSE: {:.4f}({:.4f}) | MSTD feature: {:.4f} | MSTD latent: {:.4f}"
              .format("Validation", val_metrics.mse_loss, best_val_metrics.mse_loss,
                      val_mstd_feature, val_mstd_latent))
        print("\t- {:<12} MSE: {:.4f}({:.4f}) | MSTD feature: {:.4f} | MSTD latent: {:.4f}\n"
              .format("Test", test_metrics.mse_loss, best_test_metrics.mse_loss,
                      test_mstd_feature, test_mstd_latent))
