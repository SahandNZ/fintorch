import torch.nn as nn

from ._feed_forward import FeedForward
from ._model import Model


class LSTM(Model):
    def __init__(
            self,
            dim_input_sequence: int,
            dim_input_feature: int,
            dim_output_sequence: int,
            dim_output_feature: int,
            num_hidden_layers: int,
            dropout: float,
            batch_norm: bool,
            activation_fn: nn.Module = None,
            dim_hidden: int = 4,
            num_layers: int = 2
    ):
        super().__init__(
            name="Long Short-Term Memory",
            short_name="LSTM",
            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn
        )

        self.lstm = nn.LSTM(
            input_size=dim_input_feature,
            hidden_size=dim_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.ff = FeedForward(
            dim_input_sequence=1,
            dim_input_feature=dim_hidden,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,
            num_hidden_layers=num_hidden_layers,
            batch_norm=batch_norm,
            dropout=dropout,
            activation_fn=activation_fn
        )

    def _forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        y_hat = self.ff(output)
        return y_hat
