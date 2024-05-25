from torch import nn

from ._feed_forward import SFFeedForward
from ._model import SFModel


class SFTransformer(SFModel):
    def __init__(
            self,
            dropout: float,
            batch_norm: bool,
            activation_fn: nn.Module,

            dim_input_sequence: int,
            dim_input_feature: int,
            dim_output_sequence: int,
            dim_output_feature: int,

            num_hidden_layers: int,
            num_head: int = 2,
            num_layers: int = 2
    ):
        super().__init__(
            name="SF Transformer",
            short_name="SF TRAN",
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn,

            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,

            num_hidden_layers=num_hidden_layers,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_input_feature,
            nhead=num_head,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ff = SFFeedForward(
            dropout=dropout,
            batch_norm=batch_norm,
            activation_fn=activation_fn,

            dim_input_sequence=dim_input_sequence,
            dim_input_feature=dim_input_feature,
            dim_output_sequence=dim_output_sequence,
            dim_output_feature=dim_output_feature,

            num_hidden_layers=num_hidden_layers,
        )

    def _forward(self, x):
        f = self.encoder(x)
        y_hat = self.ff(f)
        return y_hat
