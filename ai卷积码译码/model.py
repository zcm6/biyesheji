from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except (ImportError, OSError):  # PyTorch is only required when training or running this model.
    torch = None
    nn = None


@dataclass(frozen=True)
class RNNDecoderConfig:
    input_size: int = 2
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    bidirectional: bool = True


if nn is not None:

    class ConvCodeRNNDecoder(nn.Module):
        """Small BiGRU decoder for the project's rate-1/2 convolutional code.

        The model follows the Sequential-RNN-Decoder idea: every time step
        receives the two coded channel observations and predicts one source bit.
        Input shape: [batch, steps, 2].
        Output shape: [batch, steps], as logits for BCEWithLogitsLoss.
        """

        def __init__(self, config: RNNDecoderConfig | None = None) -> None:
            super().__init__()
            self.config = config or RNNDecoderConfig()
            gru_dropout = self.config.dropout if self.config.num_layers > 1 else 0.0
            self.rnn = nn.GRU(
                input_size=self.config.input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                batch_first=True,
                dropout=gru_dropout,
                bidirectional=self.config.bidirectional,
            )
            direction_factor = 2 if self.config.bidirectional else 1
            self.readout = nn.Linear(self.config.hidden_size * direction_factor, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y, _ = self.rnn(x)
            return self.readout(y).squeeze(-1)


else:
    ConvCodeRNNDecoder = None


def require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("AI卷积码译码模型需要 PyTorch，请先安装 torch。")


def config_to_dict(config: RNNDecoderConfig) -> dict[str, int | float | bool]:
    return {
        "input_size": config.input_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "bidirectional": config.bidirectional,
    }


def config_from_dict(data: dict) -> RNNDecoderConfig:
    return RNNDecoderConfig(
        input_size=int(data.get("input_size", 2)),
        hidden_size=int(data.get("hidden_size", 64)),
        num_layers=int(data.get("num_layers", 2)),
        dropout=float(data.get("dropout", 0.0)),
        bidirectional=bool(data.get("bidirectional", True)),
    )
