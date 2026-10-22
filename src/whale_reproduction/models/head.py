import torch
from torch import nn

from ..config import ModelConfig


class PredictionHead(nn.Module):
    def __init__(self, config: ModelConfig, input_dim: int) -> None:
        super().__init__()
        head = config.head
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in head.hidden_dims:
            layers.append(nn.Linear(previous, hidden))
            layers.append(nn.SiLU())
            if head.dropout > 0:
                layers.append(nn.Dropout(head.dropout))
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self._mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self._mlp(features).squeeze(-1)
