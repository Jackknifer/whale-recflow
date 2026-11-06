import torch
from torch import nn

from ..config import ModelConfig


# 每条目标输出一个 logit，sigmoid 和概率指标由评估过程计算。
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
