from dataclasses import dataclass

import torch
from torch import nn

from ..config import ModelConfig

NUMERICAL_GROUP_SIZES = (2, 4, 2)
HISTORY_RELATIVE_AGE_DIM = 1


@dataclass(frozen=True)
class VocabularySizes:
    categorical: dict[str, int]
    feedback: dict[str, int]

    def as_mapping(self) -> dict[str, int]:
        return dict(self.categorical)


class NumericalTokenizer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self._mlp(values)


class InputEmbedding(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        vocabulary: VocabularySizes,
        categorical_fields: tuple[str, ...],
        history_video_fields: tuple[str, ...],
        history_feedback_fields: tuple[str, ...],
    ) -> None:
        super().__init__()
        dim = config.embedding_dim
        hidden = config.numerical_hidden_dim
        sizes = vocabulary.as_mapping()
        missing = [field for field in categorical_fields if field not in sizes]
        if missing:
            raise KeyError(f"类别字典缺少字段 {missing}")
        self._fields = tuple(categorical_fields)
        self._history_video_fields = tuple(history_video_fields)
        self._history_feedback_fields = tuple(history_feedback_fields)
        self._categorical = nn.ModuleDict(
            {field: nn.Embedding(sizes[field], dim) for field in self._fields}
        )
        self._feedback = nn.ModuleDict(
            {
                field: nn.Embedding(
                    int(vocabulary.feedback[field]), dim
                )
                for field in self._history_feedback_fields
            }
        )
        self._numerical = nn.ModuleList(
            NumericalTokenizer(size, hidden, dim) for size in NUMERICAL_GROUP_SIZES
        )
        history_input = (
            len(self._history_video_fields) + len(self._history_feedback_fields)
        ) * dim + HISTORY_RELATIVE_AGE_DIM
        self._history_mlp = None
        if config.name != "hstu" or config.hstu.ranking_input == "pooled":
            self._history_mlp = nn.Sequential(
                nn.Linear(history_input, config.sequence_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.sequence_hidden_dim, dim),
            )
        self._history_input_dim = history_input

    @property
    def history_input_dim(self) -> int:
        return self._history_input_dim

    def nonsequence_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = [self._categorical[field](batch[field]) for field in self._fields]
        for index, name in enumerate(("duration", "request_time", "video_age")):
            tokens.append(self._numerical[index](batch[name]))
        return torch.stack(tokens, dim=1)

    def sequence_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._history_mlp is None:
            raise ValueError("交替排序输入使用独立的物品和行为 MLP")
        parts = [
            self._categorical[field](batch[f"history_{field}"])
            for field in self._history_video_fields
        ]
        parts.extend(
            self._feedback[field](batch[f"history_{field}"])
            for field in self._history_feedback_fields
        )
        parts.append(batch["history_age"].unsqueeze(-1))
        concatenated = torch.cat(parts, dim=-1)
        return self._history_mlp(concatenated)

    def ranking_components(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        items = [self._categorical[field](batch[f"history_{field}"]) for field in self._history_video_fields]
        items.append(batch["history_age"].unsqueeze(-1))
        actions = [self._feedback[field](batch[f"history_{field}"]) for field in self._history_feedback_fields]
        candidate = self.nonsequence_embedding(batch).flatten(start_dim=1)
        return torch.cat(items, dim=-1), torch.cat(actions, dim=-1), candidate

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonsequence = self.nonsequence_embedding(batch)
        sequence = self.sequence_embedding(batch)
        mask = batch["history_mask"].unsqueeze(-1).to(sequence.dtype)
        return nonsequence, sequence * mask
