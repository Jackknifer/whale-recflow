import torch

from ...config import ModelConfig
from ..base import BaseModel
from ..embedding import VocabularySizes
from ..head import PredictionHead
from .block import WukongStack


class WukongModel(BaseModel):
    def __init__(
        self,
        config: ModelConfig,
        vocabulary: VocabularySizes,
        max_history_length: int,
        categorical_fields: tuple[str, ...],
        history_video_fields: tuple[str, ...],
        history_feedback_fields: tuple[str, ...],
    ) -> None:
        super().__init__(
            config,
            vocabulary,
            max_history_length,
            categorical_fields,
            history_video_fields,
            history_feedback_fields,
        )
        self._stack = WukongStack(config, config.input_tokens)
        self._head = PredictionHead(
            config, config.nonsequence_tokens * config.embedding_dim
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        nonsequence = self._embedding.nonsequence_embedding(batch)
        representation = self._stack(nonsequence)
        return self._head(self._flatten_nonsequence(representation))
