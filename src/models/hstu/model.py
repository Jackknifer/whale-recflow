import torch

from ...config import ModelConfig
from ..base import BaseModel
from ..embedding import VocabularySizes
from ..head import PredictionHead
from .block import HSTUStack
from .ranking_input import RankingInput


# HSTU-only 编码交替序列，使用当前候选位置的表示完成预测。
class HSTUModel(BaseModel):
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
        self._ranking_input = RankingInput(
            config,
            max_history_length,
            len(history_video_fields),
            len(history_feedback_fields),
        )
        self._stack = HSTUStack(config, self._ranking_input.max_tokens)
        self._head = PredictionHead(config, config.embedding_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ranking = self._ranking_input(batch, self._embedding)
        encoded = self._stack(
            ranking.tokens, ranking.mask, ranking.time_bucket, ranking.causal_mask
        )
        # 每个样本的历史长度不同，需分别读取对应候选位置。
        rows = torch.arange(encoded.size(0), device=encoded.device)
        return self._head(encoded[rows, ranking.candidate_positions])
