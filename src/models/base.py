import torch
from torch import nn

from ..config import ModelConfig
from .embedding import InputEmbedding, VocabularySizes


# 共用输入编码和输出形状约定，具体模型实现各自的 forward。
class BaseModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        vocabulary: VocabularySizes,
        max_history_length: int,
        categorical_fields: tuple[str, ...],
        history_video_fields: tuple[str, ...],
        history_feedback_fields: tuple[str, ...],
    ) -> None:
        super().__init__()
        self._config = config
        self._max_history_length = max_history_length
        self._embedding = InputEmbedding(
            config,
            vocabulary,
            categorical_fields,
            history_video_fields,
            history_feedback_fields,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def _prepare(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._embedding(batch)

    def _flatten_nonsequence(self, nonsequence: torch.Tensor) -> torch.Tensor:
        # 保留批次维度，将非序列 Token 连接为预测头的输入。
        return nonsequence.reshape(nonsequence.size(0), -1)

    @property
    def history_input_dim(self) -> int:
        return self._embedding.history_input_dim
