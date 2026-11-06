import torch
from torch import nn

from ...config import ModelConfig
from ..base import BaseModel
from ..embedding import VocabularySizes
from ..head import PredictionHead
from ..hstu.block import HSTUBlock
from ..wukong.block import WukongBlock
from .fusion import FusionBlock


# 每层分别更新两类特征，再将历史信息融合到非序列 Token。
class WhaleModel(BaseModel):
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
        input_tokens = config.input_tokens
        self._wukong_blocks = nn.ModuleList()
        self._hstu_blocks = nn.ModuleList()
        self._fusion_blocks = nn.ModuleList()
        # 每层使用独立的 Wukong、HSTU 和 Fusion 参数。
        for layer in range(config.num_layers):
            self._wukong_blocks.append(
                WukongBlock(
                    config, input_tokens if layer == 0 else config.nonsequence_tokens
                )
            )
            self._hstu_blocks.append(HSTUBlock(config, max_history_length))
            self._fusion_blocks.append(FusionBlock(config))
        self._head = PredictionHead(
            config, config.nonsequence_tokens * config.embedding_dim
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        nonsequence, sequence = self._prepare(batch)
        mask = batch["history_mask"]
        time_bucket = batch["time_bucket"]
        causal_mask = batch["causal_mask"]
        for wukong, hstu, fusion in zip(
            self._wukong_blocks, self._hstu_blocks, self._fusion_blocks
        ):
            nonsequence = wukong(nonsequence)
            sequence = hstu.forward_masked(sequence, mask, time_bucket, causal_mask)
            nonsequence = fusion(nonsequence, sequence, mask)
        # 最后一层的非序列表示汇总了逐层融合的信息。
        return self._head(self._flatten_nonsequence(nonsequence))
