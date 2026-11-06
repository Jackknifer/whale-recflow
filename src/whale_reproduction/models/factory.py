from dataclasses import replace

import torch
from torch import nn

from ..config import ModelConfig
from .embedding import InputEmbedding, VocabularySizes
from .fusion import FusionBlock
from .head import PredictionHead
from .hstu import HSTUBlock, HSTUStack
from .ranking_input import RankingInput
from .wukong import WukongBlock, WukongStack

MODEL_NAMES = ("wukong", "hstu", "whale")


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
        return nonsequence.reshape(nonsequence.size(0), -1)

    @property
    def history_input_dim(self) -> int:
        return self._embedding.history_input_dim


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
        self._ranking_input = RankingInput(config, max_history_length, len(history_video_fields), len(history_feedback_fields))
        self._stack = HSTUStack(config, self._ranking_input.max_tokens)
        self._head = PredictionHead(config, config.embedding_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ranking = self._ranking_input(batch, self._embedding)
        encoded = self._stack(ranking.tokens, ranking.mask, ranking.time_bucket, ranking.causal_mask)
        rows = torch.arange(encoded.size(0), device=encoded.device)
        return self._head(encoded[rows, ranking.candidate_positions])


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
        for layer in range(config.num_layers):
            self._wukong_blocks.append(
                WukongBlock(config, input_tokens if layer == 0 else config.nonsequence_tokens)
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
        return self._head(self._flatten_nonsequence(nonsequence))


MODEL_REGISTRY = {
    "wukong": WukongModel,
    "hstu": HSTUModel,
    "whale": WhaleModel,
}


def build_model(
    name: str,
    config: ModelConfig,
    vocabulary: VocabularySizes,
    max_history_length: int,
    categorical_fields: tuple[str, ...],
    history_video_fields: tuple[str, ...],
    history_feedback_fields: tuple[str, ...],
) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型 {name}，可选值为 {MODEL_NAMES}")
    return MODEL_REGISTRY[name](
        replace(config, name=name),
        vocabulary,
        max_history_length,
        categorical_fields,
        history_video_fields,
        history_feedback_fields,
    )
